from typing import Any
import numpy as np

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def serialize_question(stem: str, labels: tuple[str, ...], texts: tuple[str, ...]) -> str:
    require(len(labels) == len(texts) and len(labels) > 0, "choice labels/text misaligned")
    return "\n".join([f"Question: {stem}", "Choices:"] + [f"{a}: {b}" for a, b in zip(labels, texts)])

def semantic_encode(model: Any, tokenizer: Any, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, dict[str, int]]:
    import torch

    outputs = []
    maximum_tokens = 0
    minimum_tokens = None
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start:start + batch_size]
        encoded = tokenizer(batch_text, add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
        attention = encoded["attention_mask"]
        lengths = attention.sum(dim=1)
        maximum_tokens = max(maximum_tokens, int(lengths.max().item()))
        minimum_tokens = int(lengths.min().item()) if minimum_tokens is None else min(minimum_tokens, int(lengths.min().item()))
        require(int(lengths.max().item()) <= int(model.config.max_position_embeddings), "semantic sequence exceeds model context")
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded, return_dict=True).last_hidden_state
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand(hidden.shape[0], -1)
        last = positions.masked_fill(encoded["attention_mask"] == 0, -1).max(dim=1).values
        require(bool(torch.all(last >= 0).item()), "empty semantic input")
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_index, last, :].float()
        norms = torch.linalg.vector_norm(pooled, ord=2, dim=1, keepdim=True)
        require(bool(torch.all(torch.isfinite(pooled)).item()) and bool(torch.all(norms > 0).item()), "invalid semantic hidden state")
        pooled = pooled / norms
        outputs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    array = np.concatenate(outputs, axis=0)
    require(array.shape[0] == len(texts) and np.isfinite(array).all(), "semantic representation output")
    return array, {"minimum_tokens": int(minimum_tokens or 0), "maximum_tokens": maximum_tokens}

"""Minimal historical t2t_full and receiver-side C2C fuser runtime copy."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable
from protocol_min import format_openbook, receiver_prompt_tensors, decode_generated, extract_answer_from_content
# Root of the official C2C repository (https://github.com/thu-nics/C2C, commit 113c3a9).
RUNTIME_ROOT=Path(os.environ['C2C_ROOT'])
C2C_FUSERS=None
HELPER_CONFIG=None
RECEIVER_CONFIG=None
FUSER_LOAD_AUDIT=[]

def head_dim(config):
    return getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

def receiver_kv_shape(length):
    return (RECEIVER_CONFIG.num_hidden_layers, 2, RECEIVER_CONFIG.num_key_value_heads, length, head_dim(RECEIVER_CONFIG))
BACKGROUND_PROMPT=("In one clear sentence, describe the most essential background knowledge "
"needed to answer the question:\n\n{question}\n\n"
"Do NOT directly solve or give answer to the question.")
def install_source_paths():
    if str(RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0,str(RUNTIME_ROOT))
def extract_prediction(text):
    return extract_answer_from_content(text)
class ReceiverBundle:
    def __init__(self, model, tokenizer):
        self.model=model
        self.tokenizer=tokenizer
        self.device=model.device


def cuda_wall_stage(
    fn: Callable[[], Any],
    device,
) -> tuple[Any, float, float]:
    import torch

    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_ns = time.perf_counter_ns()
    start_event.record()
    value = fn()
    end_event.record()
    torch.cuda.synchronize(device)
    end_ns = time.perf_counter_ns()
    return value, (end_ns - start_ns) / 1e6, float(start_event.elapsed_time(end_event))

def wall_stage(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    value = fn()
    end = time.perf_counter_ns()
    return value, (end - start) / 1e6

class T2THelperBundle:
    def __init__(self, model, tokenizer):
        self.device=model.device
        self.stem_only=False
        self.tokenizer=tokenizer
        self.model=model

    def run(self, example: dict[str, Any]) -> dict[str, Any]:
        stages: dict[str, float] = {}
        cuda: dict[str, float] = {}

        def prepare():
            body = (
                str(example.get("question_stem", ""))
                if self.stem_only
                else format_openbook(example, use_template=False)
            )
            prompt = BACKGROUND_PROMPT.format(question=body)
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.device)
            return body, inputs

        (body, inputs), stages["prompt_tokenization"] = wall_stage(prepare)

        def generate():
            import torch

            with torch.inference_mode():
                return self.model.generate(
                    inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

        outputs, stages["helper_compute"], cuda["helper_compute"] = cuda_wall_stage(
            generate, self.device
        )

        def decode():
            generated = outputs[:, inputs.shape[-1] :]
            text = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return text, generated[0].detach().cpu().tolist()

        (text, ids), stages["helper_decode"] = wall_stage(decode)
        return {
            "helper_body": body,
            "helper_message": text,
            "helper_generated_token_ids": ids,
            "helper_input_token_count": int(inputs.shape[1]),
            "helper_output_token_count": len(ids),
            "stages": stages,
            "cuda_stages": cuda,
            "logical_application_bytes": len(text.encode("utf-8")),
            "unicode_character_count": len(text),
        }

class T2TReceiverBundle(ReceiverBundle):
    def consume(
        self,
        example: dict[str, Any],
        helper_body: str,
        helper_message: str,
        constraint=None,
    ) -> dict[str, Any]:
        stages: dict[str, float] = {}
        cuda: dict[str, float] = {}

        def prepare():
            question = format_openbook(example, use_template=True)
            messages = [
                {
                    "role": "user",
                    "content": BACKGROUND_PROMPT.format(question=helper_body),
                },
                {"role": "assistant", "content": helper_message},
                {"role": "user", "content": question},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.device)
            return inputs

        inputs, stages["prompt_tokenization"] = wall_stage(prepare)

        def generate():
            import torch

            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    **({'prefix_allowed_tokens_fn': constraint.callback(inputs.shape[1])} if constraint is not None else {}),
                )
            generated = outputs[:, inputs.shape[-1] :]
            return generated

        generated, stages["receiver_prepare_generate"], cuda[
            "receiver_prepare_generate"
        ] = cuda_wall_stage(generate, self.device)

        def decode_parse():
            text = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return text, extract_prediction(text)

        (text, prediction), stages["receiver_decode_parser"] = wall_stage(decode_parse)
        return {
            "generated_text": text,
            "generated_token_ids": generated[0].detach().cpu().tolist(),
            "receiver_input_token_count": int(inputs.shape[1]),
            "receiver_prompt_token_ids": inputs[0].detach().cpu().tolist(),
            "receiver_output_token_count": int(generated.shape[1]),
            "prediction": prediction,
            "stages": stages,
            "cuda_stages": cuda,
        }

def cache_to_stacked(cache, length: int):
    import torch

    layers = []
    for layer in range(len(cache)):
        key, value = cache[layer]
        key = key[:, :, :length, :]
        value = value[:, :, :length, :]
        if key.shape[0] != 1 or value.shape != key.shape:
            raise ValueError("unexpected cache batch/shape")
        layers.append(torch.stack((key[0], value[0]), dim=0))
    return torch.stack(layers, dim=0).contiguous()

def stacked_to_cache(stacked, device):
    import torch
    from transformers.cache_utils import DynamicCache

    if stacked.ndim != 5:
        raise ValueError(f"stacked KV must be rank 5, got {stacked.shape}")
    cache = DynamicCache()
    for layer in range(stacked.shape[0]):
        key = stacked[layer, 0].unsqueeze(0).to(device)
        value = stacked[layer, 1].unsqueeze(0).to(device)
        cache.update(key, value, layer)
    return cache

def apply_projected_kv_to_receiver_base(base_stacked, projected):
    """Apply released C2C output with the historical receiver-cache semantics.

    The released evaluation fuser emits float32 projected KV, but the
    historical wrapper assigns each layer into the receiver's retained
    bfloat16 DynamicCache slices.  Tensor assignment performs the dtype
    conversion at receiver-cache insertion; the communicated ``projected``
    tensor itself is not cast or mutated.
    """

    import torch

    if base_stacked.ndim != 5 or projected.ndim != 5:
        raise ValueError(
            "C2C base and projected KV must be rank 5; "
            f"got base={base_stacked.shape} projected={projected.shape}"
        )
    if tuple(base_stacked.shape) != tuple(projected.shape):
        raise ValueError(
            "C2C base/projected KV shape mismatch; "
            f"base={tuple(base_stacked.shape)} "
            f"projected={tuple(projected.shape)}"
        )
    if base_stacked.dtype != torch.bfloat16:
        raise ValueError(
            f"C2C receiver base KV must be bfloat16, got {base_stacked.dtype}"
        )
    if projected.dtype != torch.float32:
        raise ValueError(
            f"C2C communicated projected KV must be float32, got {projected.dtype}"
        )
    if base_stacked.device != projected.device:
        raise ValueError(
            "C2C base/projected KV must be on the receiver device before "
            f"insertion; base={base_stacked.device} projected={projected.device}"
        )

    # Match wrapper.py's per-layer assignment into bfloat16 key/value cache
    # slices.  Do not replace this with a cast of the transmitted tensor:
    # float32 remains the audited communication representation.
    for layer in range(base_stacked.shape[0]):
        base_stacked[layer, 0] = projected[layer, 0]
        base_stacked[layer, 1] = projected[layer, 1]

    if base_stacked.dtype != torch.bfloat16:
        raise RuntimeError(
            f"C2C receiver-cache dtype changed unexpectedly: {base_stacked.dtype}"
        )
    return base_stacked

def historical_c2c_receiver_cache(base_stacked, projected, device):
    """Build the receiver DynamicCache after historical projected-KV insertion."""

    import torch

    applied = apply_projected_kv_to_receiver_base(base_stacked, projected)
    cache = stacked_to_cache(applied, device)
    if len(cache) != applied.shape[0]:
        raise RuntimeError(
            f"C2C receiver-cache layer count {len(cache)} != {applied.shape[0]}"
        )
    for layer in range(len(cache)):
        key, value = cache[layer]
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise RuntimeError(
                "C2C receiver DynamicCache dtype mismatch after insertion; "
                f"layer={layer} key={key.dtype} value={value.dtype}"
            )
    return cache

def load_c2c_fusers(device):
    import re
    import torch

    install_source_paths()
    from rosetta.model.projector import load_projector

    count = len(
        [name for name in os.listdir(C2C_FUSERS) if re.match(r"projector_\d+\.pt", name)]
    )
    if count != RECEIVER_CONFIG.num_hidden_layers:
        raise ValueError(f"expected one C2C fuser per receiver layer, found {count}")
    projectors = []
    for index in range(count):
        model = load_projector(str(C2C_FUSERS / f"projector_{index}.json"))
        model = model.to(device).eval()
        state = torch.load(
            C2C_FUSERS / f"projector_{index}.pt",
            map_location=device,
            weights_only=True,
        )
        status = model.load_state_dict(state, strict=True)
        assert not status.missing_keys and not status.unexpected_keys
        assert all(torch.isfinite(t).all().item() for t in state.values())
        assert model.source_dim == head_dim(HELPER_CONFIG)
        assert model.target_dim == head_dim(RECEIVER_CONFIG)
        assert model.source_num_heads == HELPER_CONFIG.num_key_value_heads
        assert model.target_num_heads == RECEIVER_CONFIG.num_key_value_heads
        FUSER_LOAD_AUDIT.append({'projector': index, 'strict': True, 'state_tensors': len(state),
                                 'parameters': sum(p.numel() for p in model.parameters()),
                                 'missing_keys': status.missing_keys, 'unexpected_keys': status.unexpected_keys})
        projectors.append(model)
    mapping = json.loads((C2C_FUSERS / "projector_config.json").read_text())
    layers = mapping["0"]["1"]
    assert all(len(pairs) == 1 for pairs in layers.values())
    mapping = {int(target): (int(pairs[0][0]), int(pairs[0][1])) for target, pairs in layers.items()}
    assert set(mapping) == set(range(RECEIVER_CONFIG.num_hidden_layers))
    assert all(0 <= src < HELPER_CONFIG.num_hidden_layers and 0 <= proj < count for src, proj in mapping.values())
    assert set(proj for src, proj in mapping.values()) == set(range(count))
    return projectors, mapping

class C2CReceiverBundle(ReceiverBundle):
    def prepare_base(self, example: dict[str, Any]) -> dict[str, Any]:
        import torch

        stages: dict[str, float] = {}
        cuda: dict[str, float] = {}
        tensors, stages["prompt_tokenization"] = wall_stage(
            lambda: receiver_prompt_tensors(self.tokenizer, example, self.device)[2]
        )
        length = tensors["input_ids"].shape[1]
        instruction_length = length - 1
        position_ids = tensors["attention_mask"].long().cumsum(-1) - 1

        def prefill():
            with torch.no_grad():
                return self.model(
                    input_ids=tensors["input_ids"][:, :instruction_length],
                    attention_mask=tensors["attention_mask"][:, :instruction_length],
                    position_ids=position_ids[:, :instruction_length],
                    use_cache=True,
                    return_dict=True,
                )

        output, stages["receiver_base_prefill"], cuda[
            "receiver_base_prefill"
        ] = cuda_wall_stage(prefill, self.device)
        stacked = cache_to_stacked(output.past_key_values, instruction_length)
        expected = receiver_kv_shape(instruction_length)
        if tuple(stacked.shape) != expected or stacked.dtype != torch.bfloat16:
            raise ValueError(
                f"base KV {tuple(stacked.shape)} {stacked.dtype}, "
                f"expected {expected} bfloat16"
            )
        return {
            "tensors": tensors,
            "position_ids": position_ids,
            "instruction_length": instruction_length,
            "base_tensor": stacked,
            "receiver_input_token_count": int(tensors["input_ids"].shape[1]),
            # ``cache_to_stacked`` above creates a distinct stacked tensor.
            # Retain the model's untouched cache solely so the amendment can
            # hash the historical base representation after the matched
            # interval without adding validation work to that interval.
            "base_cache_for_validation": output.past_key_values,
            "stages": stages,
            "cuda_stages": cuda,
        }

class C2CSharerHelperBundle:
    """Frozen helper prefill that exposes the actual fuser source cache."""

    def __init__(self, helper, receiver_tokenizer):
        self.device=helper.device
        self.helper=helper
        self.tokenizer=receiver_tokenizer

    def run(self, example: dict[str, Any]) -> dict[str, Any]:
        import torch

        stages: dict[str, float] = {}
        cuda: dict[str, float] = {}

        def prepare():
            _, _, tensors = receiver_prompt_tensors(
                self.tokenizer, example, self.device
            )
            return tensors, tensors["input_ids"].shape[1] - 1

        (tensors, instruction_length), stages[
            "helper_prompt_tokenization"
        ] = wall_stage(prepare)

        def helper_prefill():
            with torch.no_grad():
                return self.helper(
                    input_ids=tensors["input_ids"][:, :instruction_length],
                    attention_mask=tensors["attention_mask"][:, :instruction_length],
                    position_ids=(
                        tensors["attention_mask"].long().cumsum(-1) - 1
                    )[:, :instruction_length],
                    use_cache=True,
                    return_dict=True,
                ).past_key_values

        source_cache, stages["helper_compute_cache_extraction"], cuda[
            "helper_compute_cache_extraction"
        ] = cuda_wall_stage(helper_prefill, self.device)
        # These are runtime assertions, not an interchange conversion.
        if type(source_cache).__module__ != "transformers.cache_utils":
            raise ValueError(
                "unexpected Sharer cache module "
                f"{type(source_cache).__module__}"
            )
        if type(source_cache).__name__ != "DynamicCache":
            raise ValueError(
                f"unexpected Sharer cache type {type(source_cache).__name__}"
            )
        if len(source_cache) != HELPER_CONFIG.num_hidden_layers:
            raise ValueError(
                f"unexpected Sharer cache layer count {len(source_cache)}"
            )
        for layer in range(len(source_cache)):
            pair = source_cache[layer]
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    f"Sharer cache layer {layer} is not a key/value tuple"
                )
            key, value = pair
            expected = (1, HELPER_CONFIG.num_key_value_heads, instruction_length, head_dim(HELPER_CONFIG))
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(
                    f"Sharer layer {layer} shape key={tuple(key.shape)} "
                    f"value={tuple(value.shape)} expected={expected}"
                )
            if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
                raise ValueError(
                    f"Sharer layer {layer} dtype key={key.dtype} "
                    f"value={value.dtype}; expected runtime bfloat16"
                )
        return {
            "source_cache": source_cache,
            "instruction_length": instruction_length,
            "helper_input_token_count": instruction_length,
            "stages": stages,
            "cuda_stages": cuda,
        }

class C2CReceiverSideFuserBundle(C2CReceiverBundle):
    """Receiver model, unchanged fusers, local fusion, and frozen generation."""

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.projectors, self.mapping = load_c2c_fusers(self.device)

    def fuse_and_generate(
        self,
        prepared: dict[str, Any],
        source_cache,
        constraint=None,
    ) -> dict[str, Any]:
        import torch
        from rosetta.model.sampling import sample_token

        stages: dict[str, float] = {}
        cuda: dict[str, float] = {}
        tensors = prepared["tensors"]
        position_ids = prepared["position_ids"]
        instruction_length = prepared["instruction_length"]
        base = prepared["base_tensor"]
        if len(source_cache) != HELPER_CONFIG.num_hidden_layers:
            raise ValueError(
                f"reconstructed Sharer cache has {len(source_cache)} layers"
            )
        base_expected = receiver_kv_shape(instruction_length)
        if tuple(base.shape) != base_expected or base.dtype != torch.bfloat16:
            raise ValueError(
                f"receiver base {tuple(base.shape)} {base.dtype}; "
                f"expected {base_expected} bfloat16"
            )

        def fuse():
            outputs = []
            for target_layer in range(RECEIVER_CONFIG.num_hidden_layers):
                source_layer, projector_index = self.mapping[target_layer]
                source_key, source_value = source_cache[source_layer]
                source_pair = (
                    source_key[:, :, :instruction_length, :],
                    source_value[:, :, :instruction_length, :],
                )
                base_pair = (
                    base[target_layer, 0].unsqueeze(0),
                    base[target_layer, 1].unsqueeze(0),
                )
                projected_key, projected_value = self.projectors[
                    projector_index
                ](source_pair, base_pair)
                outputs.append(
                    torch.stack((projected_key[0], projected_value[0]), dim=0)
                )
            return torch.stack(outputs, dim=0).contiguous()

        projected, stages["receiver_side_fuser_compute"], cuda[
            "receiver_side_fuser_compute"
        ] = cuda_wall_stage(fuse, self.device)
        projected_expected = receiver_kv_shape(instruction_length)
        if (
            tuple(projected.shape) != projected_expected
            or projected.dtype != torch.float32
        ):
            raise ValueError(
                f"local projected KV {tuple(projected.shape)} "
                f"{projected.dtype}; expected {projected_expected} float32"
            )

        def apply_cache():
            return historical_c2c_receiver_cache(
                base, projected, self.device
            )

        cache, stages["receiver_cache_prepare_apply"], cuda[
            "receiver_cache_prepare_apply"
        ] = cuda_wall_stage(apply_cache, self.device)

        def generate():
            with torch.no_grad():
                prefill = self.model(
                    input_ids=tensors["input_ids"][:, instruction_length:],
                    attention_mask=tensors["attention_mask"],
                    position_ids=position_ids[:, instruction_length:],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                current_past = prefill.past_key_values
                all_ids = tensors["input_ids"]
                attention = tensors["attention_mask"]
                logits = prefill.logits[:, -1, :]
                generated: list[torch.Tensor] = []
                eos_value = self.model.generation_config.eos_token_id
                eos_set = (
                    set(eos_value if isinstance(eos_value, list) else [eos_value])
                    if eos_value is not None
                    else set()
                )
                pad_value = self.model.generation_config.pad_token_id
                if pad_value is None and eos_set:
                    pad_value = next(iter(eos_set))
                finished = torch.zeros(
                    all_ids.shape[0], dtype=torch.bool, device=self.device
                )
                for _ in range(64):
                    if constraint is not None:
                        from format_constraint import restrict_logits
                        allowed = constraint.callback(tensors['input_ids'].shape[1])(0, all_ids[0])
                        logits = restrict_logits(logits, allowed)
                    next_token = sample_token(
                        logits, temperature=0.0, top_p=1.0, top_k=-1
                    )
                    if not isinstance(next_token, torch.Tensor):
                        next_token = torch.tensor(
                            [next_token], device=self.device, dtype=torch.long
                        )
                    if eos_set:
                        just_finished = torch.zeros_like(finished)
                        for eos_id in eos_set:
                            just_finished |= next_token == eos_id
                        finished |= just_finished
                        if pad_value is not None:
                            next_token = torch.where(
                                finished,
                                torch.tensor(
                                    pad_value,
                                    device=self.device,
                                    dtype=next_token.dtype,
                                ),
                                next_token,
                            )
                    generated.append(next_token)
                    token_column = next_token.unsqueeze(1)
                    all_ids = torch.cat((all_ids, token_column), dim=1)
                    attention = torch.cat(
                        (
                            attention,
                            torch.ones(
                                (attention.shape[0], 1),
                                dtype=attention.dtype,
                                device=self.device,
                            ),
                        ),
                        dim=1,
                    )
                    if eos_set and torch.all(finished):
                        break
                    decoded = self.model(
                        input_ids=token_column,
                        attention_mask=attention,
                        past_key_values=current_past,
                        use_cache=True,
                        return_dict=True,
                    )
                    current_past = decoded.past_key_values
                    logits = decoded.logits[:, -1, :]
            return torch.stack(generated, dim=1)

        generated, stages["receiver_generation"], cuda[
            "receiver_generation"
        ] = cuda_wall_stage(generate, self.device)

        def decode_parse():
            text = decode_generated(self.tokenizer, generated[0])
            return text, extract_prediction(text)

        (text, prediction), stages["receiver_decode_parser"] = wall_stage(
            decode_parse
        )
        return {
            "generated_text": text,
            "generated_token_ids": generated[0].detach().cpu().tolist(),
            "receiver_input_token_count": int(tensors["input_ids"].shape[1]),
            "receiver_output_token_count": int(generated.shape[1]),
            "prediction": prediction,
            "stages": stages,
            "cuda_stages": cuda,
            "projected_tensor_for_validation": projected,
            "applied_cache_for_validation": cache,
            "applied_stacked_for_validation": base,
            "receiver_cache_shape": list(projected_expected),
            "receiver_cache_dtype": "bfloat16",
        }

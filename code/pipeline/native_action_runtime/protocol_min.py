"""Minimal snapshot of the prompt and answer-extraction protocol used by every run."""
from __future__ import annotations
import re
from typing import Any, Dict, Optional


def build_prompt(dataset: str, locale: str, question: str, choices: str, use_cot: bool, use_template: bool = True) -> str:
    """
    Build a localized prompt for a given dataset and locale.

    Currently supports:
    - dataset: "mmmlu"
      - locale: "SW_KE" (Swahili). Other locales fall back to English.

    Args:
        dataset: Dataset identifier (e.g., "mmmlu")
        locale: Locale/subject code (e.g., "SW_KE")
        question: Question text
        choices: Formatted choices string
        use_cot: Whether to include CoT instruction

    Returns:
        Localized prompt string
    """
    
        # Unified default English templates (shared by MMLU and MMMLU)
    if not use_cot:
        template = """Accurately answer the following question:

{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D".
- Do not include any explanations, additional text, or punctuation besides the answer.

The correct answer is"""

    else:
        template = """Accurately answer the following question:
                   
{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Let's think step by step and explain your reasoning briefly.
- Then give the final answer starting with The correct answer is"""

    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{choices}}", choices)

    if not use_template:
        prompt = question + "\n\nChoices:\n" + choices

    return prompt

def extract_answer_from_content(text: str) -> Optional[str]:
    """
    Extract answer from model output with robust multi-pattern matching.
    Supports multiple languages and response formats.
    
    Args:
        text: Model output text
        
    Returns:
        Extracted answer letter or None
    """
    text = text.strip()
    if not text:
        return None

    # Define multiple answer patterns for different languages and formats
    answer_patterns = [
        # English patterns
        r'Answer:\s*(.*)',
        r'answer:\s*(.*)',
        r'ANSWER:\s*(.*)',
        r'Your answer:\s*(.*)',
        r'your answer:\s*(.*)',
        r'YOUR ANSWER:\s*(.*)',
        r'The answer is\s*(.*)',
        r'the answer is\s*(.*)',
        r'THE ANSWER IS\s*(.*)',
        r'Correct answer is\s*(.*)',
        r'correct answer is\s*(.*)',
        r'Correct answer is:\s*(.*)',
        r'correct answer is:\s*(.*)',
        r'Correct answer:\s*(.*)',
        r'correct answer:\s*(.*)',
        r'CORRECT ANSWER:\s*(.*)',
        
        # Swahili patterns
        r'Jibu lako:\s*(.*)',
        r'jibu lako:\s*(.*)',
        r'JIBU LAKO:\s*(.*)',
        r'Jibu:\s*(.*)',
        r'jibu:\s*(.*)',
        r'JIBU:\s*(.*)',
        r'Jibu sahihi:\s*(.*)',
        r'jibu sahihi:\s*(.*)',
        r'JIBU SAHIHI:\s*(.*)',
        
        # Other common patterns
        r'Response:\s*(.*)',
        r'response:\s*(.*)',
        r'RESPONSE:\s*(.*)',
        r'Choice:\s*(.*)',
        r'choice:\s*(.*)',
        r'CHOICE:\s*(.*)',
        r'Option:\s*(.*)',
        r'option:\s*(.*)',
        r'OPTION:\s*(.*)',
    ]
    
    # 1. Try to match any of the answer patterns
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            # Search for first A-D letter in the matched part
            for char in answer_part:
                if char in {'A', 'B', 'C', 'D'}:
                    return char
    
    # 2. Look for standalone A-D letters that are likely answers
    # Prioritize letters at the end of text or with clear answer-like context
    standalone_patterns = [
        r'\b([A-D])(?:\s*[.,!?:)]?\s*$)',  # A-D at end of text with optional punctuation
        r'\b([A-D])(?:\s*[.,!?:)]\s)',     # A-D followed by punctuation and space
        r'(?:^|\s)([A-D])(?:\s*$)',        # A-D at start or with word boundary at end
    ]
    
    for pattern in standalone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Check if this looks like mathematical expressions rather than answers
            math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
            has_math = any(indicator in text for indicator in math_indicators)
            has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
            
            # If it has math indicators but no answer indicators, it's likely mathematical notation
            if has_math and not has_answer_indicators:
                continue  # Skip this match, try next pattern
            
            return matches[-1].upper()
    
    # 3. Fallback: find all A-D letters but be more selective
    all_letters = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if all_letters:
        # Check if this looks like mathematical expressions rather than answers
        math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
        has_math = any(indicator in text for indicator in math_indicators)
        has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
        
        # If it has math indicators but no answer indicators, it's likely mathematical notation
        if has_math and not has_answer_indicators:
            return None
        
        # Otherwise, return the last letter found
        return all_letters[-1].upper()
    
    # 3. Search backwards for any A-D letter as fallback
    for char in reversed(text):
        if char in {'A', 'B', 'C', 'D'}:
            return char

    return None

def apply_generation_config(model: Any, generation_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Apply generation configuration to a model and handle sampling parameters.
    
    This function applies the provided generation config to the model and removes
    sampling parameters (temperature, top_p, top_k, min_p) when do_sample=False
    to avoid warnings from the transformers library. If no config is provided,
    it defaults to greedy decoding with cleaned sampling parameters.
    
    Args:
        model: Model object with generation_config attribute
        generation_config: Optional generation configuration dictionary.
                          If None, defaults to greedy decoding (do_sample=False).
    """
    if not hasattr(model, 'generation_config'):
        return
    
    # If no config provided, default to greedy decoding
    if not generation_config:
        generation_config = {'do_sample': False}
    
    # Apply all configuration parameters
    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)
    
    # Disable sampling parameters if do_sample=False to avoid warnings
    # We set them to None instead of deleting, since some model code may
    # access these attributes unconditionally.
    if not generation_config.get('do_sample', True):
        sampling_params = ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty']
        for param in sampling_params:
            try:
                setattr(model.generation_config, param, None)
            except Exception:
                # If the backend does not allow setting, ignore silently
                pass

def choice_texts(example: dict[str, Any]) -> list[str]:
    raw = example.get("choices")
    if isinstance(raw, dict):
        values = list(raw.get("text", []))
    elif isinstance(raw, list):
        values = [
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in raw
        ]
    else:
        values = []
    if len(values) != 4:
        raise ValueError(f"expected four choices, got {len(values)}")
    return [str(value) for value in values]

def format_openbook(example: dict[str, Any], use_template: bool = True) -> str:

    choices = "".join(
        f"{chr(65 + index)}. {text}\n"
        for index, text in enumerate(choice_texts(example))
    )
    return build_prompt(
        dataset="mmlu-redux",
        locale="",
        question=str(example.get("question_stem", "")),
        choices=choices,
        use_cot=False,
        use_template=use_template,
    )

def receiver_prompt_tensors(tokenizer, example: dict[str, Any], device):
    prompt = format_openbook(example, use_template=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokenized = tokenizer(rendered, return_tensors="pt")
    return prompt, rendered, {
        key: value.to(device) for key, value in tokenized.items()
    }

def decode_generated(tokenizer, ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip("\n")

def generate_receiver(receiver, tokenizer, tensors: dict[str, Any], prefix_allowed_tokens_fn=None) -> tuple[str, Any]:
    import torch

    with torch.inference_mode():
        output = receiver.generate(
            **({'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn} if prefix_allowed_tokens_fn is not None else {}),
            input_ids=tensors["input_ids"],
            attention_mask=tensors.get("attention_mask"),
            do_sample=False,
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=(
                receiver.generation_config.pad_token_id
                if receiver.generation_config.pad_token_id is not None
                else tokenizer.pad_token_id
            ),
            eos_token_id=(
                receiver.generation_config.eos_token_id
                if receiver.generation_config.eos_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    generated = output[0, tensors["input_ids"].shape[1] :]
    return decode_generated(tokenizer, generated), generated

# Exact source-function extraction; no labels or scoring.
_INSTRUCTION_ABCD = '"The correct answer is A/B/C/D"'

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

def display_labels(n):
    assert 1 <= n <= 26
    return [chr(65 + i) for i in range(n)]

def choices_block(texts):
    return ''.join(f'{chr(65 + i)}. {t}\n' for i, t in enumerate(texts))

def receiver_prompt(rec):
    """User message for the receiver in Receiver-only and C2C; also the last user turn in Text.

    The instruction lists the question's own display labels (A/B/C for three options, A/B/C/D for
    four, A/B/C/D/E for five); for four-option questions it is byte-identical to the OpenBookQA template.
    """
    prompt = build_prompt(dataset='mmlu-redux', locale='', question=str(rec['question_stem']),
                          choices=choices_block(rec['choice_text']), use_cot=False, use_template=True)
    assert prompt.count(_INSTRUCTION_ABCD) == 1
    labels = '/'.join(display_labels(len(rec['choice_text'])))
    return prompt.replace(_INSTRUCTION_ABCD, f'"The correct answer is {labels}"')

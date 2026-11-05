"""ARC-Challenge adaptation of the OpenBookQA protocol (CPU only, no model loading).

- The original option order is kept; display labels are A, B, C, ... in that order, and gold is
  mapped from the original label to the display label through a reversible mapping.
- The prompt, the Text helper's question body and the routing query reuse build_prompt /
  serialize_question unchanged, without format_openbook's "exactly four options" restriction; the
  receiver instruction lists each question's own legal labels (v2, 2026-09-11). For four-option
  questions the output is byte-identical to the OpenBookQA protocol.
- The parser is the corrected parser parameterised by the question's legal display labels, with one
  added rule: an illegal single-letter label right after an explicit answer marker (for example D on a
  three-option question) is invalid.
"""
import re
import sys
from pathlib import Path

from protocol_min import build_prompt  # noqa: E402
from query_features import serialize_question  # noqa: E402

# Copied verbatim from legacy_methods.py (that module imports torch, so it is not imported here).
BACKGROUND_PROMPT = ("In one clear sentence, describe the most essential background knowledge "
                     "needed to answer the question:\n\n{question}\n\n"
                     "Do NOT directly solve or give answer to the question.")


def display_labels(n):
    assert 1 <= n <= 26
    return [chr(65 + i) for i in range(n)]


def choices_block(texts):
    return ''.join(f'{chr(65 + i)}. {t}\n' for i, t in enumerate(texts))


_INSTRUCTION_ABCD = '"The correct answer is A/B/C/D"'


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


def helper_body(rec):
    """Question body the Text (t2t_full) helper sees (format_openbook(use_template=False))."""
    return build_prompt(dataset='mmlu-redux', locale='', question=str(rec['question_stem']),
                        choices=choices_block(rec['choice_text']), use_cot=False, use_template=False)


def helper_prompt(rec):
    return BACKGROUND_PROMPT.format(question=helper_body(rec))


def router_query(rec):
    return serialize_question(rec['question_stem'], tuple(rec['choice_labels']), tuple(rec['choice_text']))


# ---------------- parser ----------------
_ANSWER_PATTERNS = [
    r'Answer:\s*(.*)', r'answer:\s*(.*)', r'ANSWER:\s*(.*)',
    r'Your answer:\s*(.*)', r'your answer:\s*(.*)', r'YOUR ANSWER:\s*(.*)',
    r'The answer is\s*(.*)', r'the answer is\s*(.*)', r'THE ANSWER IS\s*(.*)',
    r'Correct answer is\s*(.*)', r'correct answer is\s*(.*)',
    r'Correct answer is:\s*(.*)', r'correct answer is:\s*(.*)',
    r'Correct answer:\s*(.*)', r'correct answer:\s*(.*)', r'CORRECT ANSWER:\s*(.*)',
    r'Jibu lako:\s*(.*)', r'jibu lako:\s*(.*)', r'JIBU LAKO:\s*(.*)',
    r'Jibu:\s*(.*)', r'jibu:\s*(.*)', r'JIBU:\s*(.*)',
    r'Jibu sahihi:\s*(.*)', r'jibu sahihi:\s*(.*)', r'JIBU SAHIHI:\s*(.*)',
    r'Response:\s*(.*)', r'response:\s*(.*)', r'RESPONSE:\s*(.*)',
    r'Choice:\s*(.*)', r'choice:\s*(.*)', r'CHOICE:\s*(.*)',
    r'Option:\s*(.*)', r'option:\s*(.*)', r'OPTION:\s*(.*)',
]
_MATH = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
_ANS_WORDS = ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi']
MARKER = re.compile(
    r'\b(?:(?:the|my|your)\s+)?(?:(?:correct|final)\s+)?'
    r'(?:answer|choice|option|response)\s*(?:is\s*:?[ \t]*|:[ \t]*)', re.I)
_DECLARED_LETTER = re.compile(r'^\s*[\"\x27(\[]?([A-Z])(?=\s*(?:[.,;:)\]\"\x27]|$))')


def legacy_extract(text, legal):
    """protocol_min.extract_answer_from_content with A-D replaced by the question's legal labels."""
    L = set(legal)
    cls = '[' + ''.join(legal) + ']'
    text = text.strip()
    if not text:
        return None
    for pattern in _ANSWER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            for char in match.group(1).strip():
                if char in L:
                    return char
    for pattern in [rf'\b({cls})(?:\s*[.,!?:)]?\s*$)', rf'\b({cls})(?:\s*[.,!?:)]\s)', rf'(?:^|\s)({cls})(?:\s*$)']:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            has_math = any(i in text for i in _MATH)
            has_ans = any(p in text.lower() for p in _ANS_WORDS)
            if has_math and not has_ans:
                continue
            return matches[-1].upper()
    letters = re.findall(rf'\b({cls})\b', text, re.IGNORECASE)
    if letters:
        has_math = any(i in text for i in _MATH)
        has_ans = any(p in text.lower() for p in _ANS_WORDS)
        if has_math and not has_ans:
            return None
        return letters[-1].upper()
    for char in reversed(text):
        if char in L:
            return char
    return None


def parse_answer(text, legal):
    """scoring.parse_answer parameterised by the legal labels; returns the one explicit legal label or INVALID."""
    legal = list(legal)
    cls = '[' + ''.join(legal) + ']'
    option_list = re.compile(rf'\b{cls}\b(?:\s*(?:/|\||,\s*(?:(?:or|and)\s+)?|\bor\b|\band\b)\s*{cls}\b)+', re.I)
    single = re.compile(rf'^\s*[\"\x27(\[]?({cls})\b', re.I)
    # Added rule: an illegal single letter right after the last explicit answer marker is INVALID (no fallback to other letters).
    declared = [m for m in (_DECLARED_LETTER.match(text[mk.end():]) for mk in MARKER.finditer(text)) if m]
    if declared and declared[-1].group(1) not in legal:
        return {'answer': None, 'valid': False, 'reason': 'illegal_label', 'historical_answer': legacy_extract(text, legal)}
    legacy = legacy_extract(text, legal)
    lists = [m for m in option_list.finditer(text)
             if len(set(re.findall(rf'\b{cls}\b', m.group().upper()))) > 1]
    if not lists:
        return {'answer': legacy, 'valid': legacy is not None,
                'reason': None if legacy is not None else 'legacy_parse_failure', 'historical_answer': legacy}
    declarations = []
    for marker in MARKER.finditer(text):
        s = single.match(text[marker.end():])
        if s:
            start = marker.end() + s.start(1)
            ambiguous = any(m.start() == start for m in lists)
            declarations.append((marker.start(), None if ambiguous else s.group(1).upper()))
    leading = single.match(text)
    if leading and not any(m.start() == leading.start(1) for m in lists):
        declarations.insert(0, (-1, leading.group(1).upper()))
    if declarations and declarations[-1][1] is not None:
        return {'answer': declarations[-1][1], 'valid': True, 'reason': None, 'historical_answer': legacy}
    return {'answer': None, 'valid': False, 'reason': 'unresolved_multiple_options', 'historical_answer': legacy}


def score(text, gold, legal, runtime_error=None):
    parsed = parse_answer(text or '', legal)
    return {**parsed, 'correct': runtime_error is None and parsed['valid'] and parsed['answer'] == gold}

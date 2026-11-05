"""Minimal gold-independent guard for unresolved lists of answer options.

All non-list cases retain the stage parser. An explicit selection takes
precedence over option lists in explanations. If an earlier list is resolved
by a later explicit answer, use that final answer.
"""
import re
from protocol_min import extract_answer_from_content as historical_parser

MARKER = re.compile(
    r'\b(?:(?:the|my|your)\s+)?(?:(?:correct|final)\s+)?'
    r'(?:answer|choice|option|response)\s*(?:is\s*:?[ \t]*|:[ \t]*)', re.I)
OPTION_LIST = re.compile(
    r'\b[A-D]\b(?:\s*(?:/|\||,\s*(?:(?:or|and)\s+)?|\bor\b|\band\b)\s*[A-D]\b)+', re.I)
SINGLE = re.compile(r'^\s*[\"\x27(\[]?([A-D])\b', re.I)


def parse_answer(text):
    legacy = historical_parser(text)
    lists = [m for m in OPTION_LIST.finditer(text)
             if len(set(re.findall(r'\b[A-D]\b', m.group().upper()))) > 1]
    if not lists:
        return {'answer': legacy, 'valid': legacy is not None,
                'reason': None if legacy is not None else 'legacy_parse_failure',
                'historical_answer': legacy}
    declarations = []
    for marker in MARKER.finditer(text):
        tail = text[marker.end():]
        single = SINGLE.match(tail)
        if single:
            answer_start = marker.end() + single.start(1)
            ambiguous = any(m.start() == answer_start for m in lists)
            declarations.append((marker.start(), None if ambiguous else single.group(1).upper()))
    # Also recognize a clear leading "C. ..." without an answer marker.
    leading = SINGLE.match(text)
    if leading and not any(m.start() == leading.start(1) for m in lists):
        declarations.insert(0, (-1, leading.group(1).upper()))
    if declarations and declarations[-1][1] is not None:
        return {'answer': declarations[-1][1], 'valid': True,
                'reason': None, 'historical_answer': legacy}
    return {'answer': None, 'valid': False, 'reason': 'unresolved_multiple_options',
            'historical_answer': legacy}


def score(text, gold, runtime_error=None):
    parsed = parse_answer(text or '')
    return {**parsed, 'correct': runtime_error is None and parsed['valid'] and parsed['answer'] == gold,
            'historical_correct': runtime_error is None and parsed['historical_answer'] == gold}

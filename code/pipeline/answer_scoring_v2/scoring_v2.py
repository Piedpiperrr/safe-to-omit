"""P2_SCORING_V2: syntax only; no gold, item identity, action, or option text."""
import re
import unicodedata

VERSION = 'P2_SCORING_V2'
IMPLEMENTATION = '2.0.0'
MARKER = re.compile(r'\b(?:(?:the|my|your)\s+)?(?:(?:final|correct)\s+)?(?:answer|choice|option|response)\s*(?:is\s*:?\s*|:\s*)', re.I)
# Generic predicative selection: no task/topic nouns in this grammar.
EMBEDDED = re.compile(r'\b(?:the|my)\s+(?:correct|right|best|final)\s+[^.!?;:\n]{1,180}?\s+is\s*(?:(?:option|choice)\s+)?', re.I)
CHOOSE = re.compile(r'\bI\s+(?:(?:finally|therefore)\s+)?(?:choose|select|pick)\s+(?:(?:option|choice)\s+)?', re.I)
REVERSE = re.compile(r'\b(?:option|choice)\s+([A-Z])\s+is\s+(?:also\s+)?(?:correct|right|the\s+(?:correct\s+)?answer)\b', re.I)
LABEL = re.compile(r'^[\s"\x27]*(?:\(([A-Za-z])\)|\[([A-Za-z])\]|([A-Za-z])\b)')
LEADING = re.compile(r'^(?:\(([A-Z])\)|\[([A-Z])\]|([A-Z])(?=\s*(?:[.):,;!?]|$)))')
JOIN = re.compile(r'^\s*(?:/|\||,\s*(?:(?:or|and)\s+)?|\b(?:or|and|versus)\b)\s*([A-Z])\b')
REVISION = re.compile(r'\b(?:actually|correction|on second thought|I revise|I change|instead|sorry)\b', re.I)
REFERENCE = re.compile(r'\b(?:if|whether|unless|assuming|suppose|supposing|hypothetically|example|quoted?|says?|said|claims?|claimed|incorrect|not)\b', re.I)
QUOTE = re.compile(r'"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')
REFUSAL = re.compile(r'\b(?:I\s+(?:cannot|can\x27t|am unable to|won\x27t)\s+(?:provide|answer|determine|select)|(?:please\s+)?(?:rephrase|clarify)\s+the\s+question)\b', re.I)


def parse_answer(raw_answer, legal_labels):
    legal = tuple(legal_labels)
    if not legal or len(set(legal)) != len(legal) or any(not re.fullmatch('[A-Z]', x) for x in legal):
        raise ValueError('legal_labels must contain distinct displayed uppercase letters')
    if raw_answer is not None and not isinstance(raw_answer, str):
        raise TypeError('raw_answer must be str or None')
    t = unicodedata.normalize('NFKC', raw_answer or '').strip()
    t = re.sub(r'^```[^\n]*\n|\n```\s*$', '', t)
    t = t.replace('**', '').replace('__', '').replace('`', '').strip()
    t = t.translate(str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"}))
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ['"', "'"] and t.count(t[0]) == 2:
        t = t[1:-1].strip()
    events, ignored = [], []

    def result(answer, reason):
        return dict(answer=answer, valid=answer is not None, reason=reason, version=VERSION,
                    implementation=IMPLEMENTATION, events=events, ignored=ignored)

    if not t:
        return result(None, 'empty')
    quotes = [(m.start(), m.end()) for m in QUOTE.finditer(t)]

    def quoted(pos):
        return t[t.rfind('\n', 0, pos)+1:pos].lstrip().startswith('>') or any(a <= pos < b for a, b in quotes)

    def prefix(pos):
        a = max(t.rfind(x, 0, pos) for x in '.!?\n') + 1
        return t[a:pos]

    def add(start, end, raw, kind, marker=''):
        label = raw.upper()
        tail = t[end:]
        issue = None
        labels = [label]
        at = 0
        while (m := JOIN.match(tail[at:])):
            labels.append(m.group(1)); at += m.end()
        if len(labels) > 1:
            issue = 'multiple_options'
        elif re.match(r'\s*[=+*/^°]', tail) or (raw.islower() and re.match(r'\s+[A-Za-z]', tail)):
            ignored.append(dict(start=start, reason='formula_unit_article_or_variable'))
            return
        elif re.match(r'\s+(?:if|unless|would|might|may|could)\b', tail, re.I):
            ignored.append(dict(start=start, reason='conditional_selection'))
            return
        events.append(dict(start=start, end=end+at, label=label, labels=labels, kind=kind, issue=issue,
                           final=bool(re.search(r'\bfinal\b', marker, re.I)), revision=bool(REVISION.search(prefix(start)))))

    first = LEADING.match(t)
    if first and not quoted(0):
        add(0, first.end(), next(x for x in first.groups() if x), 'leading')
    elif (m := re.match(r'^([A-Z])\b', t)) and JOIN.match(t[m.end():]):
        add(0, m.end(), m.group(1), 'leading_list')
    spans = []
    for pattern, kind in [(MARKER, 'declaration'), (EMBEDDED, 'embedded_selection'), (CHOOSE, 'choice_verb')]:
        for marker in pattern.finditer(t):
            start, end = marker.span()
            if any(a <= start < b or start <= a < end for a, b in spans):
                continue
            if quoted(start) or REFERENCE.search(prefix(start)) or REFERENCE.search(marker.group()):
                ignored.append(dict(start=start, reason='quoted_hypothetical_or_negated')); continue
            m = LABEL.match(t[end:])
            if m:
                add(start, end+m.end(), next(x for x in m.groups() if x), kind, marker.group())
                spans.append((start, end+m.end()))
            elif kind == 'declaration':
                # Explanation continuations after an answer are not fresh choices.
                tail = t[end:]
                if re.match(r'(?:because|since|due\s+to|that\b|based\s+on)\b', tail, re.I):
                    ignored.append(dict(start=start, reason='explanation_continuation')); continue
                if re.match(r'(?:not|neither)\b', tail, re.I):
                    issue = 'negated_declaration'
                else:
                    issue = 'no_explicit_option'
                events.append(dict(start=start, end=end, label=None, labels=[], kind=kind, issue=issue,
                                   final=bool(re.search(r'\bfinal\b', marker.group(), re.I)), revision=bool(REVISION.search(prefix(start)))))
                spans.append((start, end))
    for m in REVERSE.finditer(t):
        if not quoted(m.start()) and not REFERENCE.search(prefix(m.start())):
            add(m.start(), m.end(), m.group(1), 'reverse_selection', m.group())
    events.sort(key=lambda e: e['start'])
    if not events:
        return result(None, 'refusal' if REFUSAL.search(t) else 'no_explicit_option')
    last = events[-1]
    # An explicit final/correction starts a new resolution scope; later claims still apply.
    cuts = [i for i, e in enumerate(events) if e['kind'] != 'leading' and (e['final'] or e['revision']) and not e['issue']]
    active = events[cuts[-1]:] if cuts else list(events)
    resolved = bool(cuts and cuts[-1] > 0)
    singles = [e for e in active if not e['issue'] and e['label'] in legal]
    if singles and len({e['label'] for e in singles}) == 1:
        first_single = singles[0]['start']
        active = [e for e in active if not (e['issue'] == 'multiple_options' and
                  len(e['labels']) == len(legal) and set(e['labels']) == set(legal) and e['start'] < first_single)]
    if any(any(x not in legal for x in e['labels']) for e in active):
        return result(None, 'illegal_label')
    issues = [e['issue'] for e in active if e['issue']]
    if issues:
        return result(None, issues[-1])
    labels = {e['label'] for e in active}
    if len(labels) != 1:
        return result(None, 'unresolved_conflict')
    answer = next(iter(labels))
    headings = [m.start() for m in re.finditer(r'(?m)^\s*(?:\([A-Z]\)|\[[A-Z]\]|[A-Z][.)])(?=\s|$)', t) if not quoted(m.start())]
    if len(headings) > 1 and not (last['kind'] != 'leading' and last['start'] > headings[-1]):
        return result(None, 'multiple_options')
    neg = re.compile(r'\b(?:(?:option|choice)\s+)?'+re.escape(answer)+r'\s+(?:is|would be)\s+(?:incorrect|wrong|not\s+(?:correct|the\s+answer))\b', re.I)
    for m in neg.finditer(t):
        if m.start() > last['start'] and not quoted(m.start()) and not REFERENCE.search(prefix(m.start())):
            return result(None, 'unresolved_conflict')
    if REFUSAL.search(t) and not resolved:
        return result(None, 'refusal')
    return result(answer, 'explicit_revision' if resolved else 'explicit_single')

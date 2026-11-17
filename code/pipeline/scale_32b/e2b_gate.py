"""Gate formal E2b stages; the authors authorized a separate pre-freeze technical smoke."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_path():return HERE/'FREEZE_DEBUG.txt'


def verify_freeze():
    p = freeze_path()
    if not p.is_file():
        raise SystemExit('E2B_NOT_FROZEN: formal stages require the reviewed protocol and commit freeze')
    receipt = json.loads(p.read_text())
    assert receipt['status'] == 'FROZEN_OPERATIONAL_REVISION' and receipt['approval_note']
    assert sha(HERE/'FREEZE.txt')==receipt['original_freeze_sha256']
    for name, expected in receipt['files'].items():
        assert sha(name) == expected, f'Frozen file changed: {name}'
    return receipt


def require_stage(stage):
    # Authors' authorization: 16 fit-only items, no gold, resource measurements only.
    # Smoke is never a statistical stage release and never supplies pilot records.
    if stage == 'smoke':
        return None
    raise SystemExit('E2B_FORMAL_ENTRY_MOVED: use run_e2b_formal.py / submit_e2b_formal.py; legacy formal entry disabled')


def stage_rows(config, stage):
    split = 'fit' if stage in ['smoke', 'pilot', 'fit_rest'] else 'dev' if stage == 'replay' else stage
    info = config['inputs'][split]
    assert sha(info['path']) == info['sha256']
    values = [json.loads(s) for s in Path(info['path']).read_text().splitlines()]
    assert all(set(r) == {'id','question_stem','choice_labels','choice_text'} for r in values)
    if stage == 'smoke': return values[:16]
    if stage == 'pilot': return values[:300]
    if stage == 'fit_rest': return values[300:]
    if stage == 'replay':
        by_id = {r['id']: r for r in values}
        return [by_id[i] for i in json.loads(Path(config['panel_ids']).read_text())]
    return values

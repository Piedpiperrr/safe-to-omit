"""Shared helpers for P2_R1_EXP drivers. Writes only inside this experiment folder."""
import sys
sys.dont_write_bytecode = True
import os, json, hashlib, time, datetime, pathlib, importlib.util, socket

NEW = pathlib.Path(__file__).resolve().parents[1]
ROOT = NEW.parent
DATA_ROOT = ROOT.parent.parent
P210 = ROOT / 'P2_10_20260911T122423Z'
MED = ROOT / 'P2_MEDIUM_PAIR_BOUNDARY_EXTENSION_STAGE1_20260915T144407Z'
MEDX = MED / 'execution_retry1_20260915T164957Z'
MMLU = ROOT / 'P2_MMLU_PRO_BREADTH_STAGE1_20260916T002337Z'
PARSER = ROOT / 'P2_SCORING_V2_20260912T191445Z/scoring_v2.py'
PARSER_SHA = 'd05978f400cf69aa6bddc56168a260ac3ea9158abaa640e9331602dc9bfcc2f9'
POP = NEW / 'records/populations'
INV = 'INVALID'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def save(p, v):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)


def append(p, v):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a') as f:
        f.write(json.dumps(v, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parser():
    assert sha(PARSER) == PARSER_SHA
    return load_module(PARSER, 'r1_scoring_v2').parse_answer


def deadline_ok(margin=0):
    return time.time() + margin < float(os.environ.get('R1_DEADLINE_EPOCH', 'inf'))


def env_record():
    import torch, transformers
    return {'utc': utc(), 'host': socket.gethostname(), 'job_id': os.environ.get('PBS_JOBID'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'CUDA_DEVICE_ORDER': os.environ.get('CUDA_DEVICE_ORDER'),
            'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version,
            'devices': [str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]}

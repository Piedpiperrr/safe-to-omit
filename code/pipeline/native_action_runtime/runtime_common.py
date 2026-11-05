import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
ACTIONS=['receiver_only','text','c2c','acw']
LAMBDAS=[0,.01,.03,.1,.3,1,3]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readrows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def freeze(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def append(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def query_only(row):return {k:row[k] for k in ['question_stem','choice_labels','choice_text','choices'] if k in row}

"""Check the downloaded dataset against its manifest and link records/ into place.

    python3 scripts/check_data.py [data dir]      (default: data)
"""
import hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


manifest = json.load(open(os.path.join(DATA, "MANIFEST.json")))
bad = [rel for rel, v in manifest["files"].items()
       if not os.path.exists(os.path.join(DATA, rel)) or sha(os.path.join(DATA, rel)) != v["sha256"]]
if bad:
    sys.exit("{} file(s) missing or changed, e.g. {}".format(len(bad), bad[0]))
link = os.path.join(ROOT, "records")
if os.path.islink(link):
    os.unlink(link)
if os.path.exists(link):
    sys.exit("records/ exists and is not a link; remove it first")
os.symlink(os.path.relpath(os.path.join(DATA, "records"), ROOT), link)
print("{} files match the manifest ({} v{}); records/ -> {}".format(
    len(manifest["files"]), manifest["dataset"], manifest["version"], os.path.relpath(DATA, ROOT) + "/records"))

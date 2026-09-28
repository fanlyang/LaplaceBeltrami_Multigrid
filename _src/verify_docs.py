"""Compare every doc now in the Overleaf project against the local deck files."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "report_slides"))

raw = open(os.path.join(HERE, "dump.json"), encoding="utf-8").read()
# mongosh echoes its interactive prompt onto the same line as our marker, so
# find the marker anywhere and take the rest of that line.
MARK = "@@JSON@@"
if raw.count(MARK) != 1:
    sys.exit("expected exactly one %s, got %d" % (MARK, raw.count(MARK)))
tail = raw[raw.index(MARK) + len(MARK):]
remote = json.loads(tail[:tail.index("\n")] if "\n" in tail else tail)

local = {}
for dirpath, _, files in os.walk(ROOT):
    for f in files:
        full = os.path.join(dirpath, f)
        rel = os.path.relpath(full, ROOT).replace("\\", "/")
        if rel.endswith((".tex", ".sty")):
            with open(full, encoding="utf-8", newline="") as fh:
                local[rel] = fh.read().replace("\r\n", "\n")

print("remote docs: %d   local text files: %d" % (len(remote), len(local)))
bad = 0
for rel in sorted(set(local) | set(remote)):
    if rel not in remote:
        print("  MISSING in Overleaf: %s" % rel); bad += 1
    elif rel not in local:
        print("  EXTRA in Overleaf  : %s" % rel); bad += 1
    elif remote[rel] != local[rel]:
        n_r, n_l = len(remote[rel]), len(local[rel])
        print("  MISMATCH %s (remote %d chars, local %d)" % (rel, n_r, n_l)); bad += 1
    else:
        print("  ok  %-44s %6d chars" % (rel, len(local[rel])))

print("\n%s" % ("ALL DOCS ROUND-TRIP EXACTLY" if bad == 0 else "%d PROBLEM(S)" % bad))
sys.exit(1 if bad else 0)

"""Build the CLSI compile request for MS_Fanyang_Report_Slides.

Mirrors web's ClsiManager._finaliseRequest on the mongo path: every doc is an
inline resource whose content is lines.join("\\n"), every fileRef is a url
resource pointing at the filestore.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PID = "6aba0488bacd0f69c4b35e4e"
MARK = "@@JSON@@"

out = subprocess.run(
    ["docker", "exec", "-i", "mongo", "mongosh", "sharelatex", "--quiet",
     "--file", "/dev/stdin"],
    stdin=open(os.path.join(HERE, "clsi_dump.js"), "rb"),
    capture_output=True, text=True, encoding="utf-8",
)
if out.returncode != 0:
    sys.exit("mongosh failed: " + out.stderr[:500])
raw = out.stdout
if raw.count(MARK) != 1:
    sys.exit("expected one %s, got %d" % (MARK, raw.count(MARK)))
tail = raw[raw.index(MARK) + len(MARK):]
data = json.loads(tail[:tail.index("\n")] if "\n" in tail else tail)

resources = []
for d in data["docs"]:
    if d["content"] is None:
        sys.exit("doc missing in mongo: " + d["path"])
    resources.append({"path": d["path"], "content": d["content"]})
    print("  doc   %-42s %6d chars" % (d["path"], len(d["content"])))
for f in data["files"]:
    if f["modified"] is None:
        sys.exit("fileRef created is not a Date: " + f["path"])
    resources.append({
        "path": f["path"],
        "url": "http://127.0.0.1:3009/history/project/%s/hash/%s" % (PID, f["hash"]),
        "modified": f["modified"],
    })
    print("  file  %-42s %s" % (f["path"], f["hash"][:12] + "..."))

request = {"compile": {
    "options": {
        "historyId": PID,                      # a *string* here
        "buildId": "a1b2c3d4e5f6-0f0e0d0c0b0a",  # ^[0-9a-f]+-[0-9a-f]+$
        "compiler": "pdflatex",
        "timeout": 180,
        "draft": False,
        "png2pdf": True,
        "stopOnFirstError": False,
        "check": "logs",
        "syncType": "full",
        "compileGroup": "standard",
    },
    "rootResourcePath": "main.tex",
    "resources": resources,
}}
path = os.path.join(HERE, "clsi_request.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(request, fh, ensure_ascii=True)
print("\nwrote %s  (%d resources)" % (path, len(resources)))

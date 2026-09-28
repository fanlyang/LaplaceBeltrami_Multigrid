"""Emit a mongosh script that creates the Overleaf project for report_slides/.

Shapes copied from the existing decks (probed 2026-09-28):
  db.projects  : rootFolder[0] = {name,_id,docs,fileRefs,folders}
                 docs entry    = {name,_id}
                 fileRefs      = {name,created(BSON Date),rev:0,hash,_id}
  db.docs      : {_id,project_id,lines,rev:1,version:0,ranges:{}}

owner_ref / lastUpdatedBy must be ObjectIds or the project is invisible in the
project list.  docs must be rev:1, never rev:0.
"""
import json, os, secrets, time, datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report_slides")
PROJECT_ID = "6aba0488bacd0f69c4b35e4e"
OWNER_ID = "6aa30275465ac518b59990eb"
NAME = "MS_Fanyang_Report_Slides"

_counter = [secrets.randbelow(0xFFFF)]


def oid():
    """A fresh, valid 24-hex ObjectId string."""
    _counter[0] += 1
    ts = int(time.time()).to_bytes(4, "big").hex()
    rnd = secrets.token_hex(5)
    return ts + rnd + _counter[0].to_bytes(3, "big").hex()


def js(s):
    """JSON string escaping is a subset of JS string escaping."""
    return json.dumps(s, ensure_ascii=True)


def folder_js(name, fid, docs, fileRefs, folders):
    """`folders` is a list of already-rendered JS object strings."""
    docs_js = ",".join("{name:%s,_id:ObjectId(%s)}" % (js(d["name"]), js(d["_id"]))
                       for d in docs)
    refs_js = ",".join(
        "{name:%s,created:new Date(%s),rev:%d,hash:%s,_id:ObjectId(%s)}"
        % (js(r["name"]), js(r["created_iso"]), r["rev"], js(r["hash"]), js(r["_id"]))
        for r in fileRefs)
    return ("{name:%s,_id:ObjectId(%s),docs:[%s],fileRefs:[%s],folders:[%s]}"
            % (js(name), js(fid), docs_js, refs_js, ",".join(folders)))


FIGURES = [
    ("RUB_Logo.pdf", "7aa4000c5cd14600da8fee5b75dda99c48275a6d"),
    ("RUB_label_2000x2000.pdf", "fbe2ff3df3d6316505707980112b5a0b9a6e548e"),
    ("RUB_schriftzug_3600x236.pdf", "cba3fd61719317bb6ba89a1aa44e7987d39674c3"),
]

# ---- collect the text files, in the tree layout -------------------------
main_id = oid()
doc_stmts = []


def read(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def add_doc(rel, doc_id, path):
    """One db.docs document.  lines = content.split('\\n') so that
    lines.join('\\n') reproduces the file byte for byte."""
    content = read(rel).replace("\r\n", "\n")
    lines = content.split("\n")
    doc_stmts.append(
        "db.docs.insertOne({_id:ObjectId(%s),project_id:ObjectId(%s),lines:[%s],"
        "rev:1,version:0,ranges:{}});" % (js(doc_id), js(PROJECT_ID),
                                          ",".join(js(l) for l in lines)))
    print(f"  {path:44s} {len(lines):5d} lines -> {doc_id}")


def frame(name):
    did = oid()
    add_doc(f"_Directory/{name}/slide.tex", did, f"_Directory/{name}/slide.tex")
    return {"name": "slide.tex", "_id": did}


# main.tex at the root
add_doc("main.tex", main_id, "main.tex")

# preface/
preface_docs = [{"name": "slide.tex", "_id": oid()}]
add_doc("preface/slide.tex", preface_docs[0]["_id"], "preface/slide.tex")

# _stylefiles/
style_docs = []
for sty in ("rublkm.sty", "defdb1.sty"):
    did = oid()
    add_doc(f"_stylefiles/{sty}", did, f"_stylefiles/{sty}")
    style_docs.append({"name": sty, "_id": did})

# _Directory/
dir_names = sorted(
    d for d in os.listdir(os.path.join(ROOT, "_Directory"))
    if os.path.isfile(os.path.join(ROOT, "_Directory", d, "slide.tex")))
frame_folders = []
for name in dir_names:
    frame_folders.append(
        folder_js(name, oid(), [frame(name)], [], []))

# _figures/ (binaries)
figure_refs = [{"name": n, "created_iso": "2026-09-28T00:00:00.000Z",
                "rev": 0, "hash": h, "_id": oid()} for n, h in FIGURES]

# ---- the project document ----------------------------------------------
sub = []
sub.append(folder_js("preface", oid(), preface_docs, [], []))
sub.append(folder_js("_Directory", oid(), [], [], frame_folders))
sub.append(folder_js("_stylefiles", oid(), style_docs, [], []))
sub.append(folder_js("_figures", oid(), [], figure_refs, []))
root = folder_js("rootFolder", oid(), [{"name": "main.tex", "_id": main_id}], [], sub)

now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

project = f"""db.projects.insertOne({{
_id:ObjectId({js(PROJECT_ID)}),
name:{js(NAME)},
lastUpdatedBy:ObjectId({js(OWNER_ID)}),
active:true,
readOnly:false,
owner_ref:ObjectId({js(OWNER_ID)}),
collaberator_refs:[],
reviewer_refs:[],
readOnly_refs:[],
pendingEditor_refs:[],
pendingReviewer_refs:[],
publicAccesLevel:"private",
compiler:"pdflatex",
spellCheckLanguage:"en",
deletedByExternalDataSource:false,
description:"",
trashed:[],
tokens:{{}},
tokenAccessReadOnly_refs:[],
tokenAccessReadAndWrite_refs:[],
overleaf:{{history:{{id:{js(PROJECT_ID)},display:true}}}},
editAccessRequests:[],
deletedDocs:[],
collabratecUsers:[],
__v:0,
version:1,
rootDoc_id:ObjectId({js(main_id)}),
lastUpdated:new Date({js(now_iso)}),
lastOpened:new Date({js(now_iso)}),
rootFolder:[{root}]
}});"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insert.js")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(doc_stmts) + "\n" + project + "\n")
print(f"\nwrote {out}: 1 project + {len(doc_stmts)} docs, "
      f"{len(frame_folders)} frames, {len(figure_refs)} figures")
print(f"rootDoc_id = {main_id}")

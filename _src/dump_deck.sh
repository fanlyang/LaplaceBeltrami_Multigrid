#!/bin/bash
# Dump the template files and structural frames of the 15th deck (project 6123a6b044fecc3ee59368aa).
set -u
cd "$(dirname "$0")/.."
MSYS_NO_PATHCONV=1 docker exec mongo mongosh sharelatex --quiet --eval '
const want = {
  "main.tex":                  "6307e4a90f9f5b0568a5c990",
  "rublkm.sty":                "d256986692a281386b0ee13f",
  "defdb1.sty":                "4643fb5103acf0a5019b2e05",
  "preface_slide.tex":         "e75e32f529738393021b2de1",
  "Outline_slide.tex":         "6d3ec326a59de4611ee57fe8",
  "MiniOutline_slide.tex":     "71b7c679238435c888c26b95",
  "Reference_slide.tex":       "b51194a9bcca6b4cae0c24a2"
};
for (const k in want) {
  print("<<<<<FILE " + k + ">>>>>");
  print(db.docs.findOne({_id: ObjectId(want[k])}).lines.join("\n"));
}
' > _src/deck/template.txt 2>&1
wc -l _src/deck/template.txt

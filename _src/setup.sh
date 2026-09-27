#!/bin/bash
# Build the skeleton of report_slides/ : style files verbatim from the 15th deck,
# logo figures copied from the local slides mirror.
set -eu
cd "$(dirname "$0")/.."                       # worktree root
mkdir -p report_slides/_stylefiles report_slides/_figures report_slides/preface report_slides/_Directory

# ---- style files, byte-for-byte from the Overleaf project ----
MSYS_NO_PATHCONV=1 docker exec mongo mongosh sharelatex --quiet --eval \
  'print(db.docs.findOne({_id:ObjectId("d256986692a281386b0ee13f")}).lines.join("\n"))' \
  > report_slides/_stylefiles/rublkm.sty
MSYS_NO_PATHCONV=1 docker exec mongo mongosh sharelatex --quiet --eval \
  'print(db.docs.findOne({_id:ObjectId("4643fb5103acf0a5019b2e05")}).lines.join("\n"))' \
  > report_slides/_stylefiles/defdb1.sty

# ---- logo figures ----
SRC=../slides_work/_figures
for f in RUB_Logo.pdf RUB_label_2000x2000.pdf RUB_schriftzug_3600x236.pdf; do
  if [ -f "$SRC/$f" ]; then cp "$SRC/$f" report_slides/_figures/; echo "copied $f";
  else echo "MISSING $f"; fi
done

echo "--- skeleton ---"
find report_slides -type f | sort

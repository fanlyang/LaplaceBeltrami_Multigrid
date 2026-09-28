// Emit {path: content} for every doc in the new project, as one JSON object.
const PID = "6aba0488bacd0f69c4b35e4e";
const p = db.projects.findOne({_id: ObjectId(PID)});
const out = {};
function walk(f, path) {
  (f.docs || []).forEach(d => {
    const doc = db.docs.findOne({_id: ObjectId(d._id)});
    const rel = path ? path + "/" + d.name : d.name;
    out[rel] = doc ? doc.lines.join("\n") : null;
  });
  (f.folders || []).forEach(g => walk(g, path ? path + "/" + g.name : g.name));
}
walk(p.rootFolder[0], "");
print("@@JSON@@" + JSON.stringify(out));

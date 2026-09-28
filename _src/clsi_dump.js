// Emit everything CLSI needs for the new project, as one JSON blob:
//   {docs: [{path, content}], files: [{path, hash, modified}]}
const PID = "6aba0488bacd0f69c4b35e4e";
const p = db.projects.findOne({_id: ObjectId(PID)});
const docs = [], files = [];
function walk(f, path) {
  (f.docs || []).forEach(d => {
    const doc = db.docs.findOne({_id: ObjectId(d._id)});
    docs.push({ path: path ? path + "/" + d.name : d.name,
                content: doc ? doc.lines.join("\n") : null });
  });
  (f.fileRefs || []).forEach(x => {
    files.push({ path: path ? path + "/" + x.name : x.name,
                 hash: x.hash,
                 modified: (x.created instanceof Date) ? x.created.getTime() : null });
  });
  (f.folders || []).forEach(g => walk(g, path ? path + "/" + g.name : g.name));
}
walk(p.rootFolder[0], "");
print("@@JSON@@" + JSON.stringify({docs: docs, files: files}));

const PID = "6aba0488bacd0f69c4b35e4e";
const OWNER = "6aa30275465ac518b59990eb";

print("--- the owner's project list (this is what the dashboard shows) ---");
db.projects.find({owner_ref: ObjectId(OWNER), trashed: {$ne: true}}, {name: 1}).forEach(
  p => print("   " + (p._id.toString() === PID ? ">>> " : "    ") + p.name));

const p = db.projects.findOne({_id: ObjectId(PID)});
if (!p) { print("PROJECT MISSING"); quit(); }
print("--- identity types (must both be ObjectId) ---");
print("   owner_ref     " + p.owner_ref.constructor.name);
print("   lastUpdatedBy " + p.lastUpdatedBy.constructor.name);
print("   rootDoc_id    " + p.rootDoc_id.constructor.name + " = " + p.rootDoc_id);
print("   overleaf.history.id = " + JSON.stringify(p.overleaf.history.id) + " (" + typeof p.overleaf.history.id + ")");
print("   compiler = " + p.compiler);

print("--- tree ---");
let nDocs = 0, nRefs = 0, badDates = 0, badDocRev = 0;
function walk(f, path) {
  (f.docs || []).forEach(d => {
    nDocs++;
    const doc = db.docs.findOne({_id: ObjectId(d._id)});
    if (!doc) { print("   !! doc missing: " + path + "/" + d.name); return; }
    if (doc.rev === 0) { badDocRev++; print("   !! rev:0 doc: " + d.name); }
  });
  (f.fileRefs || []).forEach(x => {
    nRefs++;
    if (!(x.created instanceof Date)) { badDates++; print("   !! fileRef created is " + typeof x.created + ": " + x.name); }
  });
  if (path === "") print("   root docs: " + (f.docs || []).map(d => d.name).join(", "));
  (f.folders || []).forEach(g => {
    if (path === "") print("   folder " + g.name + "  (" + (g.docs || []).length + " docs, " + (g.folders || []).length + " subfolders, " + (g.fileRefs || []).length + " fileRefs)");
    walk(g, path + "/" + g.name);
  });
}
walk(p.rootFolder[0], "");
print("--- summary ---");
print("   docs in tree   = " + nDocs + "   (db.docs for project = " + db.docs.countDocuments({project_id: ObjectId(PID)}) + ")");
print("   fileRefs       = " + nRefs);
print("   fileRef created not a Date = " + badDates);
print("   docs with rev:0            = " + badDocRev);
print("   rootFolder length = " + p.rootFolder.length);

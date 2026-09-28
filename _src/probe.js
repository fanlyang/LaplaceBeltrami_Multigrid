// Probe: owner identity + the logo fileRefs (name -> hash) of the source decks.
const src = ["6123a6b044fecc3ee59368aa", "6ab00545ae1d68efd7bacc07", "6ab97e78f1eb520f9e954b58"];
for (const id of src) {
  const p = db.projects.findOne({_id: ObjectId(id)});
  print("=== " + p.name + "  " + id);
  print("   owner_ref      = " + p.owner_ref + "  (" + (p.owner_ref && p.owner_ref.constructor ? p.owner_ref.constructor.name : typeof p.owner_ref) + ")");
  print("   lastUpdatedBy  = " + p.lastUpdatedBy + "  (" + (p.lastUpdatedBy && p.lastUpdatedBy.constructor ? p.lastUpdatedBy.constructor.name : typeof p.lastUpdatedBy) + ")");
  print("   compiler       = " + p.compiler);
  const keys = Object.keys(p).filter(k => k !== "rootFolder");
  print("   keys: " + keys.join(", "));
  function walk(f, path) {
    (f.fileRefs || []).forEach(x => print("   BLB " + path + "/" + x.name + "  hash=" + x.hash + "  created=" + (x.created instanceof Date ? "Date" : typeof x.created)));
    (f.folders || []).forEach(g => walk(g, path + "/" + g.name));
  }
  walk(p.rootFolder[0], "");
}

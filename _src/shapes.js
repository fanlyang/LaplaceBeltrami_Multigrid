// Probe the exact document shapes so the hand-built project matches.
const p = db.projects.findOne({_id: ObjectId("6123a6b044fecc3ee59368aa")});
print("--- project keys ---");
print(Object.keys(p).join(", "));
print("--- scalars worth copying ---");
["active","readOnly","publicAccesLevel","spellCheckLanguage","deletedByExternalDataSource",
 "trashed","version","compiler","description","overleaf"].forEach(k => print("   " + k + " = " + JSON.stringify(p[k])));
print("   collaberator_refs = " + JSON.stringify(p.collaberator_refs));
print("   reviewer_refs = " + JSON.stringify(p.reviewer_refs));
print("   readOnly_refs = " + JSON.stringify(p.readOnly_refs));
print("   tokens = " + JSON.stringify(p.tokens));
print("   rootDoc_id = " + p.rootDoc_id + " (" + (p.rootDoc_id && p.rootDoc_id.constructor.name) + ")");
print("   lastUpdated = " + p.lastUpdated + " (" + (p.lastUpdated && p.lastUpdated.constructor.name) + ")");

print("--- rootFolder[0] keys ---");
print(Object.keys(p.rootFolder[0]).join(", "));
const f = p.rootFolder[0];
print("   name=" + JSON.stringify(f.name) + "  _id=" + f._id + " (" + (f._id && f._id.constructor.name) + ")");
print("   docs entry: " + JSON.stringify(f.docs[0]));
print("   folders entry keys: " + Object.keys(f.folders[0]).join(", "));
print("   folders[0]: name=" + JSON.stringify(f.folders[0].name) + " _id=" + f.folders[0]._id + " (" + (f.folders[0]._id && f.folders[0]._id.constructor.name) + ")");
print("   folders[0].docs entry: " + JSON.stringify(f.folders[0].docs[0]));
print("   folders[0] all keys: " + JSON.stringify(Object.keys(f.folders[0])));

// a nested folder with a fileRef, to see that shape
function findFig(fo, path) {
  (fo.fileRefs || []).forEach(x => print("   fileRef @ " + path + ": " + JSON.stringify(x) + "  keys=" + Object.keys(x).join(",")));
  (fo.folders || []).forEach(g => findFig(g, path + "/" + g.name));
}
findFig(p.rootFolder[0], "");

print("--- one db.docs document ---");
const d = db.docs.findOne({project_id: p._id});
print("   keys: " + Object.keys(d).join(", "));
print("   _id=" + d._id + " (" + d._id.constructor.name + ")  project_id=" + d.project_id + " (" + d.project_id.constructor.name + ")  rev=" + d.rev);
print("   ranges type=" + (Array.isArray(d.ranges) ? "array" : typeof d.ranges) + " len=" + (d.ranges && d.ranges.length));
print("   ranges sample: " + JSON.stringify((d.ranges || []).slice(0, 2)));
print("   version=" + d.version + " (" + typeof d.version + ")  lines=" + d.lines.length);

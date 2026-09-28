// Initialize the new project's history, in-process, with no credential.
//
// Must run with cwd = /overleaf/services/history-v1, because
// storage/lib/mongodb.js reads config.mongo.uri and node-config looks for
// ./config relative to cwd.  process.exit(0) is required: the mongodb client
// keeps the event loop alive, so without it the call succeeds and then hangs.
const { chunkStore } = require("/overleaf/services/history-v1/storage");

chunkStore.initializeProject("6aba0488bacd0f69c4b35e4e")
  .then(r => { console.log("INIT_OK " + JSON.stringify(r)); process.exit(0); })
  .catch(e => { console.log("INIT_ERR " + e.message); process.exit(1); });

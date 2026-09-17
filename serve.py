#!/usr/bin/env python3
"""Serves the multigrid evidence bundle on localhost.

    python serve.py [port]        # default 8000

The repository is not served as a directory tree. Every URL below is mapped
explicitly onto one file, so the server exposes the two write-ups and their
evidence and nothing else -- no source tree, no .git, no build directories.
That also keeps the URLs stable if the files move within the repository.

Only routes listed in ROUTES are reachable; everything else is a 404.
"""

import http.server
import os
import socketserver
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# url path -> (file relative to the repository root, content type)
ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/comparison": ("bench/report.html", "text/html; charset=utf-8"),
    "/results.csv": ("bench/results-d3.csv", "text/csv; charset=utf-8"),
    "/figure.pdf": ("bench/scaling.pdf", "application/pdf"),
    "/figure.png": ("bench/scaling.png", "image/png"),
    "/thesis-section.pdf": ("latex/mg-results-standalone.pdf", "application/pdf"),
    "/thesis-section.tex": ("latex/mg-results.tex", "text/plain; charset=utf-8"),
}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        entry = ROUTES.get(path)
        if entry is None:
            self.send_error(404, "Not found")
            return

        rel, ctype = entry
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            self.send_error(404, "Missing: " + rel)
            return

        with open(full, "rb") as fh:
            body = fh.read()

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # One quiet line per request, so the terminal stays readable.
        sys.stderr.write("  %s\n" % (fmt % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    # Loopback only: this serves the project's evidence, so it is deliberately
    # not reachable from the local network.
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print("serving the multigrid evidence bundle")
        for path in sorted(ROUTES):
            print("  http://localhost:%d%s" % (port, path))
        print("\nCtrl-C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

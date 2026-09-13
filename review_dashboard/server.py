"""Review dashboard server (stdlib only).

Serves the static dashboard + page crops + each book's final zip, and
exposes the human-review API backed by qbank.review (append-only
ledgers, fingerprinted decisions, verified edits).

    python3 review_dashboard/server.py [--port 8000]
"""
import argparse
import json
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from qbank import config, review  # noqa: E402
from qbank.export import build_final_zip, gate_final_zip  # noqa: E402


def _maybe_export(book: str):
    """Same contract as the Railway dashboard: the moment this book's
    last REVIEW table is decided its gate opens and its zip rebuilds,
    so Download is ready without a re-run. Never raises."""
    try:
        if not gate_final_zip(config.OUTPUT_ROOT, book)["locked"]:
            return build_final_zip(config.OUTPUT_ROOT, subject=book)
    except Exception:                                # noqa: BLE001
        pass
    return None


def _crop_file(name: str):
    for _book, root in books():
        cand = root / "crops" / name
        if cand.is_file():
            return cand
    legacy = HERE / "crops" / name
    return legacy if legacy.is_file() else None


def books():
    roots = [d for d in sorted(REPO.glob("qbank_output_*"))
             if (d / "split").is_dir()]
    single = config.OUTPUT_ROOT
    if (single / "split").is_dir() and single not in roots:
        roots.append(single)
    seen = set()
    for root in roots:
        for sub in sorted((root / "split").iterdir()):
            if sub.is_dir() and sub.name not in seen:
                seen.add(sub.name)
                yield sub.name, root


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def do_GET(self):
        if self.path == "/api/queue":
            items = []
            for book, root in books():
                items += review.review_tables(root)
            return self._json(items)
        if self.path == "/api/audit":
            flags = []
            for book, root in books():
                rep = root / "data" / "audit_report.jsonl"
                if rep.exists():
                    for line in rep.read_text().splitlines():
                        if line.strip():
                            row = json.loads(line)
                            row["book"] = book
                            flags.append(row)
            return self._json(flags)
        if self.path.startswith("/zip/"):
            book = self.path[len("/zip/"):].strip("/").upper()
            for b, root in books():
                if b == book:
                    z = root / f"final_export_{book}.zip"
                    if z.exists():
                        self.send_response(200)
                        self.send_header("Content-Type", "application/zip")
                        self.send_header(
                            "Content-Disposition",
                            f'attachment; filename="{b}_final_export.zip"')
                        body = z.read_bytes()
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        return self.wfile.write(body)
            return self._json({"error": "no zip for book"}, 404)
        if self.path.startswith("/api/question/"):
            qid = self.path[len("/api/question/"):].strip("/")
            got = review.find_question(config.OUTPUT_ROOT, qid)
            if got is None:
                return self._json({"ok": False,
                                   "error": f"no question {qid} on disk"},
                                  404)
            return self._json(got)
        if self.path.startswith("/api/lookup"):
            import urllib.parse as _up
            term = _up.parse_qs(_up.urlparse(self.path).query
                                ).get("term", [""])[0]
            return self._json(review.lookup_questions(
                config.OUTPUT_ROOT, term))
        if self.path.startswith("/assets/"):
            import urllib.parse as _up
            rel = _up.unquote(self.path[len("/assets/"):])
            base = config.ASSETS_DIR.resolve()
            f = (base / rel).resolve()
            if f.is_relative_to(base) and f.is_file():
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/webp")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            return self._json({"error": "no such asset"}, 404)
        if self.path.startswith("/crops/"):
            name = self.path[len("/crops/"):].strip("/")
            if "/" not in name and name.endswith(".png"):
                f = _crop_file(name)
                if f:
                    body = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
            return self._json({"error": "no such crop"}, 404)
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        root = self._root(body.get("book"))
        if root is None:
            return self._json({"error": "unknown book"}, 404)
        if self.path == "/api/decision":
            row = review.record_decision(
                root, body["book"], body["q_id"], body["table_id"],
                body.get("action", "approve"), body.get("note", ""))
            out = _maybe_export(body["book"])
            resp = dict(row)
            resp["zip_built"] = bool(out and out.get("ok"))
            return self._json(resp)
        if self.path == "/api/edit":
            res = review.apply_table_edit(
                root, body["book"], body["q_id"], body["table_id"],
                body.get("markdown", ""))
            if res.get("ok") and body.get("action"):
                review.record_decision(
                    root, body["book"], body["q_id"], body["table_id"],
                    body["action"], "saved via edit")
                out = _maybe_export(body["book"])
                res["zip_built"] = bool(out and out.get("ok"))
            return self._json(res)
        if self.path == "/api/table/delete":
            try:
                res = review.delete_table(
                    root, body["book"], body["q_id"], body["table_id"])
            except KeyError as exc:
                return self._json({"ok": False,
                                   "why": f"missing field {exc}"}, 400)
            if res.get("ok"):
                out = _maybe_export(body["book"])
                res["zip_built"] = bool(out and out.get("ok"))
            return self._json(res)
        if self.path.startswith("/api/question/") \
                and self.path.endswith("/edit"):
            qid = self.path[len("/api/question/"):-len("/edit")].strip("/")
            return self._json(review.apply_question_edit(
                root, body.get("book", ""), qid, body.get("patch"),
                note=body.get("note", "saved via question editor")))
        return self._json({"error": "no route"}, 404)

    def _root(self, book):
        for b, root in books():
            if b == (book or "").upper():
                return root
        return None

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"review dashboard on 0.0.0.0:{args.port}")
    srv.serve_forever()

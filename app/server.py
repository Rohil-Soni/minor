"""Photoholic GUI: a local web app to review a finished run.

    ai\\.venv\\Scripts\\python.exe app\\server.py runs\\Photos_v2
    ai\\.venv\\Scripts\\python.exe app\\server.py "D:\\Photos\\Trip\\.photoholic" --port 8765

Shows every group with all of its photos, and compares any photo side by side with the
group's pick (synced zoom/pan). Picks you change are saved to ``user_picks.json`` in the run
folder. Photos are only read: nothing is moved, renamed or deleted.

Only binds to 127.0.0.1 and only serves photos listed in the run (looked up by id).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "ai"))

from photoholic_ai import bridge  # noqa: E402
from photoholic_ai.grouping import parse_capture_time  # noqa: E402
from photoholic_ai.report import parse_number  # noqa: E402

log = logging.getLogger("photoholic.gui")

STATIC_DIR = APP_DIR / "static"
PICKS_NAME = "user_picks.json"
CACHE_DIR = "gui_cache"
# long side in px; "full" serves the original file
SIZES = {"thumb": 360, "preview": 2048}
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                 ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}  # fmt: skip
_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def _name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _clock(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Run:
    """A finished pipeline run, loaded from its run folder."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        results_path = self.run_dir / bridge.RESULTS_NAME
        if not results_path.is_file():
            raise FileNotFoundError(
                f"no {bridge.RESULTS_NAME} in {self.run_dir}. Run ai/run_pipeline.py first."
            )
        self.results = json.loads(results_path.read_text(encoding="utf-8"))
        filtered = bridge.load_filtered(self.run_dir / bridge.FILTERED_NAME)
        scored, _ = bridge.load_scored(self.run_dir)
        self.lock = threading.Lock()

        self.records = {r["id"]: r for r in filtered["images"]}
        self.scores = {r["id"]: r for r in scored["images"]}
        # id -> file on disk; the only files this server will ever read
        self.files: dict[str, Path] = {i: Path(r["path"]) for i, r in self.records.items()}
        self.rejected = []
        for n, r in enumerate(filtered.get("rejected", []), start=1):
            rid = f"rej_{n:06d}"
            self.files[rid] = Path(r["path"])
            self.rejected.append(
                {"id": rid, "name": _name(r["path"]), "reason": r.get("reason"),
                 "sharpness": r.get("sharpness_score")}
            )  # fmt: skip
        self.picks = self._load_picks()

    def _load_picks(self) -> dict[str, str]:
        path = self.run_dir / PICKS_NAME
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): v for k, v in data.get("picks", {}).items()}

    def _member(self, m: dict[str, Any]) -> dict[str, Any]:
        rec = self.records.get(m["id"], {})
        sc = self.scores.get(m["id"], {})
        t = parse_capture_time(rec)
        return {
            "id": m["id"],
            "name": _name(m["path"]),
            "rank": m["rank"],
            "final": m.get("final_score"),
            "sharp_norm": m.get("sharpness_norm"),
            "aes_norm": m.get("aesthetic_norm"),
            "aes_mean": sc.get("aesthetic_mean"),
            "aes_std": sc.get("aesthetic_std"),
            "sharpness": rec.get("sharpness_score"),
            "time": t,
            "clock": _clock(t),
            "iso": parse_number(rec.get("iso")),
            "shutter": rec.get("shutter_speed"),
            "focal": rec.get("focal_length"),
            "raw": _name(rec["arw_path"]) if rec.get("arw_path") else None,
        }

    def payload(self) -> dict[str, Any]:
        groups = []
        for c in self.results["clusters"]:
            members = [self._member(m) for m in c["members"]]
            times = [m["time"] for m in members if m["time"] is not None]
            cid = str(c["cluster_id"])
            groups.append(
                {
                    "id": c["cluster_id"],
                    "best_id": c["best_id"],
                    "pick_id": self.picks.get(cid, c["best_id"]),
                    "start": min(times) if times else None,
                    "span": (max(times) - min(times)) if len(times) > 1 else 0.0,
                    "members": members,
                }
            )
        return {
            "run_dir": str(self.run_dir),
            "stats": self.results["stats"],
            "weights": self.results["weights"],
            "model": self.results.get("model"),
            "groups": groups,
            "rejected": self.rejected,
            "unscored": [
                {"name": _name(r["path"]), "status": r["status"]}
                for r in self.results.get("unscored", [])
            ],
        }

    def set_pick(self, cluster_id: int, image_id: str) -> dict[str, str]:
        cluster = next((c for c in self.results["clusters"] if c["cluster_id"] == cluster_id), None)
        if cluster is None:
            raise KeyError(f"no group {cluster_id}")
        if image_id not in {m["id"] for m in cluster["members"]}:
            raise KeyError(f"{image_id} is not in group {cluster_id}")
        with self.lock:
            if image_id == cluster["best_id"]:
                self.picks.pop(str(cluster_id), None)  # back to the automatic pick
            else:
                self.picks[str(cluster_id)] = image_id
            bridge.write_json(
                self.run_dir / PICKS_NAME,
                {
                    "note": "manual picks from the Photoholic GUI; group id -> image id",
                    "picks": self.picks,
                    "files": {k: str(self.files[v]) for k, v in self.picks.items()},
                },
            )
        return dict(self.picks)

    def image_bytes(self, image_id: str, size: str) -> tuple[bytes, str]:
        """JPEG bytes for ``thumb`` / ``preview`` (cached), or the original for ``full``."""
        if image_id not in self.files:
            raise KeyError(image_id)
        src = self.files[image_id]
        if size == "full":
            return src.read_bytes(), "image/jpeg"
        if size not in SIZES:
            raise KeyError(size)
        cache = self.run_dir / CACHE_DIR / size / f"{image_id}.jpg"
        if not cache.is_file():
            cache.parent.mkdir(parents=True, exist_ok=True)
            long_side = SIZES[size]
            with Image.open(src) as im:
                im.draft("RGB", (long_side, long_side))
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85 if size == "thumb" else 90)
            tmp = cache.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_bytes(buf.getvalue())
            os.replace(tmp, cache)
        return cache.read_bytes(), "image/jpeg"


class Handler(BaseHTTPRequestHandler):
    run: Run  # set on the subclass in make_server

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=86400" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path == "/api/run":
                return self._json(self.run.payload())
            if path.startswith("/img/"):
                image_id = path.removeprefix("/img/")
                size = parse_qs(url.query).get("size", ["thumb"])[0]
                if not _ID.match(image_id):
                    return self._error(HTTPStatus.BAD_REQUEST, "bad id")
                try:
                    body, ctype = self.run.image_bytes(image_id, size)
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, "unknown image or size")
                except OSError as e:
                    return self._error(HTTPStatus.NOT_FOUND, f"cannot read image: {e}")
                return self._send(HTTPStatus.OK, body, ctype, cache=True)
            return self._static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents or not target.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        ctype = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        self._send(HTTPStatus.OK, target.read_bytes(), ctype)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/pick":
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(min(length, 10_000)) or b"{}")
            picks = self.run.set_pick(int(data["group"]), str(data["id"]))
        except (ValueError, KeyError, TypeError) as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e).strip("'\""))
        self._json({"picks": picks})


def make_server(run: Run, port: int) -> ThreadingHTTPServer:
    handler = type("RunHandler", (Handler,), {"run": run})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Review a Photoholic run in the browser.")
    p.add_argument("run_dir", type=Path, help="the run folder (.photoholic or --run-dir)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        run = Run(args.run_dir)
        server = make_server(run, args.port)
    except (FileNotFoundError, bridge.BridgeError) as e:
        log.error("%s", e)
        return 1
    except OSError as e:
        log.error("cannot listen on port %d (%s). Try --port %d", args.port, e, args.port + 1)
        return 1

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    n = len(run.results["clusters"])
    print(f"Photoholic GUI: {n} groups from {run.run_dir}")
    print(f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

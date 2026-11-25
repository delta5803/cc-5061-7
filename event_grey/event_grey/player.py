"""Browser-based motion-analysis player for event grayscale reconstruction.

Runs on the Python standard library alone (``http.server``); no Tk/Qt is
needed, so it works on headless machines and is reachable from any browser.
The GUI is the main entry point of the toolkit -- the CLI only prepares
datasets, videos and reports.

Endpoints serve data computed by :class:`event_grey.analysis.AnalysisSession`;
the browser never invents pixel values, it renders what the backend returns.
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

from .analysis import AnalysisSession
from .simulator import SCENES

WEB_ROOT = __file__.rsplit("/", 1)[0].rsplit("/", 1)[0] + "/web"


def _png(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def make_handler(session_holder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path, qs = parsed.path, parse_qs(parsed.query)
            try:
                self._route(path, qs)
            except Exception as exc:  # surfaced in the GUI, never silent
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def _route(self, path, qs):
            if path in ("/", "/index.html"):
                with open(WEB_ROOT + "/index.html", "rb") as f:
                    self._send(f.read(), "text/html; charset=utf-8")
                return
            if path == "/app.js":
                with open(WEB_ROOT + "/app.js", "rb") as f:
                    self._send(f.read(), "application/javascript")
                return
            if path == "/style.css":
                with open(WEB_ROOT + "/style.css", "rb") as f:
                    self._send(f.read(), "text/css")
                return

            s = session_holder["session"]
            if s is None:
                self._json({"error": "no dataset loaded"}, 409)
                return

            if path == "/api/state":
                self._json({
                    "shape": s.shape,
                    "duration": s.duration,
                    "fps": s.fps,
                    "n_frames": len(s.output_times),
                    "n_events": int(s.events.size),
                    "contrast": s.contrast,
                    "scenes": list(SCENES.keys()),
                    "current_scene": s.bundle["scene_name"],
                    "output_times": s.output_times.tolist(),
                    "anchor_times": [a["t"] for a in s.anchors],
                    "anomalies": s.anomalies,
                    "anchor_summary": s.anchor_summary(),
                    "trace_consistency": getattr(
                        s, "trace_consistency",
                        {"max_abs_diff": 0.0, "pixels_checked": 0}),
                })
            elif path == "/api/reload":
                scene = qs.get("scene", [s.bundle["scene_name"]])[0]
                fps = float(qs.get("fps", [s.fps])[0])
                coverage = float(qs.get("coverage", [0.15])[0])
                corrupt = float(qs.get("corrupt", [0.0])[0])
                session_holder["session"] = AnalysisSession.from_simulation(
                    scene_name=scene, fps=fps, anchor_coverage=coverage,
                    corrupt_fraction=corrupt,
                )
                self._json({"ok": True})
            elif path == "/api/frame":
                t = float(qs.get("t", [0])[0])
                fr = s.reconstructed_frame(t)
                show_ref = qs.get("ref", ["0"])[0] == "1"
                payload = _png(fr["rgb"])
                if show_ref:
                    ref = s.reference_frame(fr["t"])
                    gap = np.full((s.shape[0], 4, 3), 255, np.uint8)
                    payload = _png(np.concatenate([fr["rgb"], gap,
                                                   np.repeat(ref[:, :, None], 3, 2)], 1))
                self._send(payload, "image/png")
            elif path == "/api/events":
                t0 = float(qs.get("t0", [0])[0])
                t1 = float(qs.get("t1", [0.05])[0])
                win = s.event_window(t0, t1)
                self._send(_png(win.pop("image")), "image/png")
            elif path == "/api/event_stats":
                t0 = float(qs.get("t0", [0])[0])
                t1 = float(qs.get("t1", [0.05])[0])
                win = s.event_window(t0, t1)
                win.pop("image")
                for k in ("on_count", "off_count", "net"):
                    win[k] = win[k].tolist()
                self._json(win)
            elif path == "/api/pixel":
                x = int(qs.get("x", [0])[0]); y = int(qs.get("y", [0])[0])
                tr = s.pixel_trace(x, y)
                pe = tr["events"]
                self._json({
                    "x": x, "y": y,
                    "times": tr["times"].tolist(),
                    "raw": tr["raw_integration"].tolist(),
                    "corrected": tr["corrected"].tolist(),
                    "events": [
                        {"t": float(e["t"]), "p": int(e["p"])} for e in pe
                    ],
                    "anchor_points": [
                        {"t": float(t), "logL": float(v), "kind": k,
                         "drift": float(d)}
                        for (t, v, k, d) in tr["anchor_points"]
                    ],
                    "count_total": int(pe.size),
                    "count_on": int((pe["p"] == 1).sum()) if pe.size else 0,
                    "count_off": int((pe["p"] == 0).sum()) if pe.size else 0,
                })
            elif path == "/api/anchors":
                self._json({"anchors": s.anchor_summary(),
                            "anomalies": s.anomalies})
            else:
                self._json({"error": "unknown endpoint"}, 404)

    return Handler


def run_server(host="127.0.0.1", port=8765, scene="moving_disk", fps=20.0,
               coverage=0.15, corrupt=0.0, open_browser=False):
    holder = {}
    holder["session"] = AnalysisSession.from_simulation(
        scene_name=scene, fps=fps, anchor_coverage=coverage,
        corrupt_fraction=corrupt,
    )
    httpd = ThreadingHTTPServer((host, port), make_handler(holder))
    url = f"http://{host}:{port}/"
    print(f"Event grayscale player running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: _open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def _open(url):
    import shutil
    import subprocess
    for browser in ("xdg-open", "open"):
        if shutil.which(browser):
            subprocess.Popen([browser, url])
            return


def main(argv=None):
    import argparse
    from .simulator import SCENES

    p = argparse.ArgumentParser(
        prog="event-gray-player",
        description="Graphical motion-analysis player for event grayscale "
                    "reconstruction (main entry point of the toolkit).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--scene", choices=list(SCENES), default="moving_disk")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--coverage", type=float, default=0.15)
    p.add_argument("--corrupt", type=float, default=0.0)
    p.add_argument("--browser", action="store_true")
    args = p.parse_args(argv)
    run_server(host=args.host, port=args.port, scene=args.scene,
               fps=args.fps, coverage=args.coverage, corrupt=args.corrupt,
               open_browser=args.browser)


if __name__ == "__main__":
    main()

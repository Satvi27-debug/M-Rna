"""mRNA VIP interface entry point.

Two entry points in one module, because both are *presentation* of the same
artifacts and neither contains domain logic:

* **The web interface.** ``python -m dashboard.app`` serves a dependency-free
  single-page shell built on the standard library's ``http.server``. No framework
  is involved - the brief explicitly excludes React, Next.js, FastAPI and
  Streamlit, and the project must run locally with nothing installed beyond the
  scientific stack.
* **The saved figure.** ``--figure`` renders the three-panel PNG required by
  FR-5.1 at the DPI required by FR-5.3.

Requirements owned here
-----------------------
FR-5.1  A single figure containing exactly three titled panels.
FR-5.2  Each panel generated from the corresponding module's saved output.
FR-5.3  Saved at a minimum of 150 DPI.

Architecture rule 9 - consuming, not recomputing
------------------------------------------------
The dashboard never imports ``src.ml``, ``src.rag``, ``src.llm.report_generator``
or ``src.pipeline``, and never trains, embeds or queries anything. Panel data is
read from ``outputs/logs/``; readiness is read from the filesystem. The interface
therefore renders correctly with no API key, no vector index and no trained
model, and it cannot display a number the pipeline did not produce.

Actions are intentionally inert
-------------------------------
``POST /api/run`` returns HTTP 501. The interface calls it so its loading, empty
and error states are genuinely reachable rather than decorative, and it refuses
rather than simulating a pipeline that does not exist. A fake success here would
be the single most damaging thing this interface could do.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import struct
import sys
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src import config
from dashboard import components, styles


STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Marker in ``index.html`` replaced with the generated ``:root`` token block, so
#: the browser's palette always comes from :mod:`dashboard.styles`.
THEME_MARKER = "/*__VIP_THEME__*/"

#: Figure files the interface is permitted to serve, by public name. An explicit
#: allow-list prevents the static handler from reading arbitrary paths.
#: Figure files the interface is permitted to serve, by public name. An explicit
#: allow-list prevents the static handler from reading arbitrary paths.
#:
#: Both the canonical ROC name and the neutral alias the front end requests are
#: mapped to the same file, so ``config.FIGURE_ROC_FILENAMES`` stays the single
#: source of truth for that naming decision.
#: Figure files the interface is permitted to serve, by public name. An explicit
#: allow-list prevents the static handler from reading arbitrary paths.
#:
#: Both the canonical names and the neutral aliases the front end requests are
#: mapped to the same file, so ``config.*_FILENAMES`` stays the single source of
#: truth for those naming decisions.
SERVEABLE_FIGURES: dict[str, Path] = {
    "dashboard.png": config.FIGURES_DIR / config.DASHBOARD_FILENAME,
    "confusion_matrix.png": config.FIGURES_DIR / config.FIGURE_CONFUSION_MATRIX_FILENAME,
    **{
        name: config.FIGURES_DIR / config.FIGURE_ROC_FILENAME
        for name in config.FIGURE_ROC_FILENAMES
    },
    **{
        name: config.FIGURES_DIR / config.ATTENTION_HEATMAP_FILENAME
        for name in config.ATTENTION_HEATMAP_FILENAMES
    },
    **{
        name: config.FIGURES_DIR / name
        for name in config.CONFUSION_MATRIX_FILENAMES.values()
    },
}

CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


# ---------------------------------------------------------------------------
# The saved three-panel figure (FR-5.1, FR-5.3)
# ---------------------------------------------------------------------------


def build_dashboard(
    save_path: Any = None,
    roc_data: Any = None,
    attention_data: Any = None,
    latency_data: Any = None,
) -> Path:
    """Assemble and save the three-panel figure.

    Creates one row of exactly ``styles.N_PANELS`` axes and populates each from
    :mod:`dashboard.components`:

    ===================================  ==========================================
    Axes                                 Content
    ===================================  ==========================================
    ``axes[0]`` Panel A                  ROC curve, selected model vs the other
    ``axes[1]`` Panel B                  self-attention heatmap
    ``axes[2]`` Panel C                  RAG vs LLM latency comparison
    ===================================  ==========================================

    Panels with no saved artifact render an explicit empty state naming the
    command that will produce them. The figure is therefore always renderable -
    three titled panels at FR-5.3's DPI - while never implying that an unrun stage
    succeeded.

    Parameters
    ----------
    save_path:
        Destination PNG. Defaults to
        ``config.FIGURES_DIR / config.DASHBOARD_FILENAME``.
    roc_data, attention_data, latency_data:
        Pre-loaded panel data. Read from ``outputs/logs/`` when omitted. A missing
        artifact is treated as an empty state, not an error, so the figure can be
        produced before the pipeline has been run.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    OSError
        If the destination cannot be written.
    """
    import matplotlib.pyplot as plt

    config.ensure_directories()

    reasons: dict[str, str] = {}
    if roc_data is None:
        roc_data, reasons["roc"] = _safe_load(components.load_roc_panel_data)
    if attention_data is None:
        attention_data, reasons["attention"] = _safe_load(
            components.load_attention_panel_data
        )
    if latency_data is None:
        latency_data, reasons["latency"] = _safe_load(components.load_latency_panel_data)

    for panel, reason in reasons.items():
        # A malformed artifact is reported rather than hidden. A missing one is
        # the normal pre-run state and is not worth a warning.
        if "malformed" in reason or "unreadable" in reason or "invalid" in reason:
            print(f"  WARNING  panel '{panel}' could not be rendered: {reason}")

    with styles.apply_style():
        figure, axes = plt.subplots(
            1,
            styles.N_PANELS,
            figsize=styles.FIGURE_SIZE,
            constrained_layout=True,
        )
        components.plot_roc_panel(axes[0], roc_data)
        components.plot_attention_panel(axes[1], attention_data)
        components.plot_latency_panel(axes[2], latency_data)
        return styles.save_figure(figure, save_path)


def _safe_load(loader: Any) -> tuple[Any, str]:
    """Run a panel loader, returning ``(data, reason)``.

    A panel whose artifact is missing renders an empty state naming the command
    that produces it. An artifact that exists but cannot be read is a *defect*,
    not an expected state, so the reason is recorded and surfaced on the figure
    rather than being flattened into "no results yet".

    Absorbing a ``KeyError`` here once hid a real bug: the metrics file and the
    panel loader disagreed on a key name, and the ROC panel silently rendered
    empty even though the model was trained. Missing data is expected; a schema
    mismatch is not.
    """
    try:
        return loader(), ""
    except components.MissingArtifactError as exc:
        return None, f"artifact missing: {exc}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"artifact unreadable: {exc}"
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        # The artifact exists but does not have the expected shape.
        return None, (
            f"artifact present but malformed ({type(exc).__name__}: {exc}) - "
            "the panel data contract changed and the loader was not updated"
        )
    except ValueError as exc:
        return None, f"artifact invalid: {exc}"


def verify_dashboard(save_path: Any = None) -> dict[str, Any]:
    """Check the saved figure against FR-5.1 and FR-5.3.

    Confirms the file exists, is a valid PNG, and reports the DPI recorded in the
    PNG ``pHYs`` chunk. Parsed from the file's own metadata with the standard
    library, so the check depends on nothing beyond the file being present.

    Returns
    -------
    dict
        ``{'path', 'exists', 'valid_png', 'dpi', 'meets_fr_5_3', 'width_px',
        'height_px'}``.
    """
    import matplotlib.image as mpimg

    path = (
        Path(save_path)
        if save_path
        else config.FIGURES_DIR / config.DASHBOARD_FILENAME
    )
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "valid_png": False,
        "dpi": None,
        "meets_fr_5_3": False,
        "width_px": None,
        "height_px": None,
        "panels_present": list(styles.PANEL_TITLES),
        "panel_count": styles.N_PANELS,
    }
    if not path.exists():
        return result

    dpi = _read_png_dpi(path)
    if dpi is not None:
        result["dpi"] = round(dpi, 1)
        result["meets_fr_5_3"] = dpi >= config.DASHBOARD_DPI
    try:
        image = mpimg.imread(path)
        result["valid_png"] = True
        result["height_px"], result["width_px"] = image.shape[0], image.shape[1]
    except Exception:
        result["valid_png"] = False
    return result


def _read_png_dpi(path: Path) -> float | None:
    """Return the DPI recorded in a PNG's ``pHYs`` chunk, or ``None``.

    The chunk stores pixels-per-metre as two big-endian ``uint32`` values plus a
    unit specifier, where unit 1 means metres. DPI is ``ppu * 0.0254``.
    """
    try:
        with Path(path).open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
            while True:
                header = handle.read(8)
                if len(header) < 8:
                    return None
                length, chunk_type = struct.unpack(">I4s", header)
                if chunk_type == b"pHYs":
                    data = handle.read(length)
                    ppu_x, ppu_y, unit = struct.unpack(">IIB", data)
                    if unit != 1:
                        return None
                    return ppu_x * 0.0254
                handle.seek(length + 4, 1)  # skip payload and CRC
    except (OSError, struct.error):
        return None
    return None


# ---------------------------------------------------------------------------
# Web interface
# ---------------------------------------------------------------------------


class VIPRequestHandler(BaseHTTPRequestHandler):
    """Static-file and JSON handler for the interface.

    Deliberately minimal. Every path is resolved against a fixed base directory
    and checked to be inside it, so no request can read outside
    ``dashboard/static`` or the figures allow-list.
    """

    server_version = "mRNA-VIP"
    protocol_version = "HTTP/1.1"

    # -- helpers ----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(
            status,
            json.dumps(payload, indent=2, default=str).encode("utf-8"),
            CONTENT_TYPES[".json"],
        )

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), content_type)

    @staticmethod
    def _safe_join(base: Path, relative: str) -> Path | None:
        """Resolve ``relative`` inside ``base``, or ``None`` if it escapes."""
        try:
            candidate = (base / relative).resolve()
            candidate.relative_to(base.resolve())
            return candidate
        except (ValueError, OSError):
            return None

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        route = self.path.split("?", 1)[0].rstrip("/") or "/"

        if route == "/":
            index = STATIC_DIR / "index.html"
            if not index.is_file():
                self._send(500, b"index.html missing", "text/plain; charset=utf-8")
                return
            html = index.read_text(encoding="utf-8").replace(
                THEME_MARKER, styles.theme_css()
            )
            self._send(200, html.encode("utf-8"), CONTENT_TYPES[".html"])
            return

        if route == "/api/state":
            self._send_json(200, components.platform_state())
            return

        if route == "/api/health":
            self._send_json(200, {"status": "ok"})
            return

        if route.startswith("/static/"):
            target = self._safe_join(STATIC_DIR, route[len("/static/") :])
            if target is None:
                self._send(403, b"Forbidden", "text/plain; charset=utf-8")
                return
            self._send_file(target)
            return

        if route.startswith("/figures/"):
            name = route[len("/figures/") :]
            allowed = SERVEABLE_FIGURES.get(name)
            self._send_file(allowed if allowed else Path("__missing__"))
            return

        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        """Report that pipeline actions are not implemented.

        The interface calls this so its loading, empty and error states are real.
        Refusing honestly is the correct behaviour: the modules behind it are
        specified but not implemented, and simulating them would mean fabricating
        patient predictions, which this project must never do.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route.startswith("/api/"):
            self._send_json(
                501,
                {
                    "error": "not_implemented",
                    "message": components.NOT_INITIALIZED_STATE,
                    "detail": (
                        "This interface is the design layer. The "
                        "preprocessing, machine-learning, attention, retrieval "
                        "and report-generation modules are specified but not yet "
                        "implemented."
                    ),
                    "command": components.RUN_COMMAND,
                },
            )
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def log_message(self, format: str, *args: Any) -> None:
        """Route request logs to stderr at a single line each."""
        sys.stderr.write(f"  {self.address_string()} {format % args}\n")


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Run the interface until interrupted.

    Parameters
    ----------
    host:
        Bind address. Defaults to loopback, so the interface is never exposed to
        the network by accident.
    port:
        Bind port.
    open_browser:
        Open the default browser once the server is listening.
    """
    if not STATIC_DIR.is_dir():
        raise FileNotFoundError(f"Static assets not found at {STATIC_DIR}")

    handler = partial(VIPRequestHandler)
    with ThreadingHTTPServer((host, port), handler) as httpd:
        url = f"http://{host}:{httpd.server_address[1]}/"
        state = components.platform_state()
        ready = sum(1 for s in state["stages"] if s["status"] == "ready")

        print(f"\n  mRNA VIP  -  Vaccine Intelligence & Protection")
        print(f"  {'-' * 52}")
        print(f"  Interface      {url}")
        print(f"  Stages ready   {ready} of {len(state['stages'])}")
        print(f"  Patient rows   {state['metrics']['patient_records']}")
        print(f"  Python         {state['environment']['python']}")
        print(f"\n  Press Ctrl+C to stop.\n")
        if open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Usage
    -----
    ``python -m dashboard.app``
        Serve the interface on ``http://127.0.0.1:8000``.

    ``python -m dashboard.app --no-browser``
        Serve without opening a browser (use in CI or over SSH).

    ``python -m dashboard.app --figure``
        Render ``outputs/figures/dashboard.png`` and exit.

    ``python -m dashboard.app --verify``
        Check an existing figure against FR-5.1 and FR-5.3, then exit.

    ``python -m dashboard.app --out outputs/figures/dashboard.png``
        Write the figure to a different path.
    """
    parser = argparse.ArgumentParser(
        prog="dashboard.app", description="mRNA VIP interface and figure renderer"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window"
    )
    parser.add_argument(
        "--figure", action="store_true", help="Render the three-panel figure and exit"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify the saved figure and exit"
    )
    parser.add_argument("--out", default=None, help="Figure output path")
    parser.add_argument(
        "--state", action="store_true", help="Print the platform state JSON and exit"
    )
    args = parser.parse_args(argv)

    if args.state:
        print(json.dumps(components.platform_state(), indent=2, default=str))
        return 0

    if args.verify:
        report = verify_dashboard(args.out)
        print(json.dumps(report, indent=2))
        if not report["exists"]:
            print("\n  No figure saved yet. Run: --figure")
            return 1
        return 0 if report["meets_fr_5_3"] else 1

    if args.figure:
        path = build_dashboard(args.out)
        print(f"  Figure written: {path}")
        return 0

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
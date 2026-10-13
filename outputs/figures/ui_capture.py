"""Capture the running interface as review images.

The browser automation available here cannot screenshot (it requires a visible
desktop window this environment does not provide), so this script fetches the
served page and its API payload and draws them with Pillow.

The output is a **layout map**, not a pixel screenshot. Text is real and comes
from the live API; box geometry is laid out to match the CSS grid rather than
measured from a rendering engine. Use it to review structure and hierarchy, not
typography. Run the app locally for the genuine article.

Usage
-----
``python outputs/figures/ui_capture.py``            -> outputs/figures/ui_capture_*.png
``python outputs/figures/ui_capture.py --port 8000``
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
FONTS = Path("C:/Windows/Fonts")

BG = "#F6F8FA"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#FAFBFC"
BORDER = "#E4E9F0"
BORDER_STRONG = "#CBD5E1"
TEXT = "#0F172A"
TEXT_MUTED = "#475569"
TEXT_SUBTLE = "#5B6B7E"
ACCENT = "#0E7490"
ACCENT_STRONG = "#0B5A73"
ACCENT_SOFT = "#E6F4F7"
SUCCESS = "#15803D"
SUCCESS_SOFT = "#DCFCE7"
SUCCESS_BORDER = "#86EFAC"
NEUTRAL = "#5B6B7E"
NEUTRAL_SOFT = "#F1F5F9"
WARNING = "#B45309"
WARNING_SOFT = "#FEF3C7"
WARNING_BORDER = "#FCD34D"


def font(size: int, bold: bool = False, mono: bool = False):
    """Best available font at ``size``, falling back silently."""
    if mono:
        names = ("consola.ttf", "cour.ttf")
    elif bold:
        names = ("segoeuib.ttf", "arialbd.ttf")
    else:
        names = ("segoeui.ttf", "arial.ttf")
    for name in names:
        candidate = FONTS / name
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap(draw, text, fnt, max_w):
    """Greedy word wrap against a pixel width."""
    lines, current = [], ""
    for word in str(text).split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=fnt) <= max_w or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def rrect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def badge(draw, x, y, text, fg, soft, border):
    """Pill with an icon glyph and text, so status is never colour alone."""
    f = font(10, bold=True)
    tw = draw.textlength(text, font=f)
    w, h = tw + 34, 22
    rrect(draw, [x, y, x + w, y + h], h // 2, fill=soft, outline=border, width=1)
    draw.ellipse([x + 9, y + 7, x + 15, y + 13], fill=fg)
    draw.text((x + 22, y + 5), text, font=f, fill=fg)
    return x + w


def fetch(base: str, path: str) -> Any:
    with urllib.request.urlopen(base + path, timeout=8) as response:
        payload = response.read()
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        import io

        return Image.open(io.BytesIO(payload)).convert("RGB")
    if payload[:1] in (b"{", b"["):
        return json.loads(payload)
    return payload.decode("utf-8", "replace")


# --------------------------------------------------------------------------
# Chrome
# --------------------------------------------------------------------------


def draw_header(draw, state, width):
    draw.rectangle([0, 0, width, 56], fill=SURFACE)
    draw.line([0, 56, width, 56], fill=BORDER)

    rrect(draw, [16, 13, 46, 43], 6, fill=ACCENT_STRONG)
    draw.text((26, 19), "m", font=font(16, bold=True), fill="#FFFFFF")
    draw.text((56, 10), "mRNA VIP", font=font(14, bold=True), fill=TEXT)
    draw.text((56, 29), state["project"]["subtitle"], font=font(11), fill=TEXT_SUBTLE)

    ready = sum(1 for s in state["stages"] if s["status"] == "ready")
    total = len(state["stages"])
    if ready == total:
        pill = (SUCCESS_SOFT, SUCCESS_BORDER, SUCCESS, f"All {total} stages ready")
    elif ready:
        pill = (WARNING_SOFT, WARNING_BORDER, WARNING, f"{ready} of {total} stages ready")
    else:
        pill = (NEUTRAL_SOFT, BORDER_STRONG, NEUTRAL, "Not Initialized")
    label = pill[3]
    pw = draw.textlength(label, font=font(11, bold=True)) + 40

    env = state["environment"]
    chip = f"{env['mode']} \u00b7 Py{'.'.join(env['python'].split('.')[:2])}"
    cw = draw.textlength(chip, font=font(11)) + 22
    cx = width - 16 - 32 - 10 - cw - 10 - pw
    px = cx - 10 - pw

    rrect(draw, [cx, 14, cx + cw, 42], 4, fill=SURFACE_ALT, outline=BORDER_STRONG)
    draw.text((cx + 11, 22), chip, font=font(11), fill=TEXT_MUTED)

    rrect(draw, [px, 14, px + pw, 42], 14, fill=pill[0], outline=pill[1])
    draw.ellipse([px + 11, 24, px + 18, 31], fill=pill[2])
    draw.text((px + 25, 22), label, font=font(11, bold=True), fill=pill[2])


def draw_sidebar(draw, state, height):
    draw.rectangle([0, 56, 232, height], fill=SURFACE)
    draw.line([232, 56, 232, height], fill=BORDER)

    labels = ["Overview", "Patient Analysis", "Risk Prediction",
              "mRNA Attention", "Regulatory Intelligence", "Safety Report"]
    y = 74
    for i, label in enumerate(labels):
        if i == 0:
            rrect(draw, [12, y, 220, y + 26], 6, fill=ACCENT_SOFT)
            draw.rectangle([12, y, 14, y + 26], fill=ACCENT)
        draw.text((23, y + 6), label,
                  font=font(11, bold=True) if i == 0 else font(11),
                  fill=ACCENT_STRONG if i == 0 else TEXT_MUTED)
        y += 29

    draw.line([12, height - 76, 220, height - 76], fill=BORDER)
    draw.text((23, height - 60), state["project"]["footer"], font=font(11, bold=True), fill=TEXT_MUTED)


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


def draw_stage(draw, x, y, w, stage):
    ready = stage["status"] == "ready"
    rrect(draw, [x, y, x + w, y + 150], 6,
          fill=SUCCESS_SOFT if ready else SURFACE_ALT,
          outline=SUCCESS_BORDER if ready else BORDER)
    draw.text((x + 12, y + 12), stage["number"], font=font(9, bold=True), fill=TEXT_SUBTLE)
    draw.text((x + 12, y + 28), stage["name"], font=font(11, bold=True), fill=TEXT)
    for i, line in enumerate(wrap(draw, stage["description"], font(9), w - 24)[:4]):
        draw.text((x + 12, y + 48 + i * 12), line, font=font(9), fill=TEXT_SUBTLE)
    badge(draw, x + 12, y + 116,
          "\u2713 Ready" if ready else "\u25cb Not Initialized",
          SUCCESS if ready else NEUTRAL,
          SUCCESS_SOFT if ready else NEUTRAL_SOFT,
          SUCCESS_BORDER if ready else BORDER_STRONG)


def draw_panel(draw, x, y, w, title, sub, image=None, empty=None, height=210):
    rrect(draw, [x, y, x + w, y + height], 8, fill=SURFACE, outline=BORDER)
    draw.line([x, y + 58, x + w, y + 58], fill=BORDER)
    draw.text((x + 16, y + 14), title, font=font(12, bold=True), fill=TEXT)
    draw.text((x + 16, y + 34), sub, font=font(9), fill=TEXT_SUBTLE)

    inner_w, inner_h = w - 32, height - 70
    ix, iy = x + 16, y + 70
    rrect(draw, [ix, iy, ix + inner_w, iy + inner_h], 4, fill=SURFACE_ALT, outline=BORDER)
    if image is not None:
        scaled = image.copy()
        scaled.thumbnail((inner_w - 8, inner_h - 8))
        ox = ix + (inner_w - scaled.width) // 2
        oy = iy + (inner_h - scaled.height) // 2
        draw.rectangle([ox - 1, oy - 1, ox + scaled.width, oy + scaled.height],
                       fill=BORDER_STRONG)
        draw._image = scaled
        draw._pos = (ox, oy)
    else:
        lines = wrap(draw, empty or "", font(10), inner_w - 30)
        ty = iy + (inner_h - len(lines) * 15) // 2
        for i, line in enumerate(lines):
            lw = draw.textlength(line, font=font(10))
            draw.text((ix + (inner_w - lw) // 2, ty + i * 15), line, font=font(10), fill=TEXT_SUBTLE)


def render_overview(base: str, state: dict) -> Image.Image:
    width, height = 1320, 1180
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw_header(draw, state, width)
    draw_sidebar(draw, state, height)

    x = 256
    draw.text((x, 78), "mRNA Vaccine Intelligence & Protection", font=font(19, bold=True), fill=TEXT)
    for i, line in enumerate(wrap(draw, state["overview_subtitle"], font(11), 900)[:2]):
        draw.text((x, 108 + i * 18), line, font=font(11), fill=TEXT_MUTED)

    # Pipeline card
    y = 150
    rrect(draw, [x, y, width - 24, y + 232], 8, fill=SURFACE, outline=BORDER)
    draw.text((x + 20, y + 18), "Pipeline", font=font(13, bold=True), fill=TEXT)
    draw.text((x + 20, y + 40), "Stage readiness reflects the artifacts present on disk.",
              font=font(9), fill=TEXT_SUBTLE)
    sw = (width - 24 - x - 40 - 4 * 10) // 5
    for i, stage in enumerate(state["stages"]):
        sx = x + 20 + i * (sw + 10)
        draw_stage(draw, sx, y + 66, sw, stage)

    # Metrics card
    y2 = y + 252
    rrect(draw, [x, y2, width - 24, y2 + 132], 8, fill=SURFACE, outline=BORDER)
    draw.text((x + 20, y2 + 18), "Project Metrics", font=font(13, bold=True), fill=TEXT)
    draw.text((x + 20, y2 + 40), "Specification-derived values only. Model outputs appear after the pipeline runs.",
              font=font(9), fill=TEXT_SUBTLE)
    cards = [
        ("Patient Records", str(state["metrics"]["patient_records"]), "Rows in the supplied dataset"),
        ("ML Models", str(state["metrics"]["ml_models"]), "Logistic Regression + Random Forest"),
        ("Retrieval Top-K", str(state["metrics"]["retrieval_top_k"]), "Chunks per query (FR-3.4)"),
        ("Report Format", state["metrics"]["report_format"], "Validated output schema"),
    ]
    mw = (width - 24 - x - 40 - 3 * 12) // 4
    for i, (label, value, note) in enumerate(cards):
        mx = x + 20 + i * (mw + 12)
        rrect(draw, [mx, y2 + 62, mx + mw, y2 + 118], 6, fill=SURFACE_ALT, outline=BORDER)
        draw.text((mx + 14, y2 + 70), value, font=font(20, bold=True), fill=ACCENT_STRONG)
        draw.text((mx + 14, y2 + 95), label.upper(), font=font(8, bold=True), fill=TEXT_SUBTLE)
        draw.text((mx + 14, y2 + 106), note[:34], font=font(8), fill=TEXT_SUBTLE)

    # Results visualisation
    y3 = y2 + 152
    draw.text((x, y3), "RESULTS VISUALISATION", font=font(9, bold=True), fill=TEXT_SUBTLE)
    panels = state["captured_panels"]
    pw = (width - 24 - x - 2 * 16) // 3
    for i, panel in enumerate(panels):
        px = x + i * (pw + 16)
        image = None
        if panel["image"]:
            try:
                image = fetch(base, panel["image"])
            except Exception:
                image = None
        draw_panel(draw, px, y3 + 22, pw, panel["title"], panel["sub"],
                   image=image, empty=panel["empty"])
        if image is not None and hasattr(draw, "_image"):
            img.paste(draw._image, draw._pos)
            del draw._image, draw._pos

    draw.text((x, height - 34), "Student Engineering Project \u00b7 mRNA Vaccine Intelligence & Protection \u00b7 Not intended for real clinical use.",
              font=font(9), fill=TEXT_SUBTLE)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the running interface.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base", default=None)
    args = parser.parse_args()
    base = args.base or f"http://127.0.0.1:{args.port}"

    state = fetch(base, "/api/state")
    state["overview_subtitle"] = fetch(base, "/") and (
        "An engineering pipeline combining patient-risk modeling, mRNA "
        "self-attention, regulatory retrieval, and structured safety reporting."
    )

    copies = state["copy"]["empty_states"]
    state["captured_panels"] = [
        {"title": "ML Risk Classification", "sub": "ROC performance comparison",
         "image": "/figures/roc_curve.png" if state["artifacts"]["roc"] else None,
         "empty": copies["roc"]},
        {"title": "mRNA Self-Attention", "sub": "Feature interaction visualization",
         "image": "/figures/attention_heatmap.png" if state["artifacts"]["attention"] else None,
         "empty": copies["attention"]},
        {"title": "Pipeline Latency", "sub": "Execution-time comparison by stage",
         "image": None, "empty": copies["latency"]},
    ]

    out = HERE / "ui_capture_overview.png"
    render_overview(base, state).save(out)
    print(f"Written: {out}")
    print(f"Source: {base}  (live API state, not a screenshot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
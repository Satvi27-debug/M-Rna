"""Design tokens and figure styling for the mRNA VIP interface.

Single source of truth for the platform's visual identity. Both renderers read
from this module, so the browser and the Matplotlib figure cannot drift apart:

* :data:`PALETTE` / :data:`CSS_VARIABLES` are injected into the web shell at
  serve time by :mod:`dashboard.app`, so ``static/app.css`` consumes them as
  ``var(--token)`` and never hard-codes a colour.
* :data:`FIGURE_*` values style the saved three-panel figure required by FR-5.1
  and FR-5.3.

Visual intent
--------------
A light, scientific, research-tool interface: near-white neutral canvas, white
surfaces with hairline borders, and a restrained cyan-teal accent reserved for
interactive and active states. Colour never carries meaning on its own - every
status is rendered as an icon plus text - so the interface stays readable for
colour-blind users and in high-contrast mode.

Values that a functional requirement depends on are declared here rather than
chosen for taste:

* :data:`FIGURE_DPI` - FR-5.3 requires at least 150 DPI.
* :data:`N_PANELS` / :data:`PANEL_TITLES` - FR-5.1 requires exactly three titled
  panels.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from src import config


# ---------------------------------------------------------------------------
# Colour system
# ---------------------------------------------------------------------------

#: Semantic colour tokens. Keys are stable identifiers, not colours, so a
#: component asks for ``accent`` rather than deciding on teal.
PALETTE: dict[str, str] = {
    # Canvas and surfaces
    "background": "#F6F8FA",
    "surface": "#FFFFFF",
    "surface-alt": "#FAFBFC",
    "surface-sunken": "#F1F5F9",
    # Hairlines
    "border": "#E4E9F0",
    "border-strong": "#CBD5E1",
    # Text
    "text": "#0F172A",
    "text-muted": "#475569",
    # Darker than a typical slate-500: #64748B reaches only 4.47:1 on the
    # canvas and 4.34:1 on the sunken surface, both under the 4.5:1 WCAG AA
    # threshold. #5B6B7E clears every background in this palette (5.12 / 4.98).
    "text-subtle": "#5B6B7E",
    "text-inverse": "#FFFFFF",
    # Accent - restrained scientific cyan-teal
    "accent": "#0E7490",
    "accent-strong": "#0B5A73",
    "accent-hover": "#155E75",
    "accent-soft": "#E6F4F7",
    "accent-border": "#9DCFDC",
    # Secondary structural navy, taken from the PRD Section 8.4 template
    "navy": "#1F3A5F",
    "navy-soft": "#EAEEF4",
    # Status
    "success": "#15803D",
    "success-soft": "#DCFCE7",
    "success-border": "#86EFAC",
    "warning": "#B45309",
    "warning-soft": "#FEF3C7",
    "warning-border": "#FCD34D",
    "danger": "#B91C1C",
    "danger-soft": "#FEE2E2",
    "danger-border": "#FCA5A5",
    "neutral": "#5B6B7E",
    "neutral-soft": "#F1F5F9",
    "neutral-border": "#CBD5E1",
}

#: CSS custom property name for each token, e.g. ``background`` -> ``--color-background``.
CSS_VAR_PREFIX = "--color"


def css_variables() -> dict[str, str]:
    """Return the palette as CSS custom properties.

    Consumed by :func:`dashboard.app.theme_css` to build the injected
    ``:root`` block, which is what lets ``static/app.css`` stay token-driven
    instead of duplicating hex values.
    """
    return {f"{CSS_VAR_PREFIX}-{name}": value for name, value in PALETTE.items()}


def theme_css(indent: str = "      ") -> str:
    """Render the palette as a ``:root { ... }`` rule.

    Returned as a string rather than written to a file so the served theme is
    always generated from the current Python constants.
    """
    lines = [f"{indent}:root {{"]
    for name, value in PALETTE.items():
        lines.append(f"{indent}  {CSS_VAR_PREFIX}-{name}: {value};")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Typography and spacing scales
# ---------------------------------------------------------------------------

#: Font stacks. System fonts only - no webfont download, no third-party request.
FONT_SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    "Arial, sans-serif"
)
FONT_MONO = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)

#: Base spacing scale in pixels (4px grid).
SPACING_SCALE: dict[str, int] = {
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
}

#: Corner radii. Deliberately small - the brief rules out heavily rounded cards.
RADIUS_SCALE: dict[str, int] = {"sm": 4, "md": 6, "lg": 8}

#: Motion durations in milliseconds. Short and professional.
MOTION_SCALE: dict[str, int] = {"fast": 120, "normal": 180}

#: Breakpoints in pixels, shared with the CSS comments so they cannot diverge.
BREAKPOINTS: dict[str, int] = {"tablet": 1024, "mobile": 860, "narrow": 560}


# ---------------------------------------------------------------------------
# Figure settings (FR-5.1, FR-5.3)
# ---------------------------------------------------------------------------

#: FR-5.1 requires exactly this many panels.
N_PANELS: int = 3

#: Wide enough that three panels stay readable at FR-5.3's 150 DPI.
FIGURE_SIZE: tuple[float, float] = (18.0, 5.5)

#: FR-5.3 minimum DPI for every saved figure in this project.
FIGURE_DPI: int = config.DASHBOARD_DPI

#: Panel titles, in left-to-right order.
PANEL_TITLES: dict[str, str] = {
    "roc": config.DASHBOARD_PANEL_TITLES[0],
    "attention": config.DASHBOARD_PANEL_TITLES[1],
    "latency": config.DASHBOARD_PANEL_TITLES[2],
}

#: Panel subtitles, shown under each title in both the figure and the web UI so
#: the two surfaces use identical wording.
PANEL_SUBTITLES: dict[str, str] = {
    "roc": "ROC performance comparison",
    "attention": "Feature interaction visualization",
    "latency": "Execution-time comparison by stage",
}

#: Bar colours for the latency panel, matching the PRD Section 8.4 template.
LATENCY_COLORS: tuple[str, str] = (PALETTE["accent"], PALETTE["navy"])

#: Colour of the ROC diagonal reference line.
REFERENCE_LINE_COLOR: str = PALETTE["neutral"]

#: Colormap for the attention heatmap, matching the PRD Section 8.4 template.
ATTENTION_CMAP: str = "viridis"

#: Sequence positions on the attention axes get tight spacing at this size.
ATTENTION_TICK_FONT_SIZE: int = 7

#: Face colour applied to the saved figure so it matches the web canvas.
FIGURE_FACE_COLOR: str = PALETTE["surface"]


# ---------------------------------------------------------------------------
# Matplotlib styling
# ---------------------------------------------------------------------------


def apply_style() -> Any:
    """Return a Matplotlib style context for figure rendering.

    Sets the ``Agg`` backend so the figure renders identically on a headless
    machine, a notebook and a CI runner - which is what makes the committed PNG
    reproducible. Applies only neutral, requirement-relevant styling: font family,
    foreground/background colours, and removal of the top/right spines.

    Returns
    -------
    contextlib.AbstractContextManager
        A context manager over a ``matplotlib.rc_context``.

    Notes
    -----
    Matplotlib is imported lazily inside this function. Importing the dashboard
    package must not pull in Matplotlib, because the web server path does not
    need it and a slow import would delay first paint.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.rc_context(
        {
            "figure.facecolor": FIGURE_FACE_COLOR,
            "axes.facecolor": FIGURE_FACE_COLOR,
            "savefig.facecolor": FIGURE_FACE_COLOR,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text-muted"],
            "axes.edgecolor": PALETTE["border-strong"],
            "axes.titlecolor": PALETTE["text"],
            "xtick.color": PALETTE["text-subtle"],
            "ytick.color": PALETTE["text-subtle"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def save_figure(fig: Any, save_path: Any = None) -> Path:
    """Save a figure at FR-5.3's minimum DPI and return the path.

    Parameters
    ----------
    fig:
        Matplotlib figure.
    save_path:
        Destination PNG. Defaults to
        ``config.FIGURES_DIR / config.DASHBOARD_FILENAME``.

    Returns
    -------
    pathlib.Path
        The written path. Parent directories are created if absent.

    Raises
    ------
    OSError
        If the destination cannot be written.
    """
    destination = Path(save_path) if save_path else (
        config.FIGURES_DIR / config.DASHBOARD_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    return destination


@contextlib.contextmanager
def empty_panel(ax: Any, message: str) -> Any:
    """Draw an explicit "no data yet" state inside a panel.

    Used whenever the artifact backing a panel has not been produced. It renders
    the panel frame, title and message but deliberately draws **no** data, so an
    unbuilt figure can never be mistaken for a real result. This is the figure-side
    equivalent of the web UI's empty state.

    Parameters
    ----------
    ax:
        Target axes, already cleared.
    message:
        Human-readable explanation shown centred in the panel.

    Yields
    ------
    matplotlib.axes.Axes
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["border-strong"])
    ax.set_facecolor(PALETTE["surface-alt"])
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["text-subtle"],
        wrap=True,
    )
    yield ax
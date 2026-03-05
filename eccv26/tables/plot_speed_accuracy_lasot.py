#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")

# Register Book Antiqua and apply it to axis tick labels.
import matplotlib.font_manager as _fm

_BOOK_ANTIQUA_PATH = "/Users/tsing/Library/Fonts/Book Antiqua.ttf"
_TICK_FONT_FAMILY = "Book Antiqua"
try:
    _fm.fontManager.addfont(_BOOK_ANTIQUA_PATH)
except Exception:
    _TICK_FONT_FAMILY = "serif"  # graceful fallback

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# =========================
# Adjustable parameters
# =========================
SCALE = 1.0  # increase for larger elements

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "forVS.csv"

# Which columns to plot
DEFAULT_FPS_COL = "OrinGPU"
DEFAULT_METRIC_COL = "LaSOT"
DEFAULT_GMACS_COL = "GFLOPs"
SUPPORTED_Y_METRICS = ["LaSOT", "TkNet"]
DEFAULT_METHOD_COL = "Method"
DEFAULT_PUB_COL = "Seires"
DEFAULT_SERIES_COL = "Seires"

# Plot look
FIG_SIZE_X = 6.2 * SCALE
FIG_SIZE_Y = 4.2 * SCALE
FIGSIZE = (FIG_SIZE_X, FIG_SIZE_Y)
USE_LOGX = False
OUTPUT_DPI = 300

AXIS_LABEL_FONTSIZE = int(round(12 * SCALE))
TITLE_FONTSIZE = int(round(14 * SCALE))
TICK_FONTSIZE = int(round(16 * SCALE))

# Title visibility
SHOW_MAIN_TITLE = False
SHOW_AXIS_TITLES = False

# Add some margins so point labels stay within axes spines
AXIS_MARGIN_X = 0.07
AXIS_MARGIN_Y = 0.07

# Axis config: minimum / maximum / interval / starting at (None = auto)
X_MIN = None
X_MAX = None
X_INTERVAL = None
X_START_AT = None

Y_MIN = None
Y_MAX = None
Y_INTERVAL = None
Y_START_AT = None

# Presets (top-level)
# mode: "auto" | "none" | "lasot_orincpu" | "tknet_oringpu"
PRESET_MODE = "auto"

PRESET_LASOT_ORINCPU = {
    "figsize": (6.0 * SCALE, 4.0 * SCALE), 
    "x_min": 10,
    "x_max": 110,
    "x_start": 0,
    "x_interval": 20,
    "y_min": 53,
    "y_max": 67,
    "y_start": 50,
    "y_interval": 5,
    "legend_anchor": "lower right",
    "legend_margin_x_ax": 0.05,
    "legend_margin_y_ax": 0.14,
    "legend_clamp_inside_axes": True,
}

PRESET_TKNET_ORINGPU = {
    "figsize": (6.0 * SCALE, 4.0 * SCALE),  
    "x_min": 40,
    "x_max": 250,
    "x_start": 50,
    "x_interval": 50,
    "y_min": 71,
    "y_max": 84,
    "y_start": 70,
    "y_interval": 2,
    "legend_anchor": "upper right",
    "legend_margin_x_ax": 0.05,
    "legend_margin_y_ax": 0.155,
    "legend_clamp_inside_axes": False,
}

# Render jobs: script will generate these figures directly (no CLI needed)
RENDER_PRESETS = [
    {"metric_col": "LaSOT", "fps_col": "M4_fps", "preset": "lasot_orincpu"},
    {"metric_col": "TkNet", "fps_col": "OrinGPU", "preset": "tknet_oringpu"},
]

# Point styles
# NOTE: Matplotlib scatter size `s` is area in pt^2.
POINT_ALPHA = 0.90
EDGE_COLOR_DEFAULT = "white"
EDGE_WIDTH_DEFAULT = 0.6 * SCALE
CENTER_DOT_ENABLED = True
CENTER_DOT_SIZE = 11.0 * SCALE  # scatter area in pt^2
CENTER_DOT_COLOR = "black"

# Highlight ours with a vivid fixed color
HIGHLIGHT_OURS = True
OURS_COLOR = "#D62728"  # vivid red (publication-friendly highlight)
OURS_EDGE_COLOR = "red"
OURS_EDGE_WIDTH = 0.6 * SCALE

# Size mapping: choose the minimum bubble radius/area first,
# then scale others by (GFLOPs / min_GFLOPs).
# mode: "radius" or "area"
SIZE_BASE_MODE = "radius"
MIN_BUBBLE_RADIUS_PT = 12.0 * SCALE
MIN_BUBBLE_AREA_PT2 = MIN_BUBBLE_RADIUS_PT**2
# Optional cap for very large bubbles (None disables cap)
MAX_BUBBLE_RADIUS_PT = 60.0 * SCALE
GMACS_FALLBACK = 1.0  # used when MACs(G) missing

# Colors: publication-friendly palette (Tableau 10 / colorblind-friendly)
COLOR_PALETTE = [
    "#95BE44",
    "#386632A4",
    "#9C755F",
    "#B07AA1",
    "#283736",
    "#EDC948",
    "#F28E2B",
    "#4E79A7",
    "#0D8183E9",
    "#BAB0AC",
]

# Labels
LABEL_ALL_POINTS = False
LABEL_OURS_POINTS = False
LABEL_TOPK_OTHERS = 0
LABEL_FONT_SIZE = int(round(12 * SCALE))
LABEL_OFFSET_PX = (int(round(5 * SCALE)), int(round(4 * SCALE)))
LABEL_FONT_FAMILY = _TICK_FONT_FAMILY          # font family for all labels
LABEL_COLOR = "#333333"                        # default label text color
LABEL_FONT_WEIGHT: Union[str, int] = "normal"  # font weight for regular labels
LABEL_STROKE_WIDTH = 2.5                        # white border width for all labels
LABEL_STROKE_COLOR = "white"                    # border color
# Ours-specific overrides
LABEL_OURS_COLOR = "#111111"
LABEL_OURS_FONT_WEIGHT: Union[str, int] = 900  # heavier than bold (700)
LABEL_OURS_STROKE_WIDTH = 3.5                  # thicker border for ours
LABEL_OURS_FONT_SIZE_DELTA = 1                 # ours font size relative to LABEL_FONT_SIZE
# Use adjustText to automatically push labels apart so they don't overlap.
# Set False to fall back to the simple offset heuristic.
LABEL_ADJUST_TEXT = True
# Draw a thin leader line from each moved label back to its data point.
LABEL_ADJUST_ARROW = True
LABEL_ADJUST_ARROW_COLOR = "#aaaaaa"
LABEL_ADJUST_ARROW_LW = 0.7
LABEL_ADJUST_EXPAND = (1.15, 1.25)             # text bounding-box expansion factor
LABEL_ADJUST_FORCE_TEXT = (0.15, 0.25)         # repulsion between labels
LABEL_ADJUST_FORCE_STATIC = (0.15, 0.25)       # repulsion from data points

# Connect same-series methods with lines
CONNECT_SERIES = True
SERIES_LINE_ALPHA = 0.8
SERIES_LINE_WIDTH = 1.4 * SCALE
SERIES_LINE_STYLE = "--"  # default dashed
SERIES_LINE_DARKEN = 0.75  # <1 means darker than bubble color

# Legend: show GMACs-to-radius demo circles at bottom-right
SHOW_SIZE_LEGEND = False
SIZE_LEGEND_GMACS = [0.5, 2.0]
SIZE_DEMO_COLOR = "#A9A9A9"
SIZE_DEMO_EDGE = "#7A7A7A"
SIZE_DEMO_ALPHA = 0.65
SIZE_DEMO_LINEWIDTH = 0.8 * SCALE

# Detailed layout controls for in-plot legend box
# Anchor options: "upper right" | "lower right" | "upper left" | "lower left"
SIZE_DEMO_ANCHOR = "lower right"
# Distance from plot border (axes coordinates)
SIZE_DEMO_MARGIN_X_AX = 0.05
SIZE_DEMO_MARGIN_Y_AX = 0.025
SIZE_DEMO_GAP_AX = 0.125           # horizontal gap between bubble center and text anchor
SIZE_DEMO_ROW_GAP_AX = 0.1       # baseline vertical gap (axes coords)
SIZE_DEMO_ROW_GAP_AUTO = False      # auto enlarge gap to avoid overlap
SIZE_DEMO_ROW_EXTRA_PAD_PT = 2.0 * SCALE
SIZE_DEMO_TEXT_COLOR = "#111111"

# Legend-only bubble compacting (does not affect main plot bubbles)
SIZE_DEMO_BUBBLE_SCALE = 0.72
SIZE_DEMO_MAX_RADIUS_PT = 40.0 * SCALE

# Keep legend inside axes safely
SIZE_DEMO_CLAMP_INSIDE_AXES = False
SIZE_DEMO_CLAMP_MARGIN_AX = 0.015

# Demo region box style
SIZE_DEMO_BOX = True
SIZE_DEMO_BOX_FACE = "#FFFFFF"
SIZE_DEMO_BOX_EDGE = "#B0B0B0"
SIZE_DEMO_BOX_ALPHA = 0.95
SIZE_DEMO_BOX_LINEWIDTH = 0.9 * SCALE
SIZE_DEMO_BOX_PAD_PT = 10 * SCALE
SIZE_DEMO_BOX_TOP_PAD_PT = 0 * SCALE  # extra gap between top row text and top border
SIZE_DEMO_BOX_ROUNDING_AX = 0.0
SIZE_DEMO_BOX_MIN_WIDTH_AX = 0.20
SIZE_DEMO_BOX_MIN_HEIGHT_AX = 0.18

# Output
DEFAULT_OUT_TEMPLATE = "lasot_speed_accuracy_{fps_col}.pdf"


def _to_float(series: pd.Series) -> pd.Series:
    # Handles empty strings, '-', and other non-numeric tokens.
    return pd.to_numeric(series.replace({"-": None, "": None}), errors="coerce")


_SERIES_NORMALIZE_RE = re.compile(r"\{.*?\}")


def _infer_series_name(method: str) -> str:
    """Infer tracker series/family from method name.

    Examples:
      - HiT-Tiny/Small/Base -> HiT
      - AsymTrack-T/S/B -> AsymTrack
      - MaST-nano/tiny/small -> MaST
      - FARTrack_{pico}/... -> FARTrack
      - ORTrack / ORTrack-D -> ORTrack
    """
    method = str(method).strip()
    if not method:
        return ""
    method = _SERIES_NORMALIZE_RE.sub("", method)
    # Prefer splitting by '-' (most common), then '_' as fallback.
    head = method.split("-", 1)[0]
    head = head.split("_", 1)[0]
    return head


def _compute_sizes_from_gmacs(
    gmacs: pd.Series,
    *,
    base_mode: str,
    min_bubble_radius_pt: float,
    min_bubble_area_pt2: float,
    max_bubble_radius_pt,
    fallback_gmacs: float,
) -> pd.Series:
    g = pd.to_numeric(gmacs, errors="coerce").fillna(fallback_gmacs)
    g = g.where(g > 0, fallback_gmacs)
    g_pos = g[g > 0]
    g_min = float(g_pos.min()) if len(g_pos) > 0 else float(fallback_gmacs)
    g_min = max(g_min, 1e-12)

    ratio = g / g_min
    mode = str(base_mode).strip().lower()
    if mode == "area":
        base_area = max(float(min_bubble_area_pt2), 1e-12)
        area = base_area * ratio
        radius = area.pow(0.5)
    else:
        base_radius = max(float(min_bubble_radius_pt), 1e-12)
        radius = base_radius * ratio

    if max_bubble_radius_pt is not None:
        radius = radius.clip(upper=float(max_bubble_radius_pt))

    # scatter `s` is area in pt^2; radius in pt -> s = r^2
    return radius.pow(2)


def _build_series_color_map(
    series_values: List[str], palette_name: Union[str, List[str]]
) -> Dict[str, Union[str, Tuple[float, float, float, float]]]:
    values = sorted({str(s) for s in series_values if str(s)})
    color_map: Dict[str, Union[str, Tuple[float, float, float, float]]] = {}

    if isinstance(palette_name, list):
        palette = [c for c in palette_name if str(c).strip()]
        if not palette:
            palette = ["#4E79A7"]
        for i, s in enumerate(values):
            color_map[s] = palette[i % len(palette)]
        return color_map

    cmap = plt.get_cmap(palette_name)
    for i, s in enumerate(values):
        # Most qualitative maps have a fixed cycle; modulo keeps stable colors.
        if hasattr(cmap, "N"):
            color_map[s] = cmap(i % cmap.N)
        else:
            color_map[s] = cmap(i / max(1, len(values) - 1))
    return color_map


def _nice_gmacs_ticks(g_min: float, g_max: float, k: int) -> List[float]:
    if k <= 1:
        return [g_min]
    if g_max <= g_min:
        return [g_min] * k
    ticks = [g_min + (g_max - g_min) * i / (k - 1) for i in range(k)]
    # Round for readability
    return [float(f"{t:.2f}") for t in ticks]


def _darken_color(color, factor: float = 0.75):
    try:
        import matplotlib.colors as mcolors

        r, g, b, a = mcolors.to_rgba(color)
        f = max(0.0, min(1.0, float(factor)))
        return (r * f, g * f, b * f, a)
    except Exception:
        return color


def _draw_size_demo(
    ax: plt.Axes,
    gmacs_values: List[float],
    *,
    anchor: Optional[str] = None,
    margin_x_ax: Optional[float] = None,
    margin_y_ax: Optional[float] = None,
    clamp_inside_axes: Optional[bool] = None,
) -> None:
    if not gmacs_values:
        return

    # Sizes for scatter (area in pt^2)
    sizes = _compute_sizes_from_gmacs(
        pd.Series(gmacs_values),
        base_mode=SIZE_BASE_MODE,
        min_bubble_radius_pt=MIN_BUBBLE_RADIUS_PT,
        min_bubble_area_pt2=MIN_BUBBLE_AREA_PT2,
        max_bubble_radius_pt=MAX_BUBBLE_RADIUS_PT,
        fallback_gmacs=GMACS_FALLBACK,
    ).tolist()

    # Compact legend bubbles only (not main plot bubbles)
    if SIZE_DEMO_MAX_RADIUS_PT is not None:
        max_area = float(SIZE_DEMO_MAX_RADIUS_PT) ** 2
        sizes = [min(float(s), max_area) for s in sizes]
    scale2 = float(SIZE_DEMO_BUBBLE_SCALE) ** 2
    sizes = [float(s) * scale2 for s in sizes]

    n = len(gmacs_values)

    # Convert bubble sizes to radii in points/pixels for robust spacing.
    radii_pt = [float(s) ** 0.5 for s in sizes]
    fig = ax.figure
    fig.canvas.draw()
    ax_bbox = ax.get_window_extent()
    ax_w_px = max(float(ax_bbox.width), 1.0)
    ax_h_px = max(float(ax_bbox.height), 1.0)
    px_per_pt = float(fig.dpi) / 72.0
    max_r_px = (max(radii_pt) if radii_pt else 0.0) * px_per_pt
    max_r_ax_x = max_r_px / ax_w_px
    max_r_ax_y = max_r_px / ax_h_px

    # Row spacing: auto mode guarantees no overlap among circles.
    base_gap_ax = float(SIZE_DEMO_ROW_GAP_AX)
    if SIZE_DEMO_ROW_GAP_AUTO:
        min_gap_ax = (2.0 * max_r_px + float(SIZE_DEMO_ROW_EXTRA_PAD_PT) * px_per_pt) / ax_h_px
        row_gap_ax = max(base_gap_ax, min_gap_ax)
    else:
        row_gap_ax = base_gap_ax

    # Anchored layout by corner.
    anchor = str(anchor if anchor is not None else SIZE_DEMO_ANCHOR).strip().lower()
    is_right = "right" in anchor
    is_upper = "upper" in anchor

    mx = float(SIZE_DEMO_MARGIN_X_AX if margin_x_ax is None else margin_x_ax)
    my = float(SIZE_DEMO_MARGIN_Y_AX if margin_y_ax is None else margin_y_ax)
    if is_right:
        x_text = 1.0 - mx
        x_circle = x_text - float(SIZE_DEMO_GAP_AX)
        text_ha = "right"
    else:
        x_text = mx
        x_circle = x_text + float(SIZE_DEMO_GAP_AX)
        text_ha = "left"

    # Keep internal legend layout consistent for all anchors:
    # always place entries top->bottom in the same order.
    # Then move the whole legend block by anchor.
    block_h = (n - 1) * row_gap_ax
    if is_upper:
        y_top = 1.0 - my
    else:
        y_top = my + block_h
    ys = [y_top - i * row_gap_ax for i in range(n)]

    clamp_flag = SIZE_DEMO_CLAMP_INSIDE_AXES if clamp_inside_axes is None else bool(clamp_inside_axes)

    # Keep legend inside axes if requested.
    if clamp_flag:
        m = float(SIZE_DEMO_CLAMP_MARGIN_AX)
        x_text = min(max(x_text, m), 1.0 - m)
        if is_right:
            x_circle = x_text - float(SIZE_DEMO_GAP_AX)
            min_x_circle = m + max_r_ax_x
            if x_circle < min_x_circle:
                x_circle = min_x_circle
                x_text = min(x_circle + float(SIZE_DEMO_GAP_AX), 1.0 - m)
        else:
            x_circle = x_text + float(SIZE_DEMO_GAP_AX)
            max_x_circle = 1.0 - m - max_r_ax_x
            if x_circle > max_x_circle:
                x_circle = max_x_circle
                x_text = max(x_circle - float(SIZE_DEMO_GAP_AX), m)

        # Shift the whole legend block if needed (do not change internal row layout).
        y_min_allowed = m + max_r_ax_y
        y_max_allowed = 1.0 - m - max_r_ax_y
        y_lo = min(ys)
        y_hi = max(ys)
        dy = 0.0
        if y_lo < y_min_allowed:
            dy += (y_min_allowed - y_lo)
        if y_hi + dy > y_max_allowed:
            dy -= (y_hi + dy - y_max_allowed)
        if abs(dy) > 0:
            ys = [y + dy for y in ys]

    # Draw artists first, then measure exact extents (includes text width).
    artists = []
    for y, g, s in zip(ys, gmacs_values, sizes):
        sc = ax.scatter(
            [x_circle],
            [y],
            s=s,
            transform=ax.transAxes,
            color=SIZE_DEMO_COLOR,
            alpha=SIZE_DEMO_ALPHA,
            edgecolors=SIZE_DEMO_EDGE,
            linewidths=SIZE_DEMO_LINEWIDTH,
            zorder=10,
            clip_on=False,
        )
        artists.append(sc)
        if CENTER_DOT_ENABLED:
            cd = ax.scatter(
                [x_circle],
                [y],
                s=CENTER_DOT_SIZE,
                transform=ax.transAxes,
                color=CENTER_DOT_COLOR,
                edgecolors="none",
                zorder=12,
                clip_on=False,
            )
            artists.append(cd)
        tx = ax.text(
            x_text,
            y,
            f"{g:g}G",
            transform=ax.transAxes,
            ha=text_ha,
            va="center",
            fontsize=TICK_FONTSIZE,
            color=SIZE_DEMO_TEXT_COLOR,
            zorder=11,
            clip_on=False,
        )
        artists.append(tx)

    if SIZE_DEMO_BOX:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [a.get_window_extent(renderer=renderer) for a in artists]
        left_px = min(bb.x0 for bb in bboxes)
        right_px = max(bb.x1 for bb in bboxes)
        bottom_px = min(bb.y0 for bb in bboxes)
        top_px = max(bb.y1 for bb in bboxes)

        px_per_pt = float(fig.dpi) / 72.0
        pad_px = float(SIZE_DEMO_BOX_PAD_PT) * px_per_pt
        top_extra_px = float(SIZE_DEMO_BOX_TOP_PAD_PT) * px_per_pt
        left_px -= pad_px
        right_px += pad_px
        bottom_px -= pad_px
        top_px += pad_px + top_extra_px

        (x0_ax, y0_ax) = ax.transAxes.inverted().transform((left_px, bottom_px))
        (x1_ax, y1_ax) = ax.transAxes.inverted().transform((right_px, top_px))

        # Enforce minimum box size in axes coordinates (for stable rendering).
        w_ax = x1_ax - x0_ax
        h_ax = y1_ax - y0_ax
        if w_ax < SIZE_DEMO_BOX_MIN_WIDTH_AX:
            if is_right:
                x0_ax = x1_ax - SIZE_DEMO_BOX_MIN_WIDTH_AX
            else:
                x1_ax = x0_ax + SIZE_DEMO_BOX_MIN_WIDTH_AX
        if h_ax < SIZE_DEMO_BOX_MIN_HEIGHT_AX:
            if is_upper:
                y0_ax = y1_ax - SIZE_DEMO_BOX_MIN_HEIGHT_AX
            else:
                y1_ax = y0_ax + SIZE_DEMO_BOX_MIN_HEIGHT_AX

        # Clamp final box within axes bounds if requested.
        if clamp_flag:
            m = float(SIZE_DEMO_CLAMP_MARGIN_AX)
            if x0_ax < m:
                dx = m - x0_ax
                x0_ax += dx
                x1_ax += dx
            if x1_ax > 1.0 - m:
                dx = x1_ax - (1.0 - m)
                x0_ax -= dx
                x1_ax -= dx
            if y0_ax < m:
                dy = m - y0_ax
                y0_ax += dy
                y1_ax += dy
            if y1_ax > 1.0 - m:
                dy = y1_ax - (1.0 - m)
                y0_ax -= dy
                y1_ax -= dy

        box = mpatches.FancyBboxPatch(
            (x0_ax, y0_ax),
            (x1_ax - x0_ax),
            (y1_ax - y0_ax),
            boxstyle=f"round,pad=0.0,rounding_size={SIZE_DEMO_BOX_ROUNDING_AX}",
            transform=ax.transAxes,
            facecolor=SIZE_DEMO_BOX_FACE,
            edgecolor=SIZE_DEMO_BOX_EDGE,
            linewidth=SIZE_DEMO_BOX_LINEWIDTH,
            alpha=SIZE_DEMO_BOX_ALPHA,
            zorder=9,
            clip_on=False,
        )
        ax.add_patch(box)


def _build_ticks(min_v, max_v, interval, start_at):
    if interval is None or interval <= 0 or min_v is None or max_v is None:
        return None
    start = min_v if start_at is None else start_at
    while start < min_v:
        start += interval
    ticks = []
    v = start
    eps = interval * 1e-9
    while v <= max_v + eps:
        ticks.append(v)
        v += interval
    return ticks if ticks else None


def _resolve_preset(metric_col: str, fps_col: str):
    mode = str(PRESET_MODE).strip().lower()
    key = (str(metric_col).strip().lower(), str(fps_col).strip().lower())

    if mode == "none":
        return None
    if mode == "lasot_orincpu":
        return PRESET_LASOT_ORINCPU
    if mode == "tknet_oringpu":
        return PRESET_TKNET_ORINGPU

    # auto mode
    if key == ("lasot", "orincpu"):
        return PRESET_LASOT_ORINCPU
    if key == ("tknet", "oringpu"):
        return PRESET_TKNET_ORINGPU
    return None


def _annotate_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    fontsize: int,
    base_offset_px: Tuple[int, int],
    color: Optional[str] = None,
    fontfamily: Optional[str] = None,
    fontweight: Union[str, int] = "normal",
    stroke_width: float = 2.5,
    stroke_color: str = "white",
) -> Tuple[List[plt.Text], List[float], List[float]]:
    """Place text labels at each data point.

    Returns (texts, xs, ys) so the caller can pass them to adjustText.
    The texts are initially placed at the data-point coordinates; adjustText
    will move them and optionally draw leader arrows.
    """
    import matplotlib.patheffects as pe

    texts: List[plt.Text] = []
    xs: List[float] = []
    ys: List[float] = []
    for _, row in df.iterrows():
        x_val = float(row[xcol])
        y_val = float(row[ycol])
        t = ax.text(
            x_val,
            y_val,
            str(row[DEFAULT_METHOD_COL]),
            fontsize=fontsize,
            color=color or "#333333",
            ha="center",
            va="center",
            fontfamily=fontfamily or _TICK_FONT_FAMILY,
            fontweight=fontweight,
            zorder=5,
        )
        t.set_path_effects([
            pe.withStroke(linewidth=stroke_width, foreground=stroke_color),
            pe.Normal(),
        ])
        texts.append(t)
        xs.append(x_val)
        ys.append(y_val)
    return texts, xs, ys


def main() -> int:
    csv_path = DEFAULT_CSV_PATH
    gmacs_col = DEFAULT_GMACS_COL
    logx = USE_LOGX
    dpi = OUTPUT_DPI
    label_all = LABEL_ALL_POINTS
    label_ours = LABEL_OURS_POINTS
    label_topk = LABEL_TOPK_OTHERS
    connect_series = CONNECT_SERIES

    preset_map = {
        "lasot_orincpu": PRESET_LASOT_ORINCPU,
        "tknet_oringpu": PRESET_TKNET_ORINGPU,
    }

    # Build one global color map so the same series/method family keeps the same color
    # across all preset figures.
    df_global = pd.read_csv(csv_path)
    required_for_color = {DEFAULT_METHOD_COL, DEFAULT_SERIES_COL}
    missing_for_color = [c for c in required_for_color if c not in df_global.columns]
    if missing_for_color:
        raise SystemExit(
            f"Missing columns for color mapping: {missing_for_color}. Available: {list(df_global.columns)}"
        )

    method_global = df_global[DEFAULT_METHOD_COL].astype(str).str.strip()
    df_global = df_global[~method_global.str.startswith("-")].copy()
    df_global["_series"] = df_global[DEFAULT_SERIES_COL].astype(str)
    empty_series_global = df_global["_series"].str.strip().eq("")
    df_global.loc[empty_series_global, "_series"] = df_global.loc[
        empty_series_global, DEFAULT_METHOD_COL
    ].apply(_infer_series_name)
    series_color_global = _build_series_color_map(df_global["_series"].tolist(), COLOR_PALETTE)

    for job in RENDER_PRESETS:
        metric_col = job["metric_col"]
        fps_col = job["fps_col"]
        preset = preset_map.get(str(job.get("preset", "")).lower())
        if metric_col not in SUPPORTED_Y_METRICS:
            raise SystemExit(
                f"Unsupported y-axis metric: {metric_col}. Supported: {SUPPORTED_Y_METRICS}"
            )

        fig_size = FIGSIZE
        x_min_cfg, x_max_cfg, x_interval_cfg, x_start_cfg = X_MIN, X_MAX, X_INTERVAL, X_START_AT
        y_min_cfg, y_max_cfg, y_interval_cfg, y_start_cfg = Y_MIN, Y_MAX, Y_INTERVAL, Y_START_AT
        if preset is not None:
            fig_size = preset.get("figsize", fig_size)
            x_min_cfg = preset.get("x_min", x_min_cfg)
            x_max_cfg = preset.get("x_max", x_max_cfg)
            x_interval_cfg = preset.get("x_interval", x_interval_cfg)
            x_start_cfg = preset.get("x_start", x_start_cfg)
            y_min_cfg = preset.get("y_min", y_min_cfg)
            y_max_cfg = preset.get("y_max", y_max_cfg)
            y_interval_cfg = preset.get("y_interval", y_interval_cfg)
            y_start_cfg = preset.get("y_start", y_start_cfg)
        legend_anchor_cfg = preset.get("legend_anchor", SIZE_DEMO_ANCHOR) if preset is not None else SIZE_DEMO_ANCHOR
        legend_margin_x_cfg = (
            preset.get("legend_margin_x_ax", SIZE_DEMO_MARGIN_X_AX)
            if preset is not None
            else SIZE_DEMO_MARGIN_X_AX
        )
        legend_margin_y_cfg = (
            preset.get("legend_margin_y_ax", SIZE_DEMO_MARGIN_Y_AX)
            if preset is not None
            else SIZE_DEMO_MARGIN_Y_AX
        )
        legend_clamp_cfg = (
            preset.get("legend_clamp_inside_axes", SIZE_DEMO_CLAMP_INSIDE_AXES)
            if preset is not None
            else SIZE_DEMO_CLAMP_INSIDE_AXES
        )

        df = pd.read_csv(csv_path)

        required_cols = {
            DEFAULT_METHOD_COL,
            DEFAULT_PUB_COL,
            DEFAULT_SERIES_COL,
            fps_col,
            metric_col,
            gmacs_col,
        }
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

        df = df.copy()
        df[fps_col] = _to_float(df[fps_col])
        df[metric_col] = _to_float(df[metric_col])
        df[gmacs_col] = _to_float(df[gmacs_col])

        # Exclude rows whose method name starts with '-'
        method_series = df[DEFAULT_METHOD_COL].astype(str).str.strip()
        df = df[~method_series.str.startswith("-")]

        df = df.dropna(subset=[fps_col, metric_col])
        df = df[(df[fps_col] > 0) & (df[metric_col] > 0)]
        df = df.reset_index(drop=True)

        # Identify ours: explicit 'Our/Ours' in series column OR MaST prefix.
        is_ours = df[DEFAULT_PUB_COL].astype(str).str.strip().str.lower().isin(["our", "ours"]) | df[
            DEFAULT_METHOD_COL
        ].astype(str).str.startswith("MaST")

        # Infer series for connecting lines.
        df["_series"] = df[DEFAULT_SERIES_COL].astype(str)
        empty_series = df["_series"].str.strip().eq("")
        df.loc[empty_series, "_series"] = df.loc[empty_series, DEFAULT_METHOD_COL].apply(
            _infer_series_name
        )

        # Point sizes from GMACs.
        df["_size"] = _compute_sizes_from_gmacs(
            df[gmacs_col],
            base_mode=SIZE_BASE_MODE,
            min_bubble_radius_pt=MIN_BUBBLE_RADIUS_PT,
            min_bubble_area_pt2=MIN_BUBBLE_AREA_PT2,
            max_bubble_radius_pt=MAX_BUBBLE_RADIUS_PT,
            fallback_gmacs=GMACS_FALLBACK,
        )

        fig, ax = plt.subplots(figsize=fig_size)

        base = df[~is_ours]
        ours = df[is_ours]

        # Layer order:
        # 1) all bubbles, 2) all lines, 3) all center dots
        # Inside each type, earlier rows stay on top of later rows.
        z_step = 0.01
        n_rows = len(df)

        # 1) bubbles
        z_base_bubble = 2.0
        for row_idx, row in df.iterrows():
            bubble_color = series_color_global.get(row["_series"], (0.2, 0.2, 0.2, 1.0))
            is_this_ours = bool(is_ours.loc[row_idx])
            if is_this_ours and HIGHLIGHT_OURS:
                bubble_color = OURS_COLOR
                edge_color = OURS_EDGE_COLOR
                edge_width = OURS_EDGE_WIDTH
            else:
                edge_color = EDGE_COLOR_DEFAULT
                edge_width = EDGE_WIDTH_DEFAULT

            # Earlier rows should stay on top of later rows.
            z = z_base_bubble + (n_rows - row_idx) * z_step
            ax.scatter(
                [row[fps_col]],
                [row[metric_col]],
                s=float(row["_size"]),
                alpha=POINT_ALPHA,
                edgecolors=edge_color,
                linewidths=edge_width,
                color=[bubble_color],
                zorder=z,
            )

        # 2) lines (ordered by first row index of each series)
        if connect_series:
            series_items = []
            for series_name, sub in df.groupby("_series"):
                if len(sub) < 2:
                    continue
                first_idx = int(sub.index.min())
                series_items.append((first_idx, series_name, sub))

            series_items.sort(key=lambda x: x[0])  # earlier row first
            z_base_line = 3.0
            for first_idx, series_name, sub in series_items:
                sub = sub.sort_values(fps_col)
                line_color = series_color_global.get(series_name, (0.2, 0.2, 0.2, 1.0))
                if HIGHLIGHT_OURS and is_ours.loc[sub.index].all():
                    line_color = OURS_COLOR
                line_color = _darken_color(line_color, SERIES_LINE_DARKEN)
                z_line = z_base_line + (n_rows - first_idx) * z_step
                ax.plot(
                    sub[fps_col],
                    sub[metric_col],
                    color=line_color,
                    alpha=SERIES_LINE_ALPHA,
                    linewidth=SERIES_LINE_WIDTH,
                    linestyle=SERIES_LINE_STYLE,
                    zorder=z_line,
                )

        # 3) center dots
        if CENTER_DOT_ENABLED:
            z_base_center = 4.0
            for row_idx, row in df.iterrows():
                zc = z_base_center + (n_rows - row_idx) * z_step
                ax.scatter(
                    [row[fps_col]],
                    [row[metric_col]],
                    s=CENTER_DOT_SIZE,
                    color=CENTER_DOT_COLOR,
                    edgecolors="none",
                    zorder=zc,
                )

        _lbl_texts: List[plt.Text] = []
        _lbl_xs: List[float] = []
        _lbl_ys: List[float] = []

        if label_all:
            # Non-ours: normal weight
            if len(base) > 0:
                t, x, y = _annotate_points(
                    ax,
                    base,
                    fps_col,
                    metric_col,
                    fontsize=LABEL_FONT_SIZE,
                    base_offset_px=LABEL_OFFSET_PX,
                    color=LABEL_COLOR,
                    fontfamily=LABEL_FONT_FAMILY,
                    fontweight=LABEL_FONT_WEIGHT,
                    stroke_width=LABEL_STROKE_WIDTH,
                    stroke_color=LABEL_STROKE_COLOR,
                )
                _lbl_texts.extend(t); _lbl_xs.extend(x); _lbl_ys.extend(y)
            # Ours: heavier weight + slightly larger
            if len(ours) > 0:
                t, x, y = _annotate_points(
                    ax,
                    ours,
                    fps_col,
                    metric_col,
                    fontsize=LABEL_FONT_SIZE + LABEL_OURS_FONT_SIZE_DELTA,
                    base_offset_px=LABEL_OFFSET_PX,
                    color=LABEL_OURS_COLOR,
                    fontfamily=LABEL_FONT_FAMILY,
                    fontweight=LABEL_OURS_FONT_WEIGHT,
                    stroke_width=LABEL_OURS_STROKE_WIDTH,
                    stroke_color=LABEL_STROKE_COLOR,
                )
                _lbl_texts.extend(t); _lbl_xs.extend(x); _lbl_ys.extend(y)
        else:
            if label_ours and len(ours) > 0:
                t, x, y = _annotate_points(
                    ax,
                    ours,
                    fps_col,
                    metric_col,
                    fontsize=LABEL_FONT_SIZE + LABEL_OURS_FONT_SIZE_DELTA,
                    base_offset_px=LABEL_OFFSET_PX,
                    color=LABEL_OURS_COLOR,
                    fontfamily=LABEL_FONT_FAMILY,
                    fontweight=LABEL_OURS_FONT_WEIGHT,
                    stroke_width=LABEL_OURS_STROKE_WIDTH,
                    stroke_color=LABEL_STROKE_COLOR,
                )
                _lbl_texts.extend(t); _lbl_xs.extend(x); _lbl_ys.extend(y)

            if label_topk and label_topk > 0 and len(base) > 0:
                topk = base.sort_values(metric_col, ascending=False).head(label_topk)
                t, x, y = _annotate_points(
                    ax,
                    topk,
                    fps_col,
                    metric_col,
                    fontsize=LABEL_FONT_SIZE,
                    base_offset_px=(LABEL_OFFSET_PX[0], -LABEL_OFFSET_PX[1] - 6),
                    color=LABEL_COLOR,
                    fontfamily=LABEL_FONT_FAMILY,
                    fontweight=LABEL_FONT_WEIGHT,
                    stroke_width=LABEL_STROKE_WIDTH,
                    stroke_color=LABEL_STROKE_COLOR,
                )
                _lbl_texts.extend(t); _lbl_xs.extend(x); _lbl_ys.extend(y)

        if _lbl_texts:
            if LABEL_ADJUST_TEXT:
                from adjustText import adjust_text as _adjust_text

                _arrow_kw: Optional[dict] = (
                    dict(arrowstyle="-", color=LABEL_ADJUST_ARROW_COLOR, lw=LABEL_ADJUST_ARROW_LW)
                    if LABEL_ADJUST_ARROW
                    else None
                )
                _adjust_text(
                    _lbl_texts,
                    x=_lbl_xs,
                    y=_lbl_ys,
                    ax=ax,
                    expand=LABEL_ADJUST_EXPAND,
                    force_text=LABEL_ADJUST_FORCE_TEXT,
                    force_static=LABEL_ADJUST_FORCE_STATIC,
                    arrowprops=_arrow_kw,
                )

        if SHOW_AXIS_TITLES:
            ax.set_xlabel(
                f"Speed (FPS) on {fps_col.replace('FPS_', '')}",
                fontsize=AXIS_LABEL_FONTSIZE,
            )
            ax.set_ylabel(f"Accuracy ({metric_col})", fontsize=AXIS_LABEL_FONTSIZE)
        if SHOW_MAIN_TITLE:
            ax.set_title(f"{metric_col} Speed-Accuracy", fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

        ax.margins(x=AXIS_MARGIN_X, y=AXIS_MARGIN_Y)

        if logx:
            ax.set_xscale("log")

        # Axis range/ticks from top config / preset
        x_min = x_min_cfg if x_min_cfg is not None else float(df[fps_col].min())
        x_max = x_max_cfg if x_max_cfg is not None else float(df[fps_col].max())
        y_min = y_min_cfg if y_min_cfg is not None else float(df[metric_col].min())
        y_max = y_max_cfg if y_max_cfg is not None else float(df[metric_col].max())

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        x_ticks = _build_ticks(x_min, x_max, x_interval_cfg, x_start_cfg)
        y_ticks = _build_ticks(y_min, y_max, y_interval_cfg, y_start_cfg)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)

        ax.grid(True, which="both", linestyle="--", linewidth=0.6 * SCALE, alpha=0.35)

        if SHOW_SIZE_LEGEND:
            _draw_size_demo(
                ax,
                SIZE_LEGEND_GMACS,
                anchor=legend_anchor_cfg,
                margin_x_ax=legend_margin_x_cfg,
                margin_y_ax=legend_margin_y_cfg,
                clamp_inside_axes=legend_clamp_cfg,
            )

        # Apply Book Antiqua to tick labels — must happen after set_xticks/set_yticks.
        fig.canvas.draw()  # force tick label Text objects to be created
        for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
            _lbl.set_fontfamily(_TICK_FONT_FAMILY)
            _lbl.set_fontsize(TICK_FONTSIZE)

        fig.tight_layout()
        out_pdf = csv_path.resolve().parent / f"speed_accuracy_{metric_col}_{fps_col}.pdf"
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)
        fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
        plt.close(fig)

        print(f"Saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

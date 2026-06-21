"""Model plot with swapped N DOF/Actuator axes; data loaded with pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon


MODEL_XLSX = Path("models_data.xlsx")
OUT_PNG    = Path("figure_with_models.png")
OUT_PDF    = Path("figure_with_models.pdf")

# =============================================================================
# Style
# =============================================================================

plt.rcParams.update(
    {
        "font.family"     : "sans-serif",
        "font.sans-serif" : ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype"    : 42,
        "ps.fonttype"     : 42,
    }
)

ACTUATOR_COLOR          = "#3b4fd1"
NDOF_COLOR              = "#0087d7"
CONTROLLER_COLOR        = "#7b4ab2"
ENVIRONMENT_COLOR       = "#2f8f46"
BOX_COLOR               = "#6f6f6f"

ROBOT_COLOR             = "#d00000"
MODEL_COLOR             = "#000000"

AXIS_LINEWIDTH          = 2.2
BOX_LINEWIDTH           = 0.72
PROJECTION_LINEWIDTH    = 0.68
VERTICAL_LINEWIDTH      = 1.1
READOUT_AXIS_LINEWIDTH  = 1.35

SPHERE_ALPHA            = 0.65
VERTICAL_LINE_ALPHA     = 0.65
PROJECTION_ALPHA        = 0.78

# =============================================================================
# Geometry
# =============================================================================

ORIGIN          = np.array([1.25, 2.58])
EMBODIMENT_VEC  = np.array([7.75, 0.00])
CONTROLLER_VEC  = np.array([2.35, -2.66])
ENVIRONMENT_VEC = np.array([0.00, 3.40])

MAIN_ACTUATOR_OFFSET = 0.06
NDOF_START           = ORIGIN
ACTUATOR_START       = ORIGIN + MAIN_ACTUATOR_OFFSET * CONTROLLER_VEC

LOWER_NDOF_BACKOFFSET = 0.06
LOWER_ACTUATOR_START  = ORIGIN + CONTROLLER_VEC
LOWER_NDOF_START      = ORIGIN + (1.0 - LOWER_NDOF_BACKOFFSET) * CONTROLLER_VEC

AXIS_TICK_START = 0.12
AXIS_TICK_END   = 0.90

FIGSIZE         = (12.2, 7.6)
DPI             = 160
X_LIMITS        = (0.0, 14.20)
Y_LIMITS        = (-0.95, 7.15)


def clip_01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def axis_pos(coord: float) -> float:
    return AXIS_TICK_START + clip_01(coord) * (AXIS_TICK_END - AXIS_TICK_START)


def project_point(controller: float = 0.0,
                  embodiment: float = 0.0,
                  environment: float = 0.0) -> np.ndarray:
    return ORIGIN + controller * CONTROLLER_VEC + embodiment * EMBODIMENT_VEC + environment * ENVIRONMENT_VEC


def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# =============================================================================
# Ticks
# =============================================================================

CONTROLLER_TICKS = [
    (0.00, "No\ncontroller"),
    (0.25, "Oscillators,\nFinite-state\nmachines"),
    (0.50, "Leaky\ninteg.\nneuron"),
    (0.75, "Integ.\nand Fire\nneuron"),
    (1.00, "H.H.\nneuron"),
]

ENVIRONMENT_TICKS = [
    (0.00, "Low"),
    (0.50, "Medium"),
    (1.00, "High"),
]

ACTUATOR_TICKS = [
    (0.00, "No actuator"),
    (0.25, "Servomotor"),
    (0.50, "Torque control"),
    (0.75, "Ekeberg-muscle"),
    (1.00, "Hill-muscle"),
]

NDOF_TICKS = [
    (0.00, "0"),
    (0.20, "10"),
    (0.40, "20"),
    (0.60, "30"),
    (0.80, "40"),
    (1.00, "50+"),
]


# =============================================================================
# Drawing utilities
# =============================================================================

def draw_arrow(ax, start: np.ndarray, vec: np.ndarray, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(start + vec),
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            lw=AXIS_LINEWIDTH,
            shrinkA=0,
            shrinkB=0,
            zorder=10,
        )
    )


def draw_tick(ax, axis_start: np.ndarray, axis_vec: np.ndarray,
              position: float, color: str, length: float = 0.16,
              lw: float = 1.25, zorder: int = 9) -> np.ndarray:
    center = axis_start + position * axis_vec
    normal = unit_vector(np.array([-axis_vec[1], axis_vec[0]]))
    a = center - 0.5 * length * normal
    b = center + 0.5 * length * normal
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=zorder)
    return center


def box_vertices() -> dict[str, np.ndarray]:
    return {
        "000": ORIGIN,
        "100": ORIGIN + EMBODIMENT_VEC,
        "010": ORIGIN + CONTROLLER_VEC,
        "110": ORIGIN + EMBODIMENT_VEC + CONTROLLER_VEC,
        "001": ORIGIN + ENVIRONMENT_VEC,
        "101": ORIGIN + EMBODIMENT_VEC + ENVIRONMENT_VEC,
        "011": ORIGIN + CONTROLLER_VEC + ENVIRONMENT_VEC,
        "111": ORIGIN + EMBODIMENT_VEC + CONTROLLER_VEC + ENVIRONMENT_VEC,
    }


def draw_box_faces(ax) -> None:
    v = box_vertices()
    faces = [
        [v["000"], v["100"], v["101"], v["001"]],
        [v["010"], v["110"], v["111"], v["011"]],
        [v["000"], v["010"], v["011"], v["001"]],
        [v["100"], v["110"], v["111"], v["101"]],
        [v["001"], v["101"], v["111"], v["011"]],
    ]
    for face, alpha in zip(faces, [0.035, 0.035, 0.025, 0.025, 0.045]):
        ax.add_patch(Polygon(face, closed=True, facecolor="white",
                             edgecolor="none", alpha=alpha, zorder=5))


def draw_box_edges(ax) -> None:
    v = box_vertices()
    edges = [
        ("000", "100"), ("100", "110"), ("110", "010"), ("010", "000"),
        ("001", "101"), ("101", "111"), ("111", "011"), ("011", "001"),
        ("000", "001"), ("100", "101"), ("010", "011"), ("110", "111"),
    ]
    for a, b in edges:
        p0, p1 = v[a], v[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                color=BOX_COLOR, lw=BOX_LINEWIDTH, ls=(0, (4, 3)),
                alpha=0.75, zorder=6)


def draw_axes(ax) -> None:
    draw_arrow(ax, NDOF_START, EMBODIMENT_VEC, NDOF_COLOR)
    draw_arrow(ax, ACTUATOR_START, EMBODIMENT_VEC, ACTUATOR_COLOR)
    draw_arrow(ax, ORIGIN, CONTROLLER_VEC, CONTROLLER_COLOR)
    draw_arrow(ax, ORIGIN, ENVIRONMENT_VEC, ENVIRONMENT_COLOR)


def draw_axis_titles(ax) -> None:
    ax.text(*(ORIGIN + ENVIRONMENT_VEC + np.array([-0.22, -0.00])),
            "Richness of\nenvironment", ha="right", va="bottom",
            fontsize=15.0, fontweight="bold", color=ENVIRONMENT_COLOR)

    ax.text(*(ORIGIN + CONTROLLER_VEC + np.array([-1.50, -0.60])),
            "Controller", ha="left", va="center",
            fontsize=15.0, fontweight="bold", color=CONTROLLER_COLOR)

    anchor = ORIGIN + EMBODIMENT_VEC + np.array([0.1, 0.85])
    ax.text(*anchor, "Embodiment:", ha="left", va="center",
            fontsize=14.2, fontweight="bold", color=MODEL_COLOR, zorder=13)
    ax.text(*(anchor + np.array([0.0, -0.36])), "N DOF +",
            ha="left", va="center", fontsize=14.2,
            fontweight="bold", color=NDOF_COLOR, zorder=13)
    ax.text(*(anchor + np.array([0.0, -0.72])), "Actuator",
            ha="left", va="center", fontsize=14.2,
            fontweight="bold", color=ACTUATOR_COLOR, zorder=13)


def draw_environment_ticks(ax) -> None:
    for coordinate, label in ENVIRONMENT_TICKS:
        tick = draw_tick(ax, ORIGIN, ENVIRONMENT_VEC, axis_pos(coordinate), ENVIRONMENT_COLOR)
        ax.text(tick[0] - 0.16, tick[1], label,
                ha="right", va="center", fontsize=11.0, color=ENVIRONMENT_COLOR)


def draw_controller_ticks(ax) -> None:
    outside_normal = unit_vector(np.array([-CONTROLLER_VEC[1], CONTROLLER_VEC[0]]))
    offsets = [0.12, 0.06, 0.00, -0.05, -0.10]
    for (coordinate, label), dy in zip(CONTROLLER_TICKS, offsets):
        tick = draw_tick(ax, ORIGIN, CONTROLLER_VEC, axis_pos(coordinate), CONTROLLER_COLOR)
        label_position = tick - 0.36 * outside_normal + np.array([-0.05, dy])
        ax.text(label_position[0], label_position[1], label,
                ha="right", va="center", fontsize=8.3,
                color=CONTROLLER_COLOR, zorder=13)


def draw_embodiment_axis_ticks(ax) -> None:
    for coordinate, _ in NDOF_TICKS:
        draw_tick(ax, NDOF_START, EMBODIMENT_VEC, axis_pos(coordinate),
                  NDOF_COLOR, length=0.14)
    for coordinate, _ in ACTUATOR_TICKS:
        draw_tick(ax, ACTUATOR_START, EMBODIMENT_VEC, axis_pos(coordinate),
                  ACTUATOR_COLOR, length=0.15)


def draw_readout_axes(ax) -> None:
    for start, color in [(LOWER_NDOF_START, NDOF_COLOR),
                         (LOWER_ACTUATOR_START, ACTUATOR_COLOR)]:
        ax.plot([start[0], start[0] + EMBODIMENT_VEC[0]],
                [start[1], start[1] + EMBODIMENT_VEC[1]],
                color=color, lw=READOUT_AXIS_LINEWIDTH, zorder=8)

    for coordinate, label in NDOF_TICKS:
        tick = draw_tick(ax, LOWER_NDOF_START, EMBODIMENT_VEC,
                         axis_pos(coordinate), NDOF_COLOR, length=0.14)
        ax.text(tick[0], tick[1] + 0.16, label,
                ha="center", va="bottom", fontsize=8.4,
                color=NDOF_COLOR, zorder=13)

    for coordinate, label in ACTUATOR_TICKS:
        tick = draw_tick(ax, LOWER_ACTUATOR_START, EMBODIMENT_VEC,
                         axis_pos(coordinate), ACTUATOR_COLOR, length=0.15)
        ax.text(tick[0], tick[1] - 0.18, label,
                ha="center", va="top", fontsize=8.0,
                color=ACTUATOR_COLOR, zorder=13)


def draw_axis_ticks_and_labels(ax) -> None:
    draw_environment_ticks(ax)
    draw_controller_ticks(ax)
    draw_embodiment_axis_ticks(ax)
    draw_readout_axes(ax)


def setup_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.axis("off")
    return fig, ax

def to_bool(value) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def load_models(path: Path = MODEL_XLSX) -> list[dict]:
    """Load the model table with pandas."""
    df = pd.read_excel(path, sheet_name="Models")
    df = df[df["include_in_plot"].astype(str).str.lower().isin(["yes", "true", "1"])]

    models = []
    for _, row in df.iterrows():
        models.append(
            {
                "label": str(row["plot_label"]),
                "is_robot": to_bool(row["is_robot"]),
                "color": ROBOT_COLOR if str(row["plot_color"]).lower() == "red" else MODEL_COLOR,
                "controller_base": float(row["controller_coord"]),
                "actuator": float(row["actuator_coord"]),
                "ndof": float(row["ndof_coord"]),
                "environment": float(row["environment_coord"]),
                "complexity_score": float(row.get("complexity_score", 0.0)),
                "label_dx": float(row.get("label_dx", 0.15)),
                "label_dy": float(row.get("label_dy", 0.20)),
            }
        )
    return models


def add_xy_offsets(entries: list[dict], max_offset: float = 0.10) -> list[dict]:
    """Offset models with the same actuator/controller position along the controller axis.

    The offset is only applied to the plotted model position. The controller
    projection still returns to the original controller tick.
    """
    grouped: dict[tuple[float, float], list[int]] = {}
    copied = [entry.copy() for entry in entries]

    for i, entry in enumerate(copied):
        key = (round(entry["controller_base"], 4), round(entry["actuator"], 4))
        grouped.setdefault(key, []).append(i)

    for indices in grouped.values():
        if len(indices) == 1:
            copied[indices[0]]["controller_plot"] = copied[indices[0]]["controller_base"]
            continue

        indices_sorted = sorted(indices, key=lambda i: copied[i]["complexity_score"])
        n = len(indices_sorted)

        # Simpler entries receive the smaller displacement, then progressively
        # more complex entries move farther along the controller axis.
        raw_offsets = np.linspace(0.0, max_offset, n)

        base = copied[indices_sorted[0]]["controller_base"]
        if base + raw_offsets[-1] > 1.0:
            raw_offsets = raw_offsets - (base + raw_offsets[-1] - 1.0)

        for idx, offset in zip(indices_sorted, raw_offsets):
            copied[idx]["controller_plot"] = clip_01(copied[idx]["controller_base"] + float(offset))

    return copied


def draw_sphere(ax, x: float, y: float, radius: float = 0.1,
                color: str = MODEL_COLOR, alpha: float = SPHERE_ALPHA) -> None:
    base_color = np.array(to_rgb(color))
    n_pixels = 90
    yy, xx = np.mgrid[-1:1:complex(n_pixels), -1:1:complex(n_pixels)]
    r2 = xx**2 + yy**2
    mask = r2 <= 1.0

    zz = np.zeros_like(xx)
    zz[mask] = np.sqrt(1.0 - r2[mask])

    light = unit_vector(np.array([-0.55, 0.65, 0.60]))
    normals = np.dstack([xx, yy, zz])
    lambert = np.maximum(0.0, normals @ light)
    highlight = np.exp(-((xx + 0.38) ** 2 + (yy - 0.42) ** 2) / 0.028)

    shade = 0.36 + 0.64 * lambert
    rgb = base_color[None, None, :] * shade[:, :, None]
    rgb = np.clip(rgb + 0.55 * highlight[:, :, None], 0.0, 1.0)

    rim = np.clip((1.0 - r2) / 0.18, 0.0, 1.0)
    rgba = np.zeros((n_pixels, n_pixels, 4))
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = alpha * mask.astype(float) * (0.55 + 0.45 * rim)

    ax.imshow(rgba, extent=[x - radius, x + radius, y - radius, y + radius],
              origin="lower", interpolation="bilinear", zorder=12)


def data_delta_from_pixels(ax, dx: float, dy: float) -> np.ndarray:
    inv = ax.transData.inverted()
    p0 = inv.transform((0, 0))
    p1 = inv.transform((dx, dy))
    return p1 - p0


def relax_text_labels(fig, ax, texts: list, iterations: int = 240, step_px: float = 5.0) -> None:
    if not texts:
        return

    for _ in range(iterations):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        moved = False
        boxes = [t.get_window_extent(renderer=renderer).expanded(1.06, 1.14) for t in texts]

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue

                ci = boxes[i].get_points().mean(axis=0)
                cj = boxes[j].get_points().mean(axis=0)
                direction = ci - cj
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm else np.array([0.0, 1.0])
                direction = direction + np.array([0.20, 0.12])
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm else direction

                delta = data_delta_from_pixels(ax, step_px * direction[0], step_px * direction[1])
                xi, yi = texts[i].get_position()
                xj, yj = texts[j].get_position()
                texts[i].set_position((xi + delta[0], yi + delta[1]))
                texts[j].set_position((xj - delta[0], yj - delta[1]))
                moved = True

        if not moved:
            break


def draw_model(ax, entry: dict):
    controller_plot = axis_pos(entry["controller_plot"])
    controller_base = axis_pos(entry["controller_base"])
    actuator = axis_pos(entry["actuator"])
    ndof = axis_pos(entry["ndof"])
    env = axis_pos(entry["environment"])

    # Model x-position follows discrete actuator type.
    floor_point = project_point(controller_plot, actuator, 0.0)
    real_point = project_point(controller_plot, actuator, env)

    # Projection to original controller tick, not to the offset model position.
    controller_axis_point = project_point(controller_base, 0.0, 0.0)

    # Actuator projection to upper/main Actuator axis.
    actuator_axis_point = ACTUATOR_START + actuator * EMBODIMENT_VEC

    # N DOF projection to lower/secondary N DOF axis.
    ndof_axis_point = LOWER_NDOF_START + ndof * EMBODIMENT_VEC

    proj_style = {
        "lw": PROJECTION_LINEWIDTH,
        "ls": (0, (1.2, 2.5)),
        "alpha": PROJECTION_ALPHA,
        "zorder": 2,
    }

    # Controller projections
    proj_c_alpha = PROJECTION_ALPHA
    ax.plot([controller_axis_point[0], floor_point[0]],
            [controller_axis_point[1], floor_point[1]],
            color=CONTROLLER_COLOR, **{**proj_style, "alpha": proj_c_alpha})
    # Actuator projections
    proj_a_alpha = 0.0
    ax.plot([actuator_axis_point[0], floor_point[0]],
            [actuator_axis_point[1], floor_point[1]],
            color=ACTUATOR_COLOR, **{**proj_style, "alpha": proj_a_alpha})
    # N DOF projections
    proj_n_alpha = PROJECTION_ALPHA
    ax.plot([ndof_axis_point[0], floor_point[0]],
            [ndof_axis_point[1], floor_point[1]],
            color=NDOF_COLOR, **{**proj_style, "alpha": proj_n_alpha})

    ax.plot([floor_point[0], real_point[0]],
            [floor_point[1], real_point[1]],
            color=ENVIRONMENT_COLOR, lw=VERTICAL_LINEWIDTH,
            alpha=VERTICAL_LINE_ALPHA, zorder=4)

    ax.scatter([floor_point[0]], [floor_point[1]], marker="D", s=8,
               facecolor="white", edgecolor=entry["color"],
               linewidth=1.05, zorder=9)

    draw_sphere(ax, real_point[0], real_point[1], color=entry["color"])

    text = ax.text(
        real_point[0] + entry["label_dx"],
        real_point[1] + entry["label_dy"],
        entry["label"],
        fontsize=8.1,
        color=entry["color"],
        fontweight="bold" if entry["is_robot"] else "normal",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=0.15),
        zorder=13,
    )
    return text


def make_figure(models: list[dict] | None = None) -> plt.Figure:
    if models is None:
        models = add_xy_offsets(load_models())

    fig, ax = setup_axes()
    texts = []
    for entry in models:
        texts.append(draw_model(ax, entry))

    draw_box_faces(ax)
    draw_box_edges(ax)
    draw_axes(ax)
    draw_axis_titles(ax)
    draw_axis_ticks_and_labels(ax)
    relax_text_labels(fig, ax, texts)

    return fig


def main() -> None:
    fig = make_figure()
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
    print(f"Loaded models from: {MODEL_XLSX}")
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()

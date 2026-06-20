"""
Plot neuromechanical model coordinates from an editable Excel workbook.

Input workbook:
    /mnt/data/neuromechanical_models_full_table_tickspace.xlsx

Coordinate convention:
    - workbook coordinates run from 0 to 1 between the first and last tick.
    - the first and last ticks are offset inside the physical axis limits.
    - each model receives exactly one projection to each of:
      controller, actuator, and N DOF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon


OUT_PNG = Path("figure_with_boxes.png")
OUT_PDF = Path("figure_with_boxes.pdf")

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

NDOF_OFFSET     = 0.08
NDOF_START      = ORIGIN + NDOF_OFFSET * CONTROLLER_VEC

AXIS_TICK_START = 0.12
AXIS_TICK_END   = 0.90

FIGSIZE         = (12.2, 7.6)
DPI             = 160
X_LIMITS        = (0.0, 12.9)
Y_LIMITS        = (-0.60, 6.6)


def clip_01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def axis_pos(coord: float) -> float:
    """Map semantic coordinate 0..1 to an offset axis position."""
    return AXIS_TICK_START + clip_01(coord) * (AXIS_TICK_END - AXIS_TICK_START)


def project_point(
    controller  : float = 0.0,
    embodiment  : float = 0.0,
    environment : float = 0.0,
) -> np.ndarray:
    return (
        ORIGIN
        + controller  * CONTROLLER_VEC
        + embodiment  * EMBODIMENT_VEC
        + environment * ENVIRONMENT_VEC
    )


def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# =============================================================================
# Ticks - coordinates are semantic; draw locations use axis_pos(...)
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
# Red template / anchor boxes
# =============================================================================

RED_COLOR             = "#ff1f1f"
RED_BOX_LINEWIDTH     = 0.65
RED_ARROW_LINEWIDTH   = 1.35
RED_FACE_ALPHA        = 0.10
RED_DASH_PATTERN      = (0, (1.2, 1.2))

# Box sizes are expressed in semantic coordinates. Because all three edge sizes
# are equal, the plotted red boxes preserve the same edge-length proportions as
# the main pseudo-3D box.
TEMPLATE_BOX_SIZE = 0.30
ANCHOR_BOX_SIZE   = 0.30


# =============================================================================
# Drawing utilities
# =============================================================================

def draw_arrow(ax, start: np.ndarray, vec: np.ndarray, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(start + vec),
            arrowstyle     = "-|>",
            mutation_scale = 14,
            color          = color,
            lw             = AXIS_LINEWIDTH,
            shrinkA        = 0,
            shrinkB        = 0,
            zorder         = 10,
        )
    )


def draw_tick(
    ax,
    axis_start: np.ndarray,
    axis_vec: np.ndarray,
    position: float,
    color: str,
    length: float = 0.16,
    lw: float = 1.25,
    zorder: int = 9,
) -> np.ndarray:
    center = axis_start + position * axis_vec
    normal = unit_vector(np.array([-axis_vec[1], axis_vec[0]]))
    a = center - 0.5 * length * normal
    b = center + 0.5 * length * normal
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=zorder)
    return center


def real_box_point(
    controller  : float = 0.0,
    embodiment  : float = 0.0,
    environment : float = 0.0,
) -> np.ndarray:
    """Project true 0..1 coordinates of the large box.

    This deliberately bypasses axis_pos(...): the red boxes and red diagonals
    are anchored to the actual corners of the main 3D box, not to the inset
    tick positions.
    """
    return project_point(
        controller  = controller,
        embodiment  = embodiment,
        environment = environment,
    )


def box_vertices() -> dict[str, np.ndarray]:
    return {
        "000" : ORIGIN,
        "100" : ORIGIN + EMBODIMENT_VEC,
        "010" : ORIGIN + CONTROLLER_VEC,
        "110" : ORIGIN + EMBODIMENT_VEC + CONTROLLER_VEC,
        "001" : ORIGIN + ENVIRONMENT_VEC,
        "101" : ORIGIN + EMBODIMENT_VEC + ENVIRONMENT_VEC,
        "011" : ORIGIN + CONTROLLER_VEC + ENVIRONMENT_VEC,
        "111" : ORIGIN + EMBODIMENT_VEC + CONTROLLER_VEC + ENVIRONMENT_VEC,
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
        ax.add_patch(
            Polygon(
                face,
                closed    = True,
                facecolor = "white",
                edgecolor = "none",
                alpha     = alpha,
                zorder    = 5,
            )
        )


def draw_box_edges(ax) -> None:
    v = box_vertices()
    edges = [
        ("000", "100"), ("100", "110"), ("110", "010"), ("010", "000"),
        ("001", "101"), ("101", "111"), ("111", "011"), ("011", "001"),
        ("000", "001"), ("100", "101"), ("010", "011"), ("110", "111"),
    ]

    for a, b in edges:
        p0, p1 = v[a], v[b]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color  = BOX_COLOR,
            lw     = BOX_LINEWIDTH,
            ls     = (0, (4, 3)),
            alpha  = 0.75,
            zorder = 6,
        )


def small_box_vertices(
    corner: tuple[float, float, float],
    size: float,
    anchor_at_111: bool = False,
) -> dict[str, np.ndarray]:
    """Return vertices for a red semantic cube.

    If anchor_at_111 is False, `corner` is the lower/front/left semantic corner.
    If anchor_at_111 is True, `corner` is the upper/back/right real-box corner.
    """
    c, e, z = corner

    if anchor_at_111:
        c0, e0, z0 = c - size, e - size, z - size
    else:
        c0, e0, z0 = c, e, z

    c1, e1, z1 = c0 + size, e0 + size, z0 + size

    return {
        "000" : real_box_point(c0, e0, z0),
        "100" : real_box_point(c0, e1, z0),
        "010" : real_box_point(c1, e0, z0),
        "110" : real_box_point(c1, e1, z0),
        "001" : real_box_point(c0, e0, z1),
        "101" : real_box_point(c0, e1, z1),
        "011" : real_box_point(c1, e0, z1),
        "111" : real_box_point(c1, e1, z1),
    }


def draw_red_box(
    ax,
    corner: tuple[float, float, float],
    size: float,
    anchor_at_111: bool = False,
) -> dict[str, np.ndarray]:
    """Draw a semi-transparent red pseudo-3D box."""
    v = small_box_vertices(corner, size, anchor_at_111=anchor_at_111)

    faces = [
        [v["000"], v["100"], v["101"], v["001"]],
        [v["010"], v["110"], v["111"], v["011"]],
        [v["000"], v["010"], v["011"], v["001"]],
        [v["100"], v["110"], v["111"], v["101"]],
        [v["001"], v["101"], v["111"], v["011"]],
    ]

    for face, alpha in zip(faces, [0.10, 0.08, 0.06, 0.06, 0.12]):
        ax.add_patch(
            Polygon(
                face,
                closed    = True,
                facecolor = RED_COLOR,
                edgecolor = "none",
                alpha     = alpha,
                zorder    = 11,
            )
        )

    edges = [
        ("000", "100"), ("100", "110"), ("110", "010"), ("010", "000"),
        ("001", "101"), ("101", "111"), ("111", "011"), ("011", "001"),
        ("000", "001"), ("100", "101"), ("010", "011"), ("110", "111"),
    ]

    for a, b in edges:
        p0, p1 = v[a], v[b]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color  = RED_COLOR,
            lw     = RED_BOX_LINEWIDTH,
            ls     = RED_DASH_PATTERN,
            zorder = 12,
        )

    return v


def draw_red_projection_arrows(ax) -> None:
    """Draw red diagonals between true main-box coordinates."""
    start      = real_box_point(0.0, 0.0, 0.0)
    end_full   = real_box_point(1.0, 1.0, 1.0)
    end_floor  = real_box_point(1.0, 1.0, 0.0)

    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end_full),
            arrowstyle     = "-|>",
            mutation_scale = 14,
            color          = RED_COLOR,
            lw             = RED_ARROW_LINEWIDTH,
            linestyle      = "-",
            shrinkA        = 0,
            shrinkB        = 0,
            zorder         = 13,
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end_floor),
            arrowstyle     = "-|>",
            mutation_scale = 14,
            color          = RED_COLOR,
            lw             = RED_ARROW_LINEWIDTH,
            linestyle      = (0, (2, 2)),
            shrinkA        = 0,
            shrinkB        = 0,
            zorder         = 13,
        )
    )


# =============================================================================
# Axes and labels
# =============================================================================

def draw_axes(ax) -> None:
    draw_arrow(ax, ORIGIN,     EMBODIMENT_VEC,  ACTUATOR_COLOR)
    draw_arrow(ax, NDOF_START, EMBODIMENT_VEC,  NDOF_COLOR)
    draw_arrow(ax, ORIGIN,     CONTROLLER_VEC,  CONTROLLER_COLOR)
    draw_arrow(ax, ORIGIN,     ENVIRONMENT_VEC, ENVIRONMENT_COLOR)


def draw_axis_titles(ax) -> None:
    ax.text(
        *(ORIGIN + ENVIRONMENT_VEC + np.array([-0.22, -0.20])),
        "Richness of\nenvironment",
        ha         = "right",
        va         = "bottom",
        fontsize   = 15.0,
        fontweight = "bold",
        color      = ENVIRONMENT_COLOR,
    )

    ax.text(
        *(ORIGIN + CONTROLLER_VEC + np.array([-1.50, -0.60])),
        "Controller",
        ha         = "left",
        va         = "center",
        fontsize   = 15.0,
        fontweight = "bold",
        color      = CONTROLLER_COLOR,
    )

    anchor = ORIGIN + EMBODIMENT_VEC + np.array([-2.40, -0.70])

    ax.text(
        *anchor,
        "Embodiment:",
        ha         = "left",
        va         = "center",
        fontsize   = 14.2,
        fontweight = "bold",
        color      = MODEL_COLOR,
        zorder     = 13,
    )
    ax.text(
        *(anchor + np.array([1.8, 0.0])),
        "Actuator +",
        ha         = "left",
        va         = "center",
        fontsize   = 14.2,
        fontweight = "bold",
        color      = ACTUATOR_COLOR,
        zorder     = 13,
    )
    ax.text(
        *(anchor + np.array([3.2, 0.0])),
        "N DOF",
        ha         = "left",
        va         = "center",
        fontsize   = 14.2,
        fontweight = "bold",
        color      = NDOF_COLOR,
        zorder     = 13,
    )


def draw_environment_ticks(ax) -> None:
    for coordinate, label in ENVIRONMENT_TICKS:
        tick = draw_tick(
            ax,
            ORIGIN,
            ENVIRONMENT_VEC,
            axis_pos(coordinate),
            ENVIRONMENT_COLOR,
        )
        ax.text(
            tick[0] - 0.16,
            tick[1],
            label,
            ha       = "right",
            va       = "center",
            fontsize = 11.0,
            color    = ENVIRONMENT_COLOR,
        )


def draw_controller_ticks(ax) -> None:
    outside_normal = unit_vector(np.array([-CONTROLLER_VEC[1], CONTROLLER_VEC[0]]))
    offsets = [0.12, 0.06, 0.00, -0.05, -0.10]

    for (coordinate, label), dy in zip(CONTROLLER_TICKS, offsets):
        tick = draw_tick(
            ax,
            ORIGIN,
            CONTROLLER_VEC,
            axis_pos(coordinate),
            CONTROLLER_COLOR,
        )
        label_position = tick - 0.36 * outside_normal + np.array([-0.05, dy])

        ax.text(
            label_position[0],
            label_position[1],
            label,
            ha       = "right",
            va       = "center",
            fontsize = 8.3,
            color    = CONTROLLER_COLOR,
            zorder   = 13,
        )


def draw_embodiment_axis_ticks(ax) -> None:
    for coordinate, _ in ACTUATOR_TICKS:
        draw_tick(
            ax,
            ORIGIN,
            EMBODIMENT_VEC,
            axis_pos(coordinate),
            ACTUATOR_COLOR,
            length=0.15,
        )

    for coordinate, _ in NDOF_TICKS:
        draw_tick(
            ax,
            NDOF_START,
            EMBODIMENT_VEC,
            axis_pos(coordinate),
            NDOF_COLOR,
            length=0.14,
        )


def draw_readout_axes(ax) -> None:
    ndof_start     = ORIGIN + CONTROLLER_VEC
    actuator_start = ndof_start - NDOF_OFFSET * CONTROLLER_VEC

    for start, color in [(actuator_start, ACTUATOR_COLOR), (ndof_start, NDOF_COLOR)]:
        ax.plot(
            [start[0], start[0] + EMBODIMENT_VEC[0]],
            [start[1], start[1] + EMBODIMENT_VEC[1]],
            color  = color,
            lw     = READOUT_AXIS_LINEWIDTH,
            zorder = 8,
        )

    for coordinate, label in ACTUATOR_TICKS:
        tick = draw_tick(
            ax,
            actuator_start,
            EMBODIMENT_VEC,
            axis_pos(coordinate),
            ACTUATOR_COLOR,
            length=0.15,
        )
        ax.text(
            tick[0],
            tick[1] + 0.18,
            label,
            ha       = "center",
            va       = "bottom",
            fontsize = 8.1,
            color    = ACTUATOR_COLOR,
            zorder   = 13,
        )

    for coordinate, label in NDOF_TICKS:
        tick = draw_tick(
            ax,
            ndof_start,
            EMBODIMENT_VEC,
            axis_pos(coordinate),
            NDOF_COLOR,
            length=0.14,
        )
        ax.text(
            tick[0],
            tick[1] - 0.18,
            label,
            ha       = "center",
            va       = "top",
            fontsize = 8.4,
            color    = NDOF_COLOR,
            zorder   = 13,
        )


def draw_axis_ticks_and_labels(ax) -> None:
    draw_environment_ticks(ax)
    draw_controller_ticks(ax)
    draw_embodiment_axis_ticks(ax)
    draw_readout_axes(ax)


# =============================================================================
# Main
# =============================================================================

def make_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.axis("off")

    draw_box_faces(ax)
    draw_box_edges(ax)
    draw_axes(ax)
    draw_axis_titles(ax)
    draw_axis_ticks_and_labels(ax)

    draw_red_box(
        ax,
        corner       = (0.0, 0.0, 0.0),
        size         = TEMPLATE_BOX_SIZE,
        anchor_at_111 = False,
    )
    draw_red_box(
        ax,
        corner       = (1.0, 1.0, 1.0),
        size         = ANCHOR_BOX_SIZE,
        anchor_at_111 = True,
    )
    draw_red_projection_arrows(ax)

    return fig


def main() -> None:
    fig = make_figure()
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()

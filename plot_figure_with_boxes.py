"""
Pseudo-3D neuromechanical axes with template/anchor cubes.

This version keeps the axis orientation, colors, and main box style of the
swapped-axis plot, but removes the individual model markers.
Instead it shows:
    - one red cube starting at the origin (0, 0, 0)
    - one red cube touching the upper corner (1, 1, 1)
    - one solid red arrow from (0, 0, 0) to (1, 1, 1)
    - one dotted red arrow from (0, 0, 0) to (1, 1, 0)

Embodiment tick labels are placed next to the Actuator and N DOF axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon


# =============================================================================
# Output files
# =============================================================================

OUT_PNG = Path("figure_with_boxes.png")
OUT_PDF = Path("figure_with_boxes.pdf")


# =============================================================================
# Font settings
# =============================================================================

plt.rcParams.update(
    {
        "font.family"     : "sans-serif",
        "font.sans-serif" : ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype"    : 42,
        "ps.fonttype"     : 42,
    }
)


# =============================================================================
# Style
# =============================================================================

ACTUATOR_COLOR       = "#3b4fd1"
NDOF_COLOR           = "#0087d7"
CONTROLLER_COLOR     = "#7b4ab2"
ENVIRONMENT_COLOR    = "#2f8f46"
BOX_COLOR            = "#6f6f6f"

MODEL_COLOR          = "#000000"
RED_COLOR            = "#ff1f1f"

AXIS_LINEWIDTH       = 2.2
BOX_LINEWIDTH        = 0.72
RED_BOX_LINEWIDTH    = 1.5
RED_ARROW_LINEWIDTH  = 1.35
READOUT_AXIS_LINEWIDTH = 1.35


# =============================================================================
# Plot geometry
# =============================================================================

ORIGIN          = np.array([1.25, 2.58])
EMBODIMENT_VEC  = np.array([7.75, 0.00])
CONTROLLER_VEC  = np.array([2.10, -2.38])
ENVIRONMENT_VEC = np.array([0.00, 3.40])

NDOF_OFFSET     = 0.08
NDOF_START      = ORIGIN + NDOF_OFFSET * CONTROLLER_VEC

FIGSIZE         = (12.2, 7.6)
DPI             = 160
X_LIMITS        = (0.0, 11.9)
Y_LIMITS        = (0.0, 6.6)


def project_point(
    controller  : float = 0.0,
    embodiment  : float = 0.0,
    environment : float = 0.0,
) -> np.ndarray:
    """Project normalized 3-axis coordinates onto the 2D plot."""
    return (
        ORIGIN
        + controller  * CONTROLLER_VEC
        + embodiment  * EMBODIMENT_VEC
        + environment * ENVIRONMENT_VEC
    )


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Return a vector with length 1."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# =============================================================================
# Axis coordinates and labels
# =============================================================================

OSC    = 0.16
LEAKY  = 0.38
LIF    = 0.58
HH     = 0.75

LOW_ENV  = 0.15
MID_ENV  = 0.50
HIGH_ENV = 0.85

NO_ACTUATOR = 0.08
SERVO       = 0.28
TORQUE      = 0.48
EKEBERG     = 0.68
HILL        = 0.88

LOW_DOF  = 0.18
MID_DOF  = 0.48
HIGH_DOF = 0.78

CONTROLLER_TICKS = [
    (OSC,   "Oscillators,\nFinite-state\nmachines"),
    (LEAKY, "Leaky\ninteg.\nneuron"),
    (LIF,   "Integ.\nand Fire\nneuron"),
    (HH,    "H.H.\nneuron"),
]

ENVIRONMENT_TICKS = [
    (LOW_ENV,  "Low"),
    (MID_ENV,  "Medium"),
    (HIGH_ENV, "High"),
]

ACTUATOR_TICKS = [
    (NO_ACTUATOR, "No actuator"),
    (SERVO,       "Servomotor"),
    (TORQUE,      "Torque control"),
    (EKEBERG,     "Ekeberg-muscle"),
    (HILL,        "Hill-muscle"),
]

NDOF_TICKS = [
    (LOW_DOF,  "Low"),
    (MID_DOF,  "Medium"),
    (HIGH_DOF, "High"),
]


# =============================================================================
# Red cube settings
# =============================================================================

# Use the same normalized size on all axes. Because each edge is built from
# one of the main axis vectors, this preserves the same edge-length ratios as
# the big box while scaling the red boxes up in volume.
ORIGIN_CUBE = {
    "origin"       : (0.00, 0.00, 0.00),
    "controller"   : 0.20,
    "embodiment"   : 0.20,
    "environment"  : 0.20,
    "anchor_at_111": False,
}

CORNER_CUBE = {
    # This cube is defined so that its top/back/right vertex coincides with
    # the main-box corner (1, 1, 1).
    "origin"       : (1.00, 1.00, 1.00),
    "controller"   : 0.20,
    "embodiment"   : 0.20,
    "environment"  : 0.20,
    "anchor_at_111": True,
}


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_arrow(ax, start: np.ndarray, vec: np.ndarray, color: str) -> None:
    """Draw one axis arrow."""
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
    """Draw one tick perpendicular to an axis and return its center."""
    center = axis_start + position * axis_vec
    normal = unit_vector(np.array([-axis_vec[1], axis_vec[0]]))

    a = center - 0.5 * length * normal
    b = center + 0.5 * length * normal

    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color  = color,
        lw     = lw,
        zorder = zorder,
    )
    return center


def box_vertices() -> dict[str, np.ndarray]:
    """Return the eight vertices of the main pseudo-3D box."""
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
    """Draw faint box faces so the main box looks closed."""
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
    """Draw the dashed edges of the main pseudo-3D box."""
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


def mini_box_vertices(
    controller: float,
    embodiment: float,
    environment: float,
    size_controller: float,
    size_embodiment: float,
    size_environment: float,
    anchor_at_111: bool = False,
) -> dict[str, np.ndarray]:
    """Return the 8 vertices of a small pseudo-3D cube/box."""
    if anchor_at_111:
        anchor = project_point(controller, embodiment, environment)
        base = (
            anchor
            - size_controller  * CONTROLLER_VEC
            - size_embodiment  * EMBODIMENT_VEC
            - size_environment * ENVIRONMENT_VEC
        )
    else:
        base = project_point(controller, embodiment, environment)

    e_vec = size_embodiment * EMBODIMENT_VEC
    c_vec = size_controller * CONTROLLER_VEC
    z_vec = size_environment * ENVIRONMENT_VEC

    return {
        "000" : base,
        "100" : base + e_vec,
        "010" : base + c_vec,
        "110" : base + e_vec + c_vec,
        "001" : base + z_vec,
        "101" : base + e_vec + z_vec,
        "011" : base + c_vec + z_vec,
        "111" : base + e_vec + c_vec + z_vec,
    }


def draw_dashed_red_box(ax, spec: dict) -> dict[str, np.ndarray]:
    """Draw one semi-transparent dashed red pseudo-3D box and return its vertices.

    The box vertices are built explicitly from the three main plot axes, so every
    red-box edge remains parallel to one of the main 3D axes.
    """
    controller, embodiment, environment = spec["origin"]

    v = mini_box_vertices(
        controller       = controller,
        embodiment       = embodiment,
        environment      = environment,
        size_controller  = spec["controller"],
        size_embodiment  = spec["embodiment"],
        size_environment = spec["environment"],
        anchor_at_111    = spec.get("anchor_at_111", False),
    )

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

    # Each edge connects two vertices that differ along only one of the three
    # axis directions, which guarantees parallelism with the main axes.
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
            ls     = (0, (1.2, 1.2)),
            zorder = 12,
        )

    return v


def draw_red_projection_arrows(ax) -> None:
    """Draw the two red projection arrows requested by the user."""
    start        = project_point(0.0, 0.0, 0.0)
    end_full     = project_point(1.0, 1.0, 1.0)
    end_dotted   = project_point(1.0, 1.0, 0.0)

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
            zorder         = 11,
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end_dotted),
            arrowstyle     = "-|>",
            mutation_scale = 14,
            color          = RED_COLOR,
            lw             = RED_ARROW_LINEWIDTH,
            linestyle      = (0, (3, 3)),
            shrinkA        = 0,
            shrinkB        = 0,
            zorder         = 11,
        )
    )


# =============================================================================
# Figure drawing
# =============================================================================

def draw_axes(ax) -> None:
    """Draw the four conceptual axes."""
    draw_arrow(ax, ORIGIN,     EMBODIMENT_VEC,  ACTUATOR_COLOR)
    draw_arrow(ax, NDOF_START, EMBODIMENT_VEC,  NDOF_COLOR)
    draw_arrow(ax, ORIGIN,     CONTROLLER_VEC,  CONTROLLER_COLOR)
    draw_arrow(ax, ORIGIN,     ENVIRONMENT_VEC, ENVIRONMENT_COLOR)


def draw_axis_titles(ax) -> None:
    """Draw the main axis titles."""
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
        *(ORIGIN + CONTROLLER_VEC + np.array([-1.50, -0.30])),
        "Controller",
        ha         = "left",
        va         = "center",
        fontsize   = 15.0,
        fontweight = "bold",
        color      = CONTROLLER_COLOR,
    )

    # Single-line embodiment title, moved down and slightly left.
    anchor = ORIGIN + EMBODIMENT_VEC + np.array([-2.50, -0.45])

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
        *(anchor + np.array([1.70, 0.00])),
        "Actuator +",
        ha         = "left",
        va         = "center",
        fontsize   = 14.2,
        fontweight = "bold",
        color      = ACTUATOR_COLOR,
        zorder     = 13,
    )
    ax.text(
        *(anchor + np.array([3.00, 0.00])),
        "N DOF",
        ha         = "left",
        va         = "center",
        fontsize   = 14.2,
        fontweight = "bold",
        color      = NDOF_COLOR,
        zorder     = 13,
    )


def draw_environment_ticks(ax) -> None:
    """Draw environment ticks and labels on the vertical axis."""
    for coordinate, label in ENVIRONMENT_TICKS:
        tick_center = draw_tick(
            ax,
            ORIGIN,
            ENVIRONMENT_VEC,
            coordinate,
            color  = ENVIRONMENT_COLOR,
            length = 0.16,
        )

        ax.text(
            tick_center[0] - 0.16,
            tick_center[1],
            label,
            ha       = "right",
            va       = "center",
            fontsize = 11.8,
            color    = MODEL_COLOR,
        )


def draw_controller_ticks(ax) -> None:
    """Draw controller ticks and labels outside the main box."""
    outside_normal = unit_vector(np.array([-CONTROLLER_VEC[1], CONTROLLER_VEC[0]]))
    vertical_offsets = [0.20, 0.10, 0.00, -0.10]

    for (coordinate, label), dy in zip(CONTROLLER_TICKS, vertical_offsets):
        tick_center = draw_tick(
            ax,
            ORIGIN,
            CONTROLLER_VEC,
            coordinate,
            color  = CONTROLLER_COLOR,
            length = 0.16,
        )

        label_pos = tick_center - 0.36 * outside_normal + np.array([-0.05, dy])

        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha       = "right",
            va       = "center",
            fontsize = 10.0,
            color    = CONTROLLER_COLOR,
            zorder   = 13,
        )


def draw_embodiment_ticks(ax) -> None:
    """Draw embodiment ticks on the main axes and readable scales at the bottom."""
    # Ticks on the actual Actuator and N DOF axes, without labels.
    for coordinate, _ in ACTUATOR_TICKS:
        draw_tick(
            ax,
            ORIGIN,
            EMBODIMENT_VEC,
            coordinate,
            color  = ACTUATOR_COLOR,
            length = 0.15,
        )

    for coordinate, _ in NDOF_TICKS:
        draw_tick(
            ax,
            NDOF_START,
            EMBODIMENT_VEC,
            coordinate,
            color  = NDOF_COLOR,
            length = 0.14,
        )

    # Bottom readout axes:
    # - N DOF starts exactly at the controller-axis tip
    # - Actuator is offset slightly above it, inside the main box
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

    # Actuator labels go above the lower Actuator axis.
    for coordinate, label in ACTUATOR_TICKS:
        tick_center = draw_tick(
            ax,
            actuator_start,
            EMBODIMENT_VEC,
            coordinate,
            color  = ACTUATOR_COLOR,
            length = 0.15,
        )

        ax.text(
            tick_center[0],
            tick_center[1] + 0.18,
            label,
            ha       = "center",
            va       = "bottom",
            fontsize = 8.6,
            color    = ACTUATOR_COLOR,
            zorder   = 13,
        )

    # N DOF labels go below the lower N DOF axis.
    for coordinate, label in NDOF_TICKS:
        tick_center = draw_tick(
            ax,
            ndof_start,
            EMBODIMENT_VEC,
            coordinate,
            color  = NDOF_COLOR,
            length = 0.14,
        )

        ax.text(
            tick_center[0],
            tick_center[1] - 0.18,
            label,
            ha       = "center",
            va       = "top",
            fontsize = 9.0,
            color    = NDOF_COLOR,
            zorder   = 13,
        )


def draw_axis_ticks_and_labels(ax) -> None:
    """Draw all axis ticks and labels."""
    draw_environment_ticks(ax)
    draw_controller_ticks(ax)
    draw_embodiment_ticks(ax)


def make_figure() -> plt.Figure:
    """Build and return the complete figure."""
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

    draw_dashed_red_box(ax, ORIGIN_CUBE)
    draw_dashed_red_box(ax, CORNER_CUBE)
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

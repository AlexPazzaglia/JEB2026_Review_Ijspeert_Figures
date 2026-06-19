"""
Neuromechanical model-complexity figure.

This script creates a pseudo-3D plot with:
    - Embodiment on the horizontal axis
    - Controller on the oblique/depth axis
    - Environment richness on the vertical axis

How to edit the figure:
    1. Add or move models in MODEL_ENTRIES.
    2. Move axis ticks by editing the *_TICKS sections.
    3. Change colors and line thicknesses in the STYLE section.

Coordinates are normalized values along each axis, roughly from 0 to 1.
For example, HILL = 0.88 means "near the right end of the embodiment axis".
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon


# =============================================================================
# Output files
# =============================================================================

OUT_PNG = Path("figure_with_models.png")
OUT_PDF = Path("figure_with_models.pdf")


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

ACTUATOR_COLOR      = "#3b4fd1"   # dark blue
NDOF_COLOR          = "#0087d7"   # light blue
CONTROLLER_COLOR    = "#7b4ab2"   # purple
ENVIRONMENT_COLOR   = "#2f8f46"   # green
BOX_COLOR           = "#6f6f6f"

ROBOT_COLOR         = "#d00000"
MODEL_COLOR         = "#000000"

AXIS_LINEWIDTH      = 2.2
BOX_LINEWIDTH       = 0.72
PROJECTION_LINEWIDTH = 0.68
VERTICAL_LINEWIDTH  = 1.1

SPHERE_ALPHA        = 0.65
VERTICAL_LINE_ALPHA = 0.65
PROJECTION_ALPHA    = 0.78
READOUT_AXIS_LINEWIDTH = 1.35


# =============================================================================
# Plot geometry
# =============================================================================

ORIGIN          = np.array([1.25, 2.58])
EMBODIMENT_VEC  = np.array([7.75, 0.00])    # long horizontal axis
CONTROLLER_VEC  = np.array([2.10, -2.38])   # oblique/depth axis
ENVIRONMENT_VEC = np.array([0.00, 3.40])    # vertical axis

NDOF_OFFSET     = 0.08
NDOF_START      = ORIGIN + NDOF_OFFSET * CONTROLLER_VEC

FIGSIZE         = (12.2, 7.6)
DPI             = 160
X_LIMITS        = (0.0, 11.9)
Y_LIMITS        = (0.0, 6.6)


def project_point(controller: float = 0.0,
                  embodiment: float = 0.0,
                  environment: float = 0.0) -> np.ndarray:
    """Project normalized 3-axis coordinates onto the 2D canvas."""
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

# Controller axis
OSC    = 0.16
LEAKY  = 0.38
LIF    = 0.58
HH     = 0.75

# Environment axis
LOW_ENV  = 0.15
MID_ENV  = 0.50
HIGH_ENV = 0.85

# Actuator axis
NO_ACTUATOR = 0.08
SERVO       = 0.28
TORQUE      = 0.48
EKEBERG     = 0.68
HILL        = 0.88

# N DOF axis
LOW_DOF  = 0.18     # between No actuator and Servomotor
MID_DOF  = 0.48     # aligned with Torque control
HIGH_DOF = 0.78     # between Ekeberg-muscle and Hill-muscle


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
# Model entries
# =============================================================================

def model(label: str,
          color: str,
          controller: float,
          embodiment: float,
          environment: float,
          label_offset: tuple[float, float],
          fontsize: float = 12.0,
          is_robot: bool = False) -> dict:
    """Create one model-entry dictionary.

    `embodiment` follows the actuator category, not the N DOF category.
    Use `label_offset` to manually move the text label relative to the point.
    """
    return {
        "label"        : label,
        "color"        : color,
        "controller"   : controller,
        "embodiment"   : embodiment,
        "environment"  : environment,
        "label_offset" : label_offset,
        "fontsize"     : fontsize,
        "is_robot"     : is_robot,
    }


LABEL_DY = 0.20

MODEL_ENTRIES = [
    # label                       color        control  actuator      environment  label offset       robot?
    model("Passive\nwalker",      MODEL_COLOR, OSC,     NO_ACTUATOR,  LOW_ENV,     (-0.70, +0.24    ) ),
    model("Dead\ntrout",          MODEL_COLOR, OSC,     NO_ACTUATOR,  MID_ENV,     (-0.65, +0.24    ) ),
    model("SLIP",                 MODEL_COLOR, OSC,     SERVO,        LOW_ENV,     (-0.42, +LABEL_DY) ),
    model("Salamandra\nrobotica", ROBOT_COLOR, OSC,     SERVO,        MID_ENV,     (-1.35, +0.00    ), is_robot=True),
    model("Rhex",                 ROBOT_COLOR, LEAKY,   SERVO,        HIGH_ENV,    (-0.52, +LABEL_DY), is_robot=True),
    model("Zbot",                 ROBOT_COLOR, LEAKY,   SERVO,        HIGH_ENV,    (+0.06, +LABEL_DY), is_robot=True),
    model("Tekken",               ROBOT_COLOR, LIF,     SERVO,        MID_ENV,     (+0.00, +LABEL_DY), is_robot=True),
    model("Fruit Fly",            MODEL_COLOR, LEAKY,   SERVO,        MID_ENV,     (+0.05, +LABEL_DY) ),
    model("Lamprey",              MODEL_COLOR, LEAKY,   EKEBERG,      MID_ENV,     (-0.15, +LABEL_DY) ),
    model("Rodent",               MODEL_COLOR, LIF,     SERVO,        MID_ENV,     (+0.15, +0.00    ) ),
    model("Salamander",           MODEL_COLOR, LIF,     EKEBERG,      MID_ENV,     (+0.15, +0.00    ) ),
    model("C. Elegans",           MODEL_COLOR, HH,      HILL,         LOW_ENV,     (+0.15, +0.00    ) ),
    model("Human 2D",             MODEL_COLOR, SERVO,   HILL,         LOW_ENV,     (-1.15, -LABEL_DY) ),
    model("Human 3D",             MODEL_COLOR, LEAKY,   HILL,         LOW_ENV,     (+0.15, -LABEL_DY) ),
]


# N DOF values used only for the bottom readout-axis projections.
# The horizontal embodiment position itself still follows the actuator category.
MODEL_DOF_VALUES = {
    "Passive\nwalker"      : LOW_DOF,
    "Dead\ntrout"          : LOW_DOF,
    "SLIP"                 : LOW_DOF,
    "Salamandra\nrobotica" : LOW_DOF,
    "Rhex"                 : LOW_DOF,
    "Zbot"                 : LOW_DOF,
    "Tekken"               : LOW_DOF,
    "Fruit Fly"            : MID_DOF,
    "Lamprey"              : MID_DOF,
    "Rodent"               : MID_DOF,
    "Salamander"           : MID_DOF,
    "C. Elegans"           : HIGH_DOF,
    "Human 2D"             : HIGH_DOF,
    "Human 3D"             : HIGH_DOF,
}




# =============================================================================
# Basic drawing utilities
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


def draw_tick(ax,
              axis_start: np.ndarray,
              axis_vec: np.ndarray,
              position: float,
              color: str,
              length: float = 0.16,
              lw: float = 1.25,
              zorder: int = 9) -> np.ndarray:
    """Draw a tick perpendicular to an axis and return its center."""
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
    """Return the eight vertices of the pseudo-3D box."""
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
    """Draw faint white faces so vertical bars appear inside a box."""
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
    """Draw the dashed pseudo-3D box edges."""
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


def draw_sphere(ax,
                x: float,
                y: float,
                radius: float = 0.082,
                color: str = MODEL_COLOR,
                alpha: float = SPHERE_ALPHA) -> None:
    """Draw a semi-transparent shaded sphere marker."""
    base_color = np.array(to_rgb(color))
    n_pixels   = 90

    yy, xx = np.mgrid[-1:1:complex(n_pixels), -1:1:complex(n_pixels)]
    r2     = xx**2 + yy**2
    mask   = r2 <= 1.0

    zz       = np.zeros_like(xx)
    zz[mask] = np.sqrt(1.0 - r2[mask])

    light      = unit_vector(np.array([-0.55, 0.65, 0.60]))
    normals    = np.dstack([xx, yy, zz])
    lambert    = np.maximum(0.0, normals @ light)
    highlight  = np.exp(-((xx + 0.38) ** 2 + (yy - 0.42) ** 2) / 0.028)

    shade = 0.36 + 0.64 * lambert
    rgb   = base_color[None, None, :] * shade[:, :, None]
    rgb   = np.clip(rgb + 0.55 * highlight[:, :, None], 0.0, 1.0)

    rim  = np.clip((1.0 - r2) / 0.18, 0.0, 1.0)
    rgba = np.zeros((n_pixels, n_pixels, 4))

    rgba[:, :, :3] = rgb
    rgba[:, :, 3]  = alpha * mask.astype(float) * (0.55 + 0.45 * rim)

    ax.imshow(
        rgba,
        extent        = [x - radius, x + radius, y - radius, y + radius],
        origin        = "lower",
        interpolation = "bilinear",
        zorder        = 12,
    )


# =============================================================================
# Figure-specific drawing
# =============================================================================

def add_overlap_offsets(entries: list[dict],
                        offset: float = 0.050,
                        actuator_offset: float = 0.040) -> list[dict]:
    """Separate overlaps for identical model coordinates and for shared actuator positions."""
    groups = defaultdict(list)

    # Jitter models that share the same controller and embodiment coordinates.
    for index, entry in enumerate(entries):
        key = (entry["controller"], round(entry["embodiment"], 3))
        groups[key].append(index)

    copied_entries = [entry.copy() for entry in entries]

    for indices in groups.values():
        jitters = [0.0] if len(indices) == 1 else np.linspace(-offset, offset, len(indices))
        for index, jitter in zip(indices, jitters):
            copied_entries[index]["controller_jitter"] = float(jitter)

    # Also separate models that share the same actuator coordinate on the lower actuator axis.
    actuator_groups = defaultdict(list)
    for index, entry in enumerate(entries):
        actuator_groups[round(entry["embodiment"], 3)].append(index)

    for indices in actuator_groups.values():
        jitters = [0.0] if len(indices) == 1 else np.linspace(-actuator_offset, actuator_offset, len(indices))
        for index, jitter in zip(indices, jitters):
            copied_entries[index]["actuator_axis_jitter"] = float(jitter)

    return copied_entries


def nearest_tick(value: float, ticks: list[tuple[float, str]]) -> float:
    """Return the coordinate of the nearest tick."""
    return min((coordinate for coordinate, _ in ticks),
               key = lambda coordinate: abs(coordinate - value))


def draw_model(ax, entry: dict) -> None:
    """Draw one model entry and its dotted projections."""
    controller_base = entry["controller"]
    controller      = controller_base + entry.get("controller_jitter", 0.0)
    actuator        = entry["embodiment"]
    environment     = entry["environment"]
    marker_color    = entry["color"]

    floor_point = project_point(controller, actuator, 0.0)
    real_point  = project_point(controller, actuator, environment)

    controller_tick_point = project_point(controller_base, 0.0, 0.0)

    # Readout axes:
    # - N DOF starts at the controller-axis tip
    # - Actuator is offset slightly above it, inside the main box
    ndof_readout_start     = ORIGIN + CONTROLLER_VEC
    actuator_readout_start = ndof_readout_start - NDOF_OFFSET * CONTROLLER_VEC

    # Apply a small offset when multiple models share the same actuator coordinate.
    actuator_axis_jitter = entry.get("actuator_axis_jitter", 0.0)

    actuator_readout_point = (
        actuator_readout_start
        + actuator * EMBODIMENT_VEC
        + actuator_axis_jitter * CONTROLLER_VEC
    )

    # N DOF is read directly on the original N DOF axis inside the box.
    ndof_value      = MODEL_DOF_VALUES.get(
        entry["label"],
        nearest_tick(actuator, NDOF_TICKS),
    )
    ndof_axis_point = NDOF_START + ndof_value * EMBODIMENT_VEC

    controller_projection = {
        "color"  : CONTROLLER_COLOR,
        "lw"     : PROJECTION_LINEWIDTH,
        "ls"     : (0, (1.2, 2.5)),
        "alpha"  : PROJECTION_ALPHA,
        "zorder" : 2,
    }
    actuator_projection = {
        "color"  : ACTUATOR_COLOR,
        "lw"     : PROJECTION_LINEWIDTH,
        "ls"     : (0, (1.2, 2.5)),
        "alpha"  : PROJECTION_ALPHA,
        "zorder" : 2,
    }
    ndof_projection = {
        "color"  : NDOF_COLOR,
        "lw"     : PROJECTION_LINEWIDTH,
        "ls"     : (0, (1.2, 2.5)),
        "alpha"  : PROJECTION_ALPHA,
        "zorder" : 2,
    }

    ax.plot(
        [controller_tick_point[0], floor_point[0]],
        [controller_tick_point[1], floor_point[1]],
        **controller_projection,
    )

    ax.plot(
        [actuator_readout_point[0], floor_point[0]],
        [actuator_readout_point[1], floor_point[1]],
        **actuator_projection,
    )

    ax.plot(
        [ndof_axis_point[0], floor_point[0]],
        [ndof_axis_point[1], floor_point[1]],
        **ndof_projection,
    )

    ax.plot(
        [floor_point[0], real_point[0]],
        [floor_point[1], real_point[1]],
        color  = ENVIRONMENT_COLOR,
        lw     = VERTICAL_LINEWIDTH,
        alpha  = VERTICAL_LINE_ALPHA,
        zorder = 4,
    )

    ax.scatter(
        [floor_point[0]],
        [floor_point[1]],
        marker    = "D",
        s         = 28,
        facecolor = "white",
        edgecolor = marker_color,
        linewidth = 1.05,
        zorder    = 9,
    )

    draw_sphere(ax, real_point[0], real_point[1], color = marker_color)
    draw_model_label(ax, entry, real_point)


def draw_model_label(ax, entry: dict, real_point: np.ndarray) -> None:
    """Draw the model name close to its sphere."""
    dx, dy = entry["label_offset"]

    ax.text(
        real_point[0] + dx,
        real_point[1] + dy,
        entry["label"],
        fontsize   = entry["fontsize"],
        color      = entry["color"],
        fontweight = "bold" if entry["is_robot"] else "normal",
        ha         = "left",
        va         = "center",
        zorder     = 13,
    )


def draw_axes(ax) -> None:
    """Draw all conceptual axes."""
    draw_arrow(ax, ORIGIN,      EMBODIMENT_VEC,  ACTUATOR_COLOR)
    draw_arrow(ax, NDOF_START,  EMBODIMENT_VEC,  NDOF_COLOR)
    draw_arrow(ax, ORIGIN,      CONTROLLER_VEC,  CONTROLLER_COLOR)
    draw_arrow(ax, ORIGIN,      ENVIRONMENT_VEC, ENVIRONMENT_COLOR)


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
        *(ORIGIN + CONTROLLER_VEC + np.array([-1.50, -0.08])),
        "Controller",
        ha         = "left",
        va         = "center",
        fontsize   = 15.0,
        fontweight = "bold",
        color      = CONTROLLER_COLOR,
    )

    # Stacked title for the two horizontal embodiment axes.
    anchor = ORIGIN + EMBODIMENT_VEC + np.array([0.28, 0.60])

    ax.text(
        *anchor,
        "Embodiment:",
        ha         = "left",
        va         = "center",
        fontsize   = 13.5,
        fontweight = "bold",
        color      = MODEL_COLOR,
        zorder     = 13,
    )
    ax.text(
        *(anchor + np.array([0.0, -0.36])),
        "Actuator +",
        ha         = "left",
        va         = "center",
        fontsize   = 13.5,
        fontweight = "bold",
        color      = ACTUATOR_COLOR,
        zorder     = 13,
    )
    ax.text(
        *(anchor + np.array([0.0, -0.72])),
        "N DOF",
        ha         = "left",
        va         = "center",
        fontsize   = 13.5,
        fontweight = "bold",
        color      = NDOF_COLOR,
        zorder     = 13,
    )


def draw_axis_ticks_and_labels(ax) -> None:
    """Draw axis ticks and place labels where they are least cluttered."""
    draw_environment_ticks(ax)
    draw_controller_ticks(ax)
    draw_embodiment_axis_ticks(ax)
    draw_embodiment_labels_on_back_edge(ax)


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
            fontsize = 9.2,
            color    = ENVIRONMENT_COLOR,
        )


def draw_controller_ticks(ax) -> None:
    """Draw controller ticks and labels just outside the box."""
    outside_normal = unit_vector(np.array([-CONTROLLER_VEC[1], CONTROLLER_VEC[0]]))

    for coordinate, label in CONTROLLER_TICKS:
        tick_center = draw_tick(
            ax,
            ORIGIN,
            CONTROLLER_VEC,
            coordinate,
            color  = CONTROLLER_COLOR,
            length = 0.16,
        )

        label_position = tick_center - 0.32 * outside_normal + np.array([-0.03, -0.01])

        ax.text(
            label_position[0],
            label_position[1],
            label,
            ha       = "right",
            va       = "center",
            fontsize = 7.4,
            color    = CONTROLLER_COLOR,
            zorder   = 13,
        )


def draw_embodiment_axis_ticks(ax) -> None:
    """Keep ticks on both embodiment axes, without labels in the center."""
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


def draw_embodiment_labels_on_back_edge(ax) -> None:
    """Draw separate lower readout axes for actuator and N DOF."""
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

    # Actuator labels above the lower actuator axis.
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

    # N DOF labels below the lower N DOF axis.
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


def make_figure(entries: list[dict] = MODEL_ENTRIES) -> plt.Figure:
    """Build and return the complete figure."""
    fig, ax = plt.subplots(figsize = FIGSIZE, dpi = DPI)

    ax.set_aspect("equal", adjustable = "box")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.axis("off")

    for entry in add_overlap_offsets(entries):
        draw_model(ax, entry)

    draw_box_faces(ax)
    draw_box_edges(ax)
    draw_axes(ax)
    draw_axis_titles(ax)
    draw_axis_ticks_and_labels(ax)

    return fig


def main() -> None:
    fig = make_figure()

    fig.savefig(OUT_PNG, bbox_inches = "tight", pad_inches = 0.08)
    fig.savefig(OUT_PDF, bbox_inches = "tight", pad_inches = 0.08)

    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()

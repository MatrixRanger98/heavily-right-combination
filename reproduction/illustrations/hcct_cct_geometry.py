"""Illustrate bounded HCCT and infinite-volume CCT regions in two dimensions.

The setup is the m=2, k=1 specialization of the blockwise
divide-and-combine model in Appendix A.3.3.  With theta=(x,y), zero
blockwise estimates, unit scale, and radial Half-Cauchy p-values,

    R_H = {(x,y): |x| + |y| <= 2 q_H},

while

    R_C = {(x,y): [f(|x|)+f(|y|)]/4 <= q_C},
    f(r) = r - 1/r.

The script produces a direct geometric comparison and a separate tail plot
showing disjoint positive-area tubes inside the CCT horn.

Recovered without mathematical/style changes from the manuscript figure
generator (SHA-256 d44608bea40eaa434081d555b0ef2b94dc5dd2a32e09401af2e0088e8705e5ec).
The original remains protected; this entry point adds immutable output handling.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf as save_figure_pdf

EXPERIMENT_ID = "illustrations/hcct-cct-geometry"
ALPHA = 0.05
Q_C = float(1.0 / np.tan(np.pi * ALPHA))
# Exact independent-study HCCT critical value reported in Appendix A.3.2.
Q_H = 13.68751
HCCT_RADIUS = 2.0 * Q_H

BLUE = "#4472C4"
ORANGE = "#D97706"
GRAY = "#666666"
LIGHT_GRAY = "#D0D0D0"
CENTRAL_LIMIT = 35.0
HORN_Y_MIN = 30.0
HORN_Y_MAX = 1.0e6
HORN_U_LIMIT = 7.0


def f_score(r: np.ndarray | float) -> np.ndarray | float:
    """Return f(r)=r-1/r for positive r."""

    return r - 1.0 / r


def inverse_f(a: np.ndarray | float) -> np.ndarray | float:
    """Return the positive solution r of r-1/r=a."""

    return 0.5 * (a + np.sqrt(a * a + 4.0))


def cct_half_width(y: np.ndarray | float) -> np.ndarray | float:
    """Exact x half-width of the CCT region at a nonzero vertical coordinate."""

    y_abs = np.abs(np.asarray(y, dtype=float))
    if np.any(y_abs == 0):
        raise ValueError("cct_half_width is defined here only for nonzero y")
    return inverse_f(4.0 * Q_C - f_score(y_abs))


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=LIGHT_GRAY, linewidth=0.6, alpha=0.55)
    ax.tick_params(labelsize=9)


def add_continuation_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.8},
        annotation_clip=False,
    )


def save_pdf(fig: plt.Figure, artifacts: ArtifactStore, stem: str) -> None:
    """Save an illustrative figure as a vector PDF."""

    label = stem.removesuffix("_no_caption").replace("_", "-")
    save_figure_pdf(fig, artifacts, f"hcct-cct-geometry-{label}.pdf", tight=True)


def make_caption_free_figures(artifacts: ArtifactStore) -> None:
    """Create four standalone panels with no embedded titles or panel labels."""

    limit = CENTRAL_LIMIT

    # 1. HCCT diamond.
    fig, ax = plt.subplots(figsize=(5.4, 5.1), constrained_layout=True)
    diamond = np.array(
        [
            [HCCT_RADIUS, 0.0],
            [0.0, HCCT_RADIUS],
            [-HCCT_RADIUS, 0.0],
            [0.0, -HCCT_RADIUS],
        ]
    )
    ax.add_patch(
        Polygon(
            diamond,
            closed=True,
            facecolor=BLUE,
            edgecolor=BLUE,
            alpha=0.24,
            linewidth=2.0,
        )
    )
    ax.axhline(0.0, color=GRAY, linewidth=0.8)
    ax.axvline(0.0, color=GRAY, linewidth=0.8)
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        aspect="equal",
        xlabel=r"$\theta_1$",
        ylabel=r"$\theta_2$",
    )
    ax.text(
        0.0,
        -31.8,
        rf"$|\theta_1|+|\theta_2|\leq {HCCT_RADIUS:.2f}$",
        ha="center",
        va="bottom",
        fontsize=10,
        color=BLUE,
    )
    style_axis(ax)
    save_pdf(fig, artifacts, "hcct_region_no_caption")
    plt.close(fig)

    # 2. CCT central region with four continuation arrows.
    y_negative = np.linspace(-limit, -1e-5, 2500)
    y_positive = np.linspace(1e-5, limit, 2500)
    y_values = np.concatenate([y_negative, y_positive])
    widths = np.minimum(cct_half_width(y_values), limit)
    right = np.column_stack([widths, y_values])
    left = np.column_stack([-widths[::-1], y_values[::-1]])
    cct_polygon = np.vstack([right, left])

    fig, ax = plt.subplots(figsize=(5.4, 5.1), constrained_layout=True)
    ax.add_patch(
        Polygon(
            cct_polygon,
            closed=True,
            facecolor=ORANGE,
            edgecolor=ORANGE,
            alpha=0.24,
            linewidth=1.8,
        )
    )
    ax.axhline(0.0, color=ORANGE, linewidth=1.5)
    ax.axvline(0.0, color=ORANGE, linewidth=1.5)
    add_continuation_arrow(ax, (29.0, 0.0), (36.0, 0.0))
    add_continuation_arrow(ax, (-29.0, 0.0), (-36.0, 0.0))
    add_continuation_arrow(ax, (0.0, 29.0), (0.0, 36.0))
    add_continuation_arrow(ax, (0.0, -29.0), (0.0, -36.0))
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        aspect="equal",
        xlabel=r"$\theta_1$",
        ylabel=r"$\theta_2$",
    )
    ax.text(
        0.0,
        31.0,
        "horns continue beyond every finite window",
        ha="center",
        va="top",
        fontsize=9.5,
        color=ORANGE,
    )
    style_axis(ax)
    save_pdf(fig, artifacts, "cct_region_no_caption")
    plt.close(fig)

    # 3. Magnified upper CCT horn.
    y_tail = np.geomspace(HORN_Y_MIN, HORN_Y_MAX, 1600)
    u_width = y_tail * cct_half_width(y_tail)
    fig, ax = plt.subplots(figsize=(5.7, 5.1), constrained_layout=True)
    ax.fill_betweenx(
        y_tail,
        -u_width,
        u_width,
        facecolor=ORANGE,
        edgecolor=ORANGE,
        alpha=0.24,
        linewidth=1.5,
    )
    ax.plot(u_width, y_tail, color=ORANGE, linewidth=1.5)
    ax.plot(-u_width, y_tail, color=ORANGE, linewidth=1.5)
    ax.axvline(1.0, color=GRAY, linestyle="--", linewidth=1.1)
    ax.axvline(-1.0, color=GRAY, linestyle="--", linewidth=1.1)
    ax.annotate(
        r"$\theta_2\to\infty$",
        xy=(0.0, 9e5),
        xytext=(0.0, 1.7e5),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.5},
        color=ORANGE,
        fontsize=10,
    )
    ax.text(
        0.0,
        42.0,
        r"actual coordinate: $\theta_1=u/\theta_2$",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=GRAY,
    )
    ax.set(
        xlim=(-HORN_U_LIMIT, HORN_U_LIMIT),
        yscale="log",
        xlabel=r"magnified coordinate $u=\theta_1\theta_2$",
        ylabel=r"$\theta_2$ (log scale)",
    )
    style_axis(ax)
    save_pdf(fig, artifacts, "cct_horn_magnified_no_caption")
    plt.close(fig)

    # 4. Equal-area tubes inside the upper CCT horn.
    y_min = 28.0
    y_max = 4.0e4
    x_min = 2.0e-6
    x_max = 0.35
    y_values = np.geomspace(y_min, y_max, 2400)
    widths = cct_half_width(y_values)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.fill_betweenx(
        y_values,
        x_min,
        widths,
        where=widths >= x_min,
        color=ORANGE,
        alpha=0.17,
        label=r"CCT region: $0<\theta_1\leq x_{\max}(\theta_2)$",
    )
    ax.plot(widths, y_values, color=ORANGE, linewidth=2.0, label="exact horn boundary")
    ax.plot(
        1.0 / y_values,
        y_values,
        color=GRAY,
        linestyle="--",
        linewidth=1.4,
        label=r"tail asymptote $\theta_1=1/\theta_2$",
    )
    dyadic_levels = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for index, level in enumerate(dyadic_levels):
        x_left = 1.0 / (8.0 * level)
        x_right = 1.0 / (4.0 * level)
        ax.add_patch(
            Rectangle(
                (x_left, level),
                x_right - x_left,
                level,
                facecolor=BLUE,
                edgecolor=BLUE,
                alpha=0.34,
                linewidth=1.0,
                label=r"disjoint tubes, each of area $1/8$" if index == 0 else None,
            )
        )
    ax.annotate(
        "infinitely many disjoint\nequal-area rectangles",
        xy=(1.0 / (4.0 * 512.0), 750.0),
        xytext=(0.014, 2500.0),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2},
        color=BLUE,
        fontsize=10,
        ha="center",
    )
    ax.text(
        4.0e-6,
        70.0,
        "shaded region continues toward $\\theta_1=0$",
        color=ORANGE,
        fontsize=9.5,
        rotation=90,
        va="bottom",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\theta_1$ (log scale)")
    ax.set_ylabel(r"$\theta_2$ (log scale)")
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    save_pdf(fig, artifacts, "cct_infinite_area_tubes_no_caption")
    plt.close(fig)


def main() -> None:
    """Write the recovered caption-free panels to a new immutable run."""
    artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
    configuration = {
        "alpha": ALPHA,
        "cauchy_cutoff": Q_C,
        "half_cauchy_cutoff": Q_H,
        "cutoff_source": "Historical rounded appendix value; retained for visual reproduction",
        "original_generator": "Paper/fig/hcct_cct_geometry/plot_hcct_cct_regions.py",
        "original_source_sha256": "d44608bea40eaa434081d555b0ef2b94dc5dd2a32e09401af2e0088e8705e5ec",
        "central_limit": CENTRAL_LIMIT,
        "horn_y_range": [HORN_Y_MIN, HORN_Y_MAX],
        "horn_u_limit": HORN_U_LIMIT,
    }
    with artifacts.data("hcct-cct-geometry-config.json").open("x", encoding="utf-8") as output:
        json.dump(configuration, output, indent=2, allow_nan=False)
        output.write("\n")
    make_caption_free_figures(artifacts)


if __name__ == "__main__":
    main()

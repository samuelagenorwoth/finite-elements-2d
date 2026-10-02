"""Figures for the README (saved in ../figures)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
from PIL import Image
from fem import mesh_square, mesh_lshape, solve

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
INK, TEAL, ORANGE, GREY = "#15222E", "#0B6E78", "#D9822B", "#8C8C8C"
CMAP = LinearSegmentedColormap.from_list("site", ["#FFFFFF", "#DCECEE", TEAL, INK])
OUTLINES = {
    "mesh_square": np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
    "mesh_lshape": np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1], [0, 0]]),
}
plt.rcParams.update({"font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK, "axes.titlecolor": INK})


def draw(ax, mesh, n, edges=False, vmax=None):
    nodes, tri = mesh(n)
    u = solve(nodes, tri)
    T = Triangulation(nodes[:, 0], nodes[:, 1], tri)
    im = ax.tripcolor(T, u, shading="gouraud", cmap=CMAP, vmin=0, vmax=vmax)
    if edges:
        ax.triplot(T, color="#6B8A92", lw=0.5, alpha=0.6)
    outline = OUTLINES[mesh.__name__]
    ax.plot(outline[:, 0], outline[:, 1], color=INK, lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    return im, u


def solutions():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, mesh, name in [(axes[0], mesh_square, "Unit square"),
                           (axes[1], mesh_lshape, "L-shaped domain")]:
        im, u = draw(ax, mesh, 64)
        ax.set_title(f"{name}   (max u = {u.max():.4f})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, label="u")
    fig.suptitle(r"Solution of $u - \Delta u = 1$, $u = 0$ on the boundary (mesh 64 x 64)")
    fig.savefig(FIGURES / "solutions.png", dpi=170)
    plt.close(fig)


def convergence():
    d = np.load(ROOT / "results" / "convergence.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharey=True)
    for ax, dom, name, (pL2, pH1) in [(axes[0], "square", "Unit square", (2, 1)),
                                      (axes[1], "lshape", "L-shaped domain", (4 / 3, 2 / 3))]:
        h = d[f"{dom}_h"]
        for key, color, label, p in [("L2", TEAL, "$L^2$ error", pL2),
                                     ("H1", ORANGE, "$H^1$ error (gradient)", pH1)]:
            e = d[f"{dom}_{key}"]
            ax.loglog(h, e, "o-", color=color, lw=2, label=label)
            ref = e[-1] * (h / h[-1]) ** p          # theory slope through the finest point
            frac = {2: "2", 1: "1", 4 / 3: "4/3", 2 / 3: "2/3"}[p]
            ax.loglog(h, ref, "--", color=color, alpha=0.6, label=f"slope {frac} (theory)")
        ax.set_title(name)
        ax.set_xlabel("Mesh size h")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Error")
    fig.suptitle("How fast the error shrinks when the mesh is refined")
    fig.savefig(FIGURES / "convergence.png", dpi=170)
    plt.close(fig)


def refinement_gif():
    """Short wide GIF for the website: both domains on finer and finer meshes."""
    frames = []
    for n in [4, 8, 16, 32]:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=80, constrained_layout=True)
        draw(axes[0], mesh_square, n, edges=True, vmax=0.07)
        draw(axes[1], mesh_lshape, n, edges=True, vmax=0.036)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"mesh {n} x {n}", fontsize=14)
        fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        frames.append(rgb.quantize(colors=64, dither=Image.Dither.NONE))
        plt.close(fig)
    frames[0].save(FIGURES / "mesh-refinement.gif", save_all=True,
                   append_images=frames[1:], duration=1200, loop=0, optimize=True)


def main():
    FIGURES.mkdir(exist_ok=True)
    solutions()
    convergence()
    refinement_gif()
    print(f"Figures saved in {FIGURES}")


if __name__ == "__main__":
    main()

"""Shared presentation rules. No orange/blue palette or overlapping line encodings."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm
import numpy as np
import pandas as pd

PALETTE = {"plum": "#713C6B", "green": "#26745C", "rose": "#B9667A",
           "ink": "#302C35", "muted": "#777078", "grid": "#E8E3E7", "paper": "#FFFFFF"}
CORPUS_LABELS = {"provo": "Provo · FFD", "natural_stories": "Natural Stories · SPR"}
SEQUENTIAL = LinearSegmentedColormap.from_list("saert_plum", ["#FAF7FA", "#D8BDCF", PALETTE["plum"]])
IMPROVEMENT = LinearSegmentedColormap.from_list("saert_improvement", [PALETTE["plum"], "#FBF9FA", PALETTE["green"]])
ERROR = IMPROVEMENT.reversed()


def set_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
        "axes.prop_cycle": plt.cycler(color=[PALETTE["plum"], PALETTE["green"], PALETTE["rose"], PALETTE["muted"]]),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["pdf", "svg", "png"]:
        path = directory / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    return paths


def annotated_heatmap(ax, frame, *, fmt=".2f", signed=False, errors=False, norm=None, title="", colorbar_label=""):
    values = frame.to_numpy(float)
    finite = values[np.isfinite(values)]
    limit = max(float(np.abs(finite).max()), 1e-6) if len(finite) else 1.
    norm = norm or (Normalize(-limit, limit) if signed else Normalize(0, limit))
    cmap = ERROR if errors and signed else IMPROVEMENT if signed else SEQUENTIAL
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(frame.columns)), frame.columns)
    ax.set_yticks(np.arange(len(frame)), frame.index)
    ax.tick_params(length=0, pad=9)
    for (i, j), value in np.ndenumerate(values):
        label = format(value, fmt) if np.isfinite(value) else "—"
        rgba = cmap(norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
        luminance = np.dot(rgba[:3], [.2126, .7152, .0722])
        ax.text(j, i, label, ha="center", va="center", fontsize=9,
                color="white" if luminance < .48 else PALETTE["ink"])
    ax.set_xticks(np.arange(-.5, len(frame.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(frame), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=16)
    if colorbar_label:
        ax.figure.colorbar(im, ax=ax, shrink=.7, pad=.03).set_label(colorbar_label)
    return im


def focus_table(frame, *, important=(), formats=None, signed_columns=(), caption=""):
    """Curated columns first; color only quantitative effects with a defined zero."""
    table = frame.style.hide(axis="index").format(formats or {}, na_rep="—")
    table = table.set_properties(**{"padding": "8px 12px", "color": PALETTE["ink"], "font-size": "13px"})
    for col in important:
        table = table.set_properties(subset=[col], **{"font-weight": "bold", "background-color": "#F1E8EF"})
    if signed_columns:
        extent = max(float(frame[list(signed_columns)].abs().max().max()), 1e-6)
        table = table.background_gradient(cmap=IMPROVEMENT, subset=list(signed_columns), axis=None,
                                          vmin=-extent, vmax=extent)
    table = table.set_table_styles([{"selector": "th", "props": [("text-align", "left"), ("border-bottom", "2px solid #713C6B")]},
                                   {"selector": "caption", "props": [("caption-side", "bottom"), ("text-align", "left"), ("padding", "10px")]}])
    return table.set_caption(caption)


def display_details(title, frame):
    from IPython.display import HTML, display
    import html
    display(HTML(f"<details><summary>{html.escape(title)}</summary>{frame.to_html(index=False, float_format=lambda x: f'{x:.3f}')}</details>"))

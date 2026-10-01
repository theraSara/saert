from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm  # noqa: F401 (re-exported)
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ── Palette ────────────────────────────────────────────────────────────────────
GREEN = "#009E73"   # Okabe-Ito bluish green  -> Provo
PINK = "#CC79A7"    # Okabe-Ito reddish purple -> Natural Stories
INK, MUTED, LIGHT, GRID = "#222222", "#6E6E6E", "#BDBDBD", "#E8E8E8"

CORPORA = ["provo", "natural_stories"]
CORPUS_COLORS = {"provo": GREEN, "natural_stories": PINK}
CORPUS_MARKERS = {"provo": "o", "natural_stories": "s"}
CORPUS_LABELS = {"provo": "Provo · FFD", "natural_stories": "Natural Stories · SPR"}
STATE_LABELS = {"post": "After word", "prefix": "Before word"}
PRIMARY_HOOKS = {"provo": "L01", "natural_stories": "L08"}
HOOK_ORDER = [f"L{i:02d}" for i in range(1, 13)] + ["final_resid_post"]
HOOK_TICKS = [str(i) for i in range(1, 13)] + ["final"]

# Representation encodings: no extra hues. SAE = filled corpus colour; dense = open marker,
# dashed line; surprisal = grey. Combined models reuse the feature style with a heavier edge.
REP_LABELS = {"surprisal": "Surprisal", "dense": "Dense state", "sae": "SAE features",
              "surprisal+dense": "Surprisal + dense", "surprisal+sae": "Surprisal + SAE"}

# Kept for existing scripts (baseline_diagnostics, sae_diagnostics, sae_regression_diagnostics).
PALETTE = {"green": GREEN, "pink": PINK, "rose": PINK, "plum": INK, "ink": INK,
           "muted": MUTED, "grid": GRID, "paper": "#FFFFFF"}
SEQUENTIAL = LinearSegmentedColormap.from_list("saert_grey", ["#FAFAFA", "#BDBDBD", "#3A3A3A"])
IMPROVEMENT = LinearSegmentedColormap.from_list("saert_improvement", [PINK, "#FAFAFA", GREEN])
ERROR = IMPROVEMENT.reversed()


# ── Project paths ──────────────────────────────────────────────────────────────
def project_root():
    """The saert/ directory, whether called from saert/ or saert/notebooks/."""
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise FileNotFoundError("Run from the saert project or its notebooks/ folder")


def paths(root=None):
    root = Path(root or project_root())
    p = {"root": root, "results": root / "results", "figures": root / "results" / "figures",
         "baseline": root / "results" / "baseline_bos", "sae": root / "results" / "sae_bos",
         "regression": root / "results" / "sae_regression_v2", "residual": root / "results" / "residual_rt",
         "interpretation": root / "results" / "feature_interpretation",
         "scaling": root / "results" / "scaling_sensitivity_v1", "config": root / "configs" / "sae_regression.json"}
    p["figures"].mkdir(parents=True, exist_ok=True)
    return p


def bootstrap_settings(root=None):
    cfg = paths(root)["config"]
    if cfg.exists():
        c = json.loads(cfg.read_text())
        return int(c.get("bootstrap", 2000)), int(c.get("seed", 42))
    return 2000, 42


# ── Style ──────────────────────────────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=[GREEN, PINK, MUTED, INK]),
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def save_figure(fig, directory, name, formats=("pdf", "svg", "png")):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        path = directory / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        out.append(path)
    return out


def rep_style(rep, corpus):
    """Marker/line style for a representation, coloured by corpus."""
    color = CORPUS_COLORS[corpus]
    if rep == "surprisal":
        return dict(color=MUTED, marker="D", mfc=MUTED, ls=":", lw=1.4)
    filled = rep.endswith("sae")
    return dict(color=color, marker=CORPUS_MARKERS[corpus], mfc=color if filled else "white",
                ls="-" if filled else "--", lw=2.0 if rep.startswith("surprisal+") else 1.6)


def corpus_legend(ax, corpora=CORPORA, **kw):
    handles = [Line2D([], [], color=CORPUS_COLORS[c], marker=CORPUS_MARKERS[c], ls="", ms=7,
                      label=CORPUS_LABELS[c]) for c in corpora]
    return ax.legend(handles=handles, **kw)


def rep_legend(ax, reps=("sae", "dense"), corpus="provo", **kw):
    handles = []
    for r in reps:
        s = rep_style(r, corpus)
        c = MUTED if r != "surprisal" else s["color"]
        handles.append(Line2D([], [], color=c, marker=s["marker"] if r == "surprisal" else "o",
                              mfc=(c if (r == "surprisal" or r.endswith("sae")) else "white"),
                              ls=s["ls"], lw=s["lw"], label=REP_LABELS[r]))
    return ax.legend(handles=handles, **kw)


# ── Plot primitives ────────────────────────────────────────────────────────────
def forest(ax, rows, *, xlabel="ΔR² (percentage points)", annotate=True):
    """rows: list of dicts with label, est, lo, hi, corpus, rep. Top-to-bottom order."""
    for i, r in enumerate(rows):
        s = rep_style(r["rep"], r["corpus"])
        ax.hlines(i, r["lo"], r["hi"], color=s["color"], lw=2.2)
        ax.plot(r["est"], i, marker=s["marker"], color=s["color"], mfc=s["mfc"], ms=8, mew=1.6, ls="")
        if annotate:
            ax.annotate(f"{r['est']:+.2f}", (r["hi"], i), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel(xlabel)
    return ax


def layer_profile(ax, frame, corpus, *, value="est", lo="lo", hi="hi", clip=None):
    """frame: columns label, rep, est, lo, hi for one corpus/state. Lines over HOOK_ORDER."""
    x = np.arange(len(HOOK_ORDER))
    for rep, g in frame.groupby("rep", sort=False):
        g = g.set_index("label").reindex(HOOK_ORDER)
        s = rep_style(rep, corpus)
        y = g[value].to_numpy(float)
        ylo, yhi = g[lo].to_numpy(float), g[hi].to_numpy(float)
        if clip is not None:
            off = y < clip[0]
            y, ylo, yhi = [np.clip(v, *clip) for v in (y, ylo, yhi)]
            for xi in x[off]:
                ax.annotate("off scale", (xi, clip[0]), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=7, color=MUTED)
        ax.fill_between(x, ylo, yhi, color=s["color"], alpha=0.12 if s["mfc"] != "white" else 0.06, lw=0)
        ax.plot(x, y, color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"], mfc=s["mfc"], ms=5,
                label=REP_LABELS[rep])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, HOOK_TICKS, fontsize=8)
    ax.set_xlabel("Layer (residual stream before block n; final = after block 12)")
    ax.grid(axis="y", color=GRID)
    return ax


# ── Backward-compatible helpers used by existing scripts ───────────────────────
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
        ax.text(j, i, label, ha="center", va="center", fontsize=9, color="white" if luminance < .48 else INK)
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


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, int(round(v)))):02x}" for v in rgb)


def _blend_with_white(hex_color, strength):
    strength = float(np.clip(strength, 0, 1))
    rgb = np.array(_hex_to_rgb(hex_color), dtype=float)
    white = np.array([255, 255, 255], dtype=float)
    return _rgb_to_hex((1 - strength) * white + strength * rgb)


def _dataset_color(value):
    label = str(value).lower()
    if "provo" in label:
        return GREEN
    if "natural" in label:
        return PINK
    return MUTED


def focus_table(frame, *, important=(), formats=None, signed_columns=(), caption=""):
    table = frame.style.hide(axis="index").format(formats or {}, na_rep="—")
    table = table.set_properties(**{"padding": "6px 10px", "color": INK, "font-size": "13px"})
    for col in important:
        table = table.set_properties(subset=[col], **{"font-weight": "bold"})
    if signed_columns:
        signed_columns = [c for c in signed_columns if c in frame.columns]
        if "Dataset" in frame.columns and signed_columns:
            maxima = {c: max(float(pd.to_numeric(frame[c], errors="coerce").clip(lower=0).max()), 1e-6)
                      for c in signed_columns}

            def corpus_gradient(row):
                styles = pd.Series("", index=row.index)
                base = _dataset_color(row.get("Dataset", ""))
                for col in signed_columns:
                    value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                    if pd.isna(value):
                        continue
                    if value > 0:
                        strength = 0.12 + 0.72 * min(value / maxima[col], 1.0)
                        bg = _blend_with_white(base, strength)
                    elif value < 0:
                        strength = 0.08 + 0.30 * min(abs(value) / max(abs(value), maxima[col]), 1.0)
                        bg = _blend_with_white(LIGHT, strength)
                    else:
                        bg = "#FFFFFF"
                    text = "#FFFFFF" if value > 0 and strength > 0.58 else INK
                    styles[col] = f"background-color: {bg}; color: {text}"
                return styles

            table = table.apply(corpus_gradient, axis=1)
        elif signed_columns:
            extent = max(float(frame[list(signed_columns)].abs().max().max()), 1e-6)
            table = table.background_gradient(cmap=IMPROVEMENT, subset=list(signed_columns), axis=None,
                                              vmin=-extent, vmax=extent)
    table = table.set_table_styles([{"selector": "th", "props": [("text-align", "left"), ("border-bottom", f"2px solid {INK}")]},
                                    {"selector": "caption", "props": [("caption-side", "bottom"), ("text-align", "left")]}])
    return table.set_caption(caption)


def display_details(title, frame):
    from IPython.display import HTML, display
    import html
    display(HTML(f"<details><summary>{html.escape(title)}</summary>"
                 f"{frame.to_html(index=False, float_format=lambda x: f'{x:.3f}')}</details>"))


def ci_text(est, lo, hi):
    return f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def shorten(text, n=48):
    text = str(text) if pd.notna(text) else ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"

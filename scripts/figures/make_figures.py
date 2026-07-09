"""Generate all figures used in the paper.

The data-bearing figures (4-9) are produced directly from the **real** Merced
target-basin run in ``results/`` (walk_forward.parquet, walk_forward_metrics.json,
zero_shot_metrics.json, min_data_sensitivity.json, walk_forward_warnings.csv,
shap_global_importance.csv).  Figures 1-3 are schematics.  The previous
synthetic seasonal-SHAP figure (fig10) is intentionally omitted: no per-timestep
SHAP result has been computed for the real basin, so it cannot be drawn from a
real file (see paper Limitations / Future work).

Outputs go to ``figures/`` as PNGs at 300 dpi.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results" / "smoke"   # synthetic smoke run (no longer plotted)
RESULTS_REAL = ROOT / "results"         # real Merced 11264500 run

# Paper colour palette (Nexus + chart sequence)
PRIMARY = "#01696F"
PRIMARY_DARK = "#0C4E54"
ACCENT = "#A84B2F"
GOLD = "#FFC553"
INK = "#28251D"
MUTED = "#7A7974"
FAINT = "#D4D1CA"
BG = "#F7F6F2"
SURFACE = "#FBFBF9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})


# ---------------------------------------------------------------------------
# Real-result loaders
# ---------------------------------------------------------------------------
def _wf_metrics():
    return json.loads((RESULTS_REAL / "walk_forward_metrics.json").read_text())


def _zero_shot():
    return json.loads((RESULTS_REAL / "zero_shot_metrics.json").read_text())


def _min_data():
    return json.loads((RESULTS_REAL / "min_data_sensitivity.json").read_text())


def _walk_forward_df():
    return pd.read_parquet(RESULTS_REAL / "walk_forward.parquet")


def _nse(o, p):
    return 1 - np.sum((p - o) ** 2) / np.sum((o - np.mean(o)) ** 2)


def _kge(o, p):
    r = np.corrcoef(o, p)[0, 1]
    a = np.std(p) / np.std(o)
    b = np.mean(p) / np.mean(o)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


# ---------------------------------------------------------------------------
# Figure 1 — Framework architecture
# ---------------------------------------------------------------------------
def fig_framework_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color=SURFACE, edge=PRIMARY, lw=1.4):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                              linewidth=lw, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), label,
                ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                    ha="center", va="center", fontsize=8.5, color=MUTED)

    def arrow(x1, y1, x2, y2, color=PRIMARY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.4, color=color))

    # Phase headers
    for i, (xc, label) in enumerate([(1.7, "PHASE 1 · PRE-TRAIN"),
                                     (5.5, "PHASE 2 · FINE-TUNE"),
                                     (9.3, "PHASE 3 · WALK-FORWARD")]):
        ax.text(xc, 5.05, label, ha="center", fontsize=9.5,
                fontweight="bold", color=PRIMARY)
        ax.add_line(Line2D([xc - 1.4, xc + 1.4], [4.85, 4.85],
                           color=PRIMARY, linewidth=1.5))

    # Phase 1
    box(0.4, 3.6, 2.6, 1.0, "CAMELS-US donors", "199-basin subset · 30+ yrs", color=BG)
    box(0.4, 2.2, 2.6, 1.0, "EA-LSTM", "256 hidden · static gate", color=SURFACE)
    box(0.4, 0.8, 2.6, 1.0, "Pre-trained\nweights θ_pre", color=GOLD, edge=ACCENT)
    arrow(1.7, 3.55, 1.7, 3.25)
    arrow(1.7, 2.15, 1.7, 1.85)

    # Phase 2
    box(4.2, 3.6, 2.6, 1.0, "Merced 11264500", "2-yr warmup (data-scarce)", color=BG)
    box(4.2, 2.2, 2.6, 1.0, "Conservative FT", "freeze LSTM, head only", color=SURFACE)
    box(4.2, 0.8, 2.6, 1.0, "Progressive FT (opt.)", "unfreeze last 25% · diff. LR", color=SURFACE)
    arrow(3.0, 3.0, 4.2, 3.0, color=ACCENT)
    arrow(5.5, 3.55, 5.5, 3.25)
    arrow(5.5, 2.15, 5.5, 1.85)

    # Phase 3
    box(8.0, 3.6, 2.6, 1.0, "Rolling-origin loop", "expand window · ~90-day refit", color=BG)
    box(8.0, 2.2, 2.6, 1.0, "Probabilistic warning", "Q95/Q99 · 1/3/7-day lead", color=SURFACE)
    box(8.0, 0.8, 2.6, 1.0, "SHAP attribution", "global importance", color=SURFACE)
    arrow(6.8, 3.0, 8.0, 3.0, color=ACCENT)
    arrow(9.3, 3.55, 9.3, 3.25)
    arrow(9.3, 2.15, 9.3, 1.85)

    # RFA underlay
    box(0.4, 0.05, 10.2, 0.5,
        "Regional Frequency Analysis · Q5/Q95/Q99 from long CAMELS record",
        color="#EAF1F0", edge=PRIMARY)

    plt.savefig(FIG / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Walk-forward schematic (real 2009-2014 windows)
# ---------------------------------------------------------------------------
def fig_walk_forward_schematic():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    years = np.arange(2009, 2016)
    ax.set_xlim(2008.7, 2015.3)
    ax.set_ylim(-0.4, 6.4)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # warmup 2009-2010, evaluation 2011-2014
    rounds = [
        (2009, 2011, 2011.25),
        (2009, 2011.75, 2012.0),
        (2009, 2012.5, 2012.75),
        (2009, 2013, 2013.5),
        (2009, 2014, 2014.5),
        (2009, 2014.75, 2015.0),
    ]
    for i, (s, t, e) in enumerate(rounds):
        y = 5 - i
        ax.add_patch(Rectangle((s, y - 0.2), t - s, 0.4,
                               facecolor=PRIMARY, edgecolor="none", alpha=0.25))
        ax.add_patch(Rectangle((t, y - 0.2), e - t, 0.4,
                               facecolor=ACCENT, edgecolor="none", alpha=0.7))
        ax.text(2008.6, y, f"Round {i+1}", ha="right", va="center",
                fontsize=8.5, color=MUTED)

    ax.text(2010, 5.7, "Training window (expands)", ha="center",
            color=PRIMARY_DARK, fontsize=9.5, fontweight="bold")
    ax.text(2013.5, 5.7, "Evaluation window (next chunk)", ha="center",
            color=ACCENT, fontsize=9.5, fontweight="bold")

    ax.axvline(2011, color=INK, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2011, -0.3, "warmup ends", ha="center", color=MUTED, fontsize=8)
    ax.set_title("Walk-forward (rolling-origin) backtest: 2-yr warmup (2009-10) "
                 "→ 4-yr evaluation (2011-14)", loc="left")
    plt.tight_layout()
    plt.savefig(FIG / "fig2_walk_forward.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Unfreezing strategies (schematic, unchanged)
# ---------------------------------------------------------------------------
def fig_unfreezing_strategies():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    layers = ["LSTM\nweights\n(low-level)",
              "LSTM\nweights\n(mid-level)",
              "LSTM\nweights\n(high-level)",
              "Dense\nhead"]
    titles = ["Approach A · Conservative",
              "Approach B · Progressive Unfreezing"]
    a_colors = [FAINT, FAINT, FAINT, PRIMARY]
    b_colors = [FAINT, FAINT, "#7AB6BB", PRIMARY]
    a_labels = ["frozen", "frozen", "frozen", "trained · LR=1e-3"]
    b_labels = ["frozen", "frozen", "trained · LR=1e-5", "trained · LR=1e-3"]

    for ax, title, colors, labels in [(axes[0], titles[0], a_colors, a_labels),
                                      (axes[1], titles[1], b_colors, b_labels)]:
        ax.set_xlim(0, 4); ax.set_ylim(0, 2.5); ax.axis("off")
        ax.set_title(title, loc="left")
        for j, (lay, c, lab) in enumerate(zip(layers, colors, labels)):
            r = FancyBboxPatch((0.05 + 1.0 * j, 0.6), 0.9, 1.2,
                               boxstyle="round,pad=0.04", linewidth=1.2,
                               edgecolor=INK, facecolor=c)
            ax.add_patch(r)
            ax.text(0.5 + 1.0 * j, 1.2, lay, ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=("white" if c == PRIMARY else INK))
            ax.text(0.5 + 1.0 * j, 0.4, lab, ha="center", va="center",
                    fontsize=8, color=MUTED, fontstyle="italic")

    plt.tight_layout()
    plt.savefig(FIG / "fig3_unfreezing.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — RFA thresholds on the real Merced record
# ---------------------------------------------------------------------------
def fig_rfa_thresholds():
    th = _wf_metrics()["thresholds"]
    df = _walk_forward_df()
    flow = df["observed"].dropna()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(flow, bins=60, color=PRIMARY, alpha=0.55, edgecolor="white")
    for q, c, lab, fmt in [("q5", ACCENT, "Q5 (drought)", "{:.3f}"),
                           ("q95", GOLD, "Q95 (flood)", "{:.2f}"),
                           ("q99", PRIMARY_DARK, "Q99 (extreme)", "{:.2f}")]:
        ax.axvline(th[q], color=c, linewidth=2,
                   label=f"{lab} = {fmt.format(th[q])} mm/d")
    ax.set_xlabel("Daily streamflow (mm/day)")
    ax.set_ylabel("Frequency")
    ax.set_title("Regional Frequency Analysis thresholds · Merced 11264500 (real)",
                 loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_rfa_thresholds.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5 — Real walk-forward hydrograph
# ---------------------------------------------------------------------------
def fig_hydrograph():
    df = _walk_forward_df()
    c = _wf_metrics()["continuous"]
    o = df["observed"].to_numpy()
    p = df["predicted"].to_numpy()
    thr = np.quantile(o, 0.95)
    mk = o >= thr
    underest = 100 * (1 - p[mk].mean() / o[mk].mean())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["observed"], color=INK, linewidth=1.0, label="Observed")
    ax.plot(df.index, df["predicted"], color=PRIMARY, linewidth=1.0,
            alpha=0.85, label="Walk-forward prediction")
    ax.fill_between(df.index, df["predicted"], df["observed"],
                    where=df["predicted"] < df["observed"],
                    color=ACCENT, alpha=0.2, label="Underprediction")
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_title(
        f"Walk-forward hydrograph · Merced 11264500 · NSE={c['NSE']:.2f} "
        f"· KGE={c['KGE']:.2f} · PBIAS={c['PBIAS']:.2f}% "
        f"(peaks underestimated ~{underest:.0f}%)",
        loc="left",
    )
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG / "fig5_hydrograph.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6 — Performance comparison (real variants; missing baselines flagged)
# ---------------------------------------------------------------------------
def fig_performance_comparison():
    zs = _zero_shot()
    wf = _wf_metrics()["continuous"]
    md = _min_data()
    md24 = next(r for r in md["results"] if r["warmup_months"] == 24)

    # Baseline evaluations (scripts/paper_supplement_analysis.py); when the
    # file is absent the two variants render as hatched "rerun pending" bars.
    try:
        be = json.loads((RESULTS_REAL / "baseline_eval_metrics.json").read_text())
    except Exception:
        be = {}
    lb = be.get("local_baseline")
    pg = be.get("finetune_progressive")

    try:
        wfb = json.loads(
            (RESULTS_REAL / "walk_forward_progressive_metrics.json").read_text()
        )["continuous"]
    except Exception:
        wfb = None

    # (label, NSE, KGE) — None marks a pending (not-yet-run) variant
    rows = [
        ("Local baseline\n(from scratch)",
         lb["NSE"] if lb else None, lb["KGE"] if lb else None),
        ("Zero-shot\ntransfer", zs["NSE"], zs["KGE"]),
        ("Min-data\n24-mo FT", md24["NSE"], md24["KGE"]),
        ("Progressive\nfine-tune (static)",
         pg["NSE"] if pg else None, pg["KGE"] if pg else None),
    ]
    if wfb:
        rows.append(("Progressive\n+ refits", wfb["NSE"], wfb["KGE"]))
    rows.append(("Conservative\n+ refits", wf["NSE"], wf["KGE"]))
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.4))

    for i, (_, nse, kge) in enumerate(rows):
        if nse is None:
            # pending variant — hatched placeholder + annotation
            ax.bar(i - width / 2, 0.0, width, color="none",
                   edgecolor=MUTED, hatch="///", linewidth=1.0)
            ax.bar(i + width / 2, 0.0, width, color="none",
                   edgecolor=MUTED, hatch="///", linewidth=1.0)
            ax.text(i, 0.04, "rerun\npending", ha="center", va="bottom",
                    fontsize=8.5, color=MUTED, fontstyle="italic")
            continue
        # Deeply negative scores (from-scratch collapse) are clipped for
        # display; the printed label always shows the true value.
        clip_floor = -0.42
        disp_nse = max(nse, clip_floor)
        disp_kge = max(kge, clip_floor)
        b1 = ax.bar(i - width / 2, disp_nse, width,
                    color=PRIMARY, edgecolor="white",
                    label="NSE" if i == 0 else None)
        b2 = ax.bar(i + width / 2, disp_kge, width,
                    color=ACCENT, edgecolor="white",
                    label="KGE" if i == 0 else None)
        for r, true_v in zip(list(b1) + list(b2), [nse, kge]):
            v = r.get_height()
            if true_v >= 0:
                ax.text(r.get_x() + r.get_width() / 2, v + 0.015,
                        f"{true_v:.2f}", ha="center", va="bottom",
                        fontsize=9, color=INK)
            else:
                ax.text(r.get_x() + r.get_width() / 2, v - 0.02,
                        f"{true_v:.2f}", ha="center", va="top",
                        fontsize=9, color=INK)

    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(-0.62, 1.05)
    ax.set_ylabel("Skill score")
    ax.set_title("Continuous skill on Merced 11264500 · fine-tuning rescues "
                 "an unskillful zero-shot transfer", loc="left")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG / "fig6_perf_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7 — Reliability diagram (computed from real warnings CSV)
# ---------------------------------------------------------------------------
def fig_reliability():
    w = pd.read_csv(RESULTS_REAL / "walk_forward_warnings.csv", index_col=0)
    y = w["flood_q95_lead3d"].to_numpy()
    prob = w["flood_q95_lead3d_prob"].to_numpy()
    edges = np.linspace(0, 1, 11)
    centers, freqs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if m.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        freqs.append(y[m].mean())
        counts.append(m.sum())
    centers = np.array(centers); freqs = np.array(freqs); counts = np.array(counts)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", label="Perfect")
    sizes = 30 + 220 * counts / max(counts.max(), 1)
    ax.scatter(centers, freqs, s=sizes, color=PRIMARY, edgecolor="white",
               alpha=0.85, label="Walk-forward (lead 3 d)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability · Q95 flood warning, 3-day lead (real)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "fig7_reliability.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8 — AUC / Brier by lead time (real early-warning metrics)
# ---------------------------------------------------------------------------
def fig_auc_by_lead():
    ew = _wf_metrics()["early_warning"]
    targets = {
        "Flood Q95": ("flood_q95", PRIMARY),
        "Flood Q99": ("flood_q99", PRIMARY_DARK),
        "Drought Q5": ("drought_q5", ACCENT),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for name, (prefix, color) in targets.items():
        leads, aucs, briers = [], [], []
        for lead in (1, 3, 7):
            k = f"{prefix}_lead{lead}d"
            if k in ew:
                leads.append(lead)
                aucs.append(ew[k]["AUC"])
                briers.append(ew[k]["Brier"])
        axes[0].plot(leads, aucs, "o-", color=color, linewidth=2,
                     markersize=7, label=name)
        axes[1].plot(leads, briers, "o-", color=color, linewidth=2,
                     markersize=7, label=name)

    axes[0].axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    axes[0].set_xlabel("Warning lead time (days)")
    axes[0].set_ylabel("AUC-ROC (higher = better)")
    axes[0].set_title("Early-warning ranking skill", loc="left")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].set_xticks([1, 3, 7])
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)

    axes[1].set_xlabel("Warning lead time (days)")
    axes[1].set_ylabel("Brier score (lower = better)")
    axes[1].set_title("Probabilistic accuracy", loc="left")
    axes[1].set_xticks([1, 3, 7])
    axes[1].legend(frameon=False, loc="upper left", fontsize=8.5)

    fig.suptitle("Flood warning is validated (AUC ≈ 0.98); drought warning "
                 "is weak and poorly calibrated", fontsize=10.5,
                 fontweight="bold", color=INK, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(FIG / "fig8_auc_lead.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 9 — Real global SHAP importance
# ---------------------------------------------------------------------------
def fig_shap_importance():
    df = pd.read_csv(RESULTS_REAL / "shap_global_importance.csv", index_col=0)
    col = df.columns[0]
    pretty = {
        "dyn_dayl(s)": "Day length",
        "dyn_srad(W/m2)": "Shortwave radiation",
        "dyn_tmax(C)": "Air temperature (max)",
        "dyn_vp(Pa)": "Vapor pressure",
        "dyn_tmin(C)": "Air temperature (min)",
        "dyn_prcp(mm/day)": "Precipitation",
    }
    dyn = df[df.index.str.startswith("dyn_")].copy()
    dyn = dyn.sort_values(col)
    names = [pretty.get(i, i) for i in dyn.index]
    vals = dyn[col].to_numpy()
    n_static_zero = int((df.index.str.startswith("static_") & (df[col] == 0)).sum())

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(names, vals, color=PRIMARY, edgecolor="white")
    for yi, v in enumerate(vals):
        ax.text(v + 0.0008, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title("Global SHAP importance · Merced 11264500 (real)", loc="left")
    ax.text(0.98, 0.06,
            f"All {n_static_zero} static attributes: mean|SHAP| = 0.000\n"
            "(single-basin explanation — statics do not vary)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=MUTED, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE,
                      edgecolor=FAINT))
    plt.tight_layout()
    plt.savefig(FIG / "fig9_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    fig_framework_architecture()
    fig_walk_forward_schematic()
    fig_unfreezing_strategies()
    fig_rfa_thresholds()
    fig_hydrograph()
    fig_performance_comparison()
    fig_reliability()
    fig_auc_by_lead()
    fig_shap_importance()
    # Remove any stale synthetic temporal-SHAP figure (no real source exists).
    stale = FIG / "fig10_shap_temporal.png"
    if stale.exists():
        stale.unlink()
    print(f"Generated {len(list(FIG.glob('*.png')))} figures in {FIG}")


if __name__ == "__main__":
    main()

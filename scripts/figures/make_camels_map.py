"""Map of all CAMELS-US catchments with the multi-target study design.

Shows: all 671 gauges colored by snow fraction (the regime axis that matters
most here), the 7 held-out target basins as labeled stars, and the donors
removed by the 50 km exclusion buffers as grey crosses.

Run: python scripts/make_camels_map.py
Out: figures/fig_camels_map.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hydro_tl_ews.data.camels import CamelsDataset
from stages.pretrain_stage import exclude_targets_and_buffer

TARGETS = {
    "11264500": ("Merced R, CA\nSierra snowmelt", (-8, 10)),
    "09107000": ("Taylor R, CO\nRockies snowmelt", (6, -14)),
    "14222500": ("EF Lewis R, WA\nPNW maritime rain", (6, 4)),
    "02128000": ("Little R, NC\nSE humid rain", (6, -12)),
    "01544500": ("Kettle Ck, PA\nNE mixed", (6, 6)),
    "11224500": ("Los Gatos Ck, CA\nsemi-arid ephemeral", (6, -18)),
    "05507600": ("Lick Ck, MO\nplains continental", (6, 8)),
}
BUFFER_KM = 50


def main() -> None:
    attrs = CamelsDataset(ROOT / "data").load_attributes()
    donors = exclude_targets_and_buffer(
        attrs, list(attrs.index), list(TARGETS), BUFFER_KM)
    buffered_out = [b for b in attrs.index
                    if b not in donors and b not in TARGETS]

    fig, ax = plt.subplots(figsize=(12, 7.2))
    d = attrs.loc[donors]
    sc = ax.scatter(d["gauge_lon"], d["gauge_lat"], c=d["frac_snow"],
                    cmap="viridis", s=16, alpha=0.75, linewidths=0,
                    label=f"donor basins (n={len(donors)})")
    b = attrs.loc[buffered_out]
    ax.scatter(b["gauge_lon"], b["gauge_lat"], marker="x", c="#999999",
               s=34, linewidths=1.4,
               label=f"excluded by {BUFFER_KM} km buffer (n={len(buffered_out)})")
    t = attrs.loc[list(TARGETS)]
    ax.scatter(t["gauge_lon"], t["gauge_lat"], marker="*", c="#d62728",
               s=340, edgecolors="black", linewidths=0.8, zorder=5,
               label=f"held-out targets (n={len(TARGETS)})")
    for bid, (label, (dx, dy)) in TARGETS.items():
        ax.annotate(label,
                    (attrs.loc[bid, "gauge_lon"], attrs.loc[bid, "gauge_lat"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, fontweight="bold", color="#7f1d1d",
                    ha="left" if dx > 0 else "right", zorder=6)

    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label("snow fraction of precipitation (frac_snow)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("CAMELS-US catchments (671): pretrain donors, exclusion buffers, "
                 "and the 7 held-out target basins")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.set_aspect(1.25)  # rough CONUS aspect for plate carree
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out = ROOT / "figures" / "fig_camels_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

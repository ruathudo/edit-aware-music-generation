"""
Figure 1 of the paper: localization and gated improvement against how far
each draft source moves a melody off the data distribution.

    python scripts/figure1.py            # writes figures/figure1.{pdf,png}

Reads only committed results, so it needs no data or checkpoints:

  x         median dNLL per unit of change (Table 2)  results/judge/plausibility_table.json,
                                                      results/judge/severity_curve.json
  panel a   AUROC, two-sided judge (untrained), for   results/judge/severity_curve.json
            D1, D3 and six generator settings (three
            temperatures, three undertrained
            checkpoints), MTC and Essen
  panel b   gated vs copy-input, % (5 seeds, +/- sd)  results/refiner/headline_s34/five_seed_summary.json

Panel b is MTC and the three headline sources only: gated gain needs a
trained refiner per condition, and only the headline sources have one. Two
panels rather than two y-axes: the measures have different units, and a shared
x is what carries the comparison.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

# (key in plausibility_table, key in zero_shot, cell in five_seed_summary, label)
SOURCES = [
    ("D2-infill", "D2-infill", "mtc_d2infill", "D2 generated"),
    ("D3", "D3", "mtc_d3", "D3 rule-based"),
    ("D1", "D1", "mtc_d1", "D1 random"),
]

BLUE, ORANGE = "#2a78d6", "#eb6834"     # categorical slots 1-2, validated all-pairs
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a3a29c"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    plaus = {r["source"]: r for r in load("results/judge/plausibility_table.json")["rows"]}
    severity = load("results/judge/severity_curve.json")["rows"]
    five = {r["cell"]: r for r in load("results/refiner/headline_s34/five_seed_summary.json")}

    x = [plaus[p]["per_unit_median"] for p, _, _, _ in SOURCES]
    gain = [five[c]["gated"] for _, _, c, _ in SOURCES]
    gain_sd = [five[c]["gated_sd"] for _, _, c, _ in SOURCES]
    pos = [five[c]["pos"] for _, _, c, _ in SOURCES]
    human = plaus["D4 human variants"]["per_unit_median"]

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.5), sharex=True)

    for ax in (ax_a, ax_b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvline(human, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
        ax.set_xlim(-0.1, 4.6)

    # (a) localization by the untrained two-sided judge, every condition
    ax_a.axhline(0.5, color=INK2, linewidth=0.8)
    ax_a.text(4.55, 0.505, "chance", ha="right", va="bottom", color=INK2, fontsize=7)
    for corpus, color, marker, ms in (("MTC", BLUE, "o", 6), ("Essen", ORANGE, "s", 5)):
        rows = [r for r in severity if r["corpus"] == corpus]
        gen = sorted((r for r in rows if r["condition"].startswith("D2")),
                     key=lambda r: r["per_unit_median"])
        ax_a.plot([r["per_unit_median"] for r in gen], [r["judge_auroc"] for r in gen],
                  "-", color=color, linewidth=1, alpha=0.6, zorder=2)
        ax_a.plot([r["per_unit_median"] for r in rows], [r["judge_auroc"] for r in rows],
                  marker, ms=ms, color=color, mec="white", mew=0.8, linestyle="none",
                  label=corpus, zorder=3)
    labels = {"D1": ("D1 random", (0, 7), "center"), "D3": ("D3 rule-based", (0, -11), "center"),
              "D2 T=0.8": ("T=0.8", (5, -2), "left"), "D2 T=1.0": ("D2 (T=1.0)", (5, -2), "left"),
              "D2 T=1.2": ("T=1.2", (5, -2), "left"), "D2 ep10": ("ep10", (5, -2), "left"),
              "D2 ep5": ("ep5", (5, -2), "left"), "D2 ep2": ("ep2", (5, -2), "left")}
    for r in severity:
        if r["corpus"] == "MTC" and r["condition"] in labels:
            text, offset, ha = labels[r["condition"]]
            ax_a.annotate(text, (r["per_unit_median"], r["judge_auroc"]), xytext=offset,
                          textcoords="offset points", ha=ha, va="bottom" if offset[1] > 0 else "top",
                          color=INK, fontsize=6.5)
    ax_a.text(human + 0.08, 0.53, "human\nvariants", ha="left", va="bottom",
              color=INK2, fontsize=7)
    ax_a.set_ylim(0.45, 0.92)
    ax_a.set_ylabel("AUROC, changed notes")
    ax_a.set_title("(a) Can the changed notes be found?", loc="left", fontsize=8, color=INK)
    ax_a.legend(loc="lower right", frameon=False, fontsize=7, handletextpad=0.3,
                bbox_to_anchor=(1.0, 0.2))

    # (b) gated improvement
    ax_b.axhline(0, color=INK2, linewidth=0.8)
    ax_b.text(4.55, 0.3, "copy-input (draft unchanged)", ha="right", va="bottom",
              color=INK2, fontsize=7)
    ax_b.errorbar(x, gain, yerr=gain_sd, fmt="o", ms=7, color=BLUE, mec="white",
                  mew=1, elinewidth=1, capsize=0, zorder=3)
    for xi, g, p, (_, _, _, name) in zip(x, gain, pos, SOURCES):
        note = (f"{name}\n+{g:.2f}%, {p}/5 seeds" if g > 0
                else f"{name}\ngate never opens\n(abstains, 5/5 seeds)")
        ax_b.annotate(note, (xi, g), xytext=(0, 8), textcoords="offset points",
                      ha="center", va="bottom", color=INK, fontsize=7, linespacing=1.1)
    ax_b.set_ylim(-2, 14)
    ax_b.set_ylabel("CC gain over copy-input (%)")
    ax_b.set_title("(b) Does gated editing improve the draft?", loc="left",
                   fontsize=8, color=INK)

    fig.supxlabel("implausibility per unit of change (median ΔNLL ÷ fraction of notes changed)",
                  fontsize=8, color=INK)
    fig.tight_layout(w_pad=2.5)
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure1.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT / 'figure1.pdf'} and .png")


if __name__ == "__main__":
    main()

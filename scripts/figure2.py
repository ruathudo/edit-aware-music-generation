"""
Figure 2 of the paper: how well the changed notes are found, by draft source.

    python scripts/figure2.py      # writes figures/figure2.{pdf,png}

Reads only committed results, so it needs no data or checkpoints:

  AUROC of the untrained two-sided judge   results/judge/severity_curve.json
  for the changed notes: random and
  rule-based corruption, and the generator
  varied two ways (training length at T=1.0;
  sampling temperature when fully trained).
  MTC and Essen.

A dot plot with one row per draft condition rather than a scatter against
implausibility per unit: the row order carries the comparison (corruption vs. a
generator that gets better down each group) without an x-axis that needs
explaining. The fully trained generator at T=1.0 (D2) appears in both generator
groups, since it is the end point of one and the middle of the other.

An earlier version had a second panel with gated editing's gain per source; it
repeated Table IV and was dropped.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"     # categorical slots 1-2, validated (CVD dE 24.7)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"

# Rows, top to bottom: (group header or None, condition key, row label)
ROWS = [
    ("Corrupted", "D1", "Random (D1)"),
    (None, "D3", "Rule-based (D3)"),
    ("Generator: training", "D2 ep2", "2 epochs"),
    (None, "D2 ep5", "5 epochs"),
    (None, "D2 ep10", "10 epochs"),
    (None, "D2 T=1.0", "fully trained (D2)"),
    ("Generator: temperature", "D2 T=1.2", "T = 1.2"),
    (None, "D2 T=1.0", "T = 1.0 (D2)"),
    (None, "D2 T=0.8", "T = 0.8"),
]

def load(path):
    return json.loads((ROOT / path).read_text())


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT / name}.pdf and .png")


def draw_panel(ax):
    auroc = {(r["corpus"], r["condition"]): r["judge_auroc"]
             for r in load("results/judge/severity_curve.json")["rows"]}

    # y positions: one unit per row, plus a gap and a header line before each group
    ys, headers, y = [], [], 0.0
    for header, _, _ in ROWS:
        if header is not None:
            y -= 0.5 if not headers else 1.0
            headers.append((header, y))
            y -= 1.0
        ys.append(y)
        y -= 1.0

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    lo, hi = 0.45, 0.92
    ax.set_xlim(lo, hi)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    # chance
    ax.axvline(0.5, color=INK2, linewidth=0.8)
    # level reached by corruption (MTC), as a faint reference band
    c_lo = min(auroc[("MTC", "D1")], auroc[("MTC", "D3")])
    c_hi = max(auroc[("MTC", "D1")], auroc[("MTC", "D3")])
    ax.axvspan(c_lo, c_hi, color=BLUE, alpha=0.08, linewidth=0, zorder=0)

    # MTC slightly above the row, Essen slightly below, so neither hides the other
    for (corpus, color, marker, ms, dy) in (("MTC", BLUE, "o", 5.5, 0.17),
                                            ("Essen", ORANGE, "s", 4.8, -0.17)):
        xs = [auroc[(corpus, key)] for _, key, _ in ROWS]
        ax.plot(xs, [yy + dy for yy in ys], marker, ms=ms, color=color, mec="white",
                mew=0.8, linestyle="none", label=corpus, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, _, label in ROWS], color=INK)
    # group headers sit in the label column, right-aligned with the row labels
    label_x = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for header, yy in headers:
        ax.text(-0.03, yy, header, ha="right", va="center", fontsize=7.5,
                fontweight="bold", color=INK, transform=label_x)

    top = 0.9
    ax.set_ylim(min(ys) - 0.7, top)
    ax.text(0.5, top, " chance", ha="left", va="top", fontsize=6.5, color=INK2)
    ax.text((c_lo + c_hi) / 2, top, "corruption\nlevel", ha="center", va="top",
            fontsize=6.5, color=INK2, linespacing=1.0)

    ax.set_xlabel("Changed notes found (AUROC)\n0.5 = chance, 1 = perfect")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              fontsize=7, handletextpad=0.3, borderaxespad=0.1, columnspacing=1.2)


def main():
    style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    draw_panel(ax)
    fig.tight_layout()
    save(fig, "figure2")


if __name__ == "__main__":
    main()

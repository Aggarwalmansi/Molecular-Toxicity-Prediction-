"""
make_diagram.py
===============
Draws the hybrid architecture diagram for the report and presentation.

Run from the repository root:
    python -m phase2.hybrid.src.make_diagram

Saves reports/plots/hybrid_architecture.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from phase2.hybrid.src.hybrid_model import get_module_dir

# Colours: blue for Member 2's frozen deep learning parts, orange for Member 3's work
DL_COLOUR = "#aacfe8"
HYBRID_COLOUR = "#f5c396"
DESCRIPTOR_COLOUR = "#c6e3c3"
NEUTRAL_COLOUR = "#e4e4e4"


def draw_box(axis, x, y, width, height, text, colour, fontsize=9, style="round,pad=0.02"):
    """Draws one labelled box."""
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=style,
        facecolor=colour,
        edgecolor="#555555",
        linewidth=1.2,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height / 2, text,
              ha="center", va="center", fontsize=fontsize, linespacing=1.4)


def draw_arrow(axis, start, end):
    """Draws one arrow between two points."""
    axis.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color="#333333", linewidth=1.4, shrinkA=0, shrinkB=0),
    )


def main():
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)
    axis.axis("off")

    axis.text(5, 9.65, "Hybrid Model: Deep Learning Features + Molecular Descriptors + Random Forest",
              ha="center", va="center", fontsize=12, fontweight="bold")
    axis.text(5, 9.25, "Member 3  |  Tox21 Molecular Toxicity Prediction",
              ha="center", va="center", fontsize=9, color="#555555")

    # ---- Left column: the deep learning branch (Member 2, frozen) ----
    draw_box(axis, 0.4, 8.1, 3.4, 0.7,
             "Phase 1 features\n2048 Morgan bits + 5 descriptors  (2053)", NEUTRAL_COLOUR)
    draw_arrow(axis, (2.1, 8.1), (2.1, 7.6))

    draw_box(axis, 0.4, 6.9, 3.4, 0.7, "Dense 512 + BatchNorm + ReLU + Dropout", DL_COLOUR)
    draw_arrow(axis, (2.1, 6.9), (2.1, 6.4))

    draw_box(axis, 0.4, 5.7, 3.4, 0.7, "Dense 256 + BatchNorm + ReLU + Dropout", DL_COLOUR)
    draw_arrow(axis, (2.1, 5.7), (2.1, 5.2))

    draw_box(axis, 0.4, 4.5, 3.4, 0.7,
             "Dense 128 + BatchNorm + ReLU + Dropout\nLATENT LAYER  (z)", DL_COLOUR)

    # The classifier head that Member 3 does not use
    draw_arrow(axis, (3.8, 4.85), (4.6, 4.85))
    draw_box(axis, 4.6, 4.5, 1.9, 0.7, "Dense 12\n(Member 2 output)", NEUTRAL_COLOUR, fontsize=8)
    axis.text(5.55, 4.25, "not used by the hybrid", ha="center", fontsize=7.5,
              color="#888888", style="italic")

    axis.text(2.1, 3.95, "MultiTaskToxicNet — trained by Member 2, weights frozen",
              ha="center", fontsize=8, color="#2a6496", style="italic")

    # Branch A arrow going down from the latent layer
    draw_arrow(axis, (2.1, 4.5), (2.1, 2.75))
    axis.text(1.90, 3.45, "Branch A", ha="right", fontsize=8.5, fontweight="bold")
    axis.text(1.90, 3.15, "128 learned features", ha="right", fontsize=8)

    # ---- Right column: the descriptor branch ----
    draw_box(axis, 6.1, 6.9, 3.5, 1.1,
             "5 molecular descriptors\nMolWt, TPSA, MolLogP,\nNumHDonors, NumHAcceptors",
             DESCRIPTOR_COLOUR)
    axis.text(7.85, 6.6, "scaled in Phase 1 using training statistics only",
              ha="center", fontsize=7.5, color="#555555", style="italic")

    draw_arrow(axis, (7.85, 6.9), (7.85, 2.75))
    axis.text(8.05, 3.45, "Branch B", ha="left", fontsize=8.5, fontweight="bold")
    axis.text(8.05, 3.15, "5 interpretable features", ha="left", fontsize=8)

    # ---- Concatenation ----
    draw_box(axis, 1.6, 2.05, 6.8, 0.7,
             "CONCATENATE  →  133 features per molecule", HYBRID_COLOUR, fontsize=10)
    draw_arrow(axis, (5.0, 2.05), (5.0, 1.55))

    # ---- Random Forest ----
    draw_box(axis, 1.6, 0.85, 6.8, 0.7,
             "12 Random Forests, one per endpoint\n100 trees, max_depth 12, class_weight balanced",
             HYBRID_COLOUR, fontsize=9)
    draw_arrow(axis, (5.0, 0.85), (5.0, 0.45))

    axis.text(5.0, 0.25,
              "12 toxicity probabilities   |   each forest trained only on molecules "
              "whose label for that endpoint was measured",
              ha="center", fontsize=8.5)

    output_dir = get_module_dir() / "reports" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hybrid_architecture.png"
    plt.savefig(output_file, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {output_file}")

    make_comparison_plot(output_dir)


def make_comparison_plot(output_dir):
    """
    Draws the three-way comparison as grouped bars, one group per endpoint.

    Reads the comparison table that train_hybrid.py already wrote, so the plot
    can never disagree with the numbers in the report.
    """
    import pandas as pd

    reports_dir = get_module_dir() / "reports"
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))

    for axis, metric, filename in [
        (axes[0], "ROC-AUC", "comparison_roc_auc.csv"),
        (axes[1], "PR-AUC", "comparison_pr_auc.csv"),
    ]:
        table = pd.read_csv(reports_dir / filename, index_col=0)
        endpoints = [name for name in table.index if name != "MACRO AVERAGE"]
        table = table.loc[endpoints]

        positions = range(len(endpoints))
        width = 0.27
        axis.bar([p - width for p in positions], table["Standalone RF"], width,
                 label="Standalone RF", color="#7fa9cf")
        axis.bar(list(positions), table["Standalone DL"], width,
                 label="Standalone DL", color="#8fc98a")
        axis.bar([p + width for p in positions], table["Hybrid"], width,
                 label="Hybrid (DL features + descriptors)", color="#eb9f61")

        axis.set_xticks(list(positions))
        axis.set_xticklabels(endpoints, rotation=30, ha="right", fontsize=9)
        axis.set_ylabel(f"Test {metric}")
        axis.set_title(f"Test {metric} by endpoint", fontsize=11, fontweight="bold")
        axis.grid(axis="y", linestyle=":", alpha=0.5)
        axis.set_axisbelow(True)
        if metric == "ROC-AUC":
            axis.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
            axis.set_ylim(0.4, 0.9)
            axis.legend(fontsize=9, loc="upper right", ncol=3)

    plt.tight_layout()
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()

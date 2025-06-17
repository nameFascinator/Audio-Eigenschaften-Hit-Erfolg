"""
Creates a bar chart showing how audio features differ between hit songs and flops.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

data_file = Path("data/key_audio_features.csv")
output_file = Path("data/plots/hit_flop_deltas_bar.png")

# Make sure output dir exists
output_file.parent.mkdir(parents=True, exist_ok=True)

def create_comparison_chart():
    # Load data and sort by absolute difference size
    df = pd.read_csv(data_file)
    df = df.sort_values("Δ (hit-flop)", key=abs, ascending=False)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["feature"], df["Δ (hit-flop)"])

    # Add zero line for reference
    ax.axhline(0, color="black", linewidth=0.8)

    # Label each bar with the difference value and significance
    for i, (bar, sig) in enumerate(zip(bars, df["sig"])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:+.2f}\n{sig}",
                ha="center", va="bottom" if height > 0 else "top",
                fontsize=10)

    # Formatting
    ax.set_ylabel("Difference (Hit - Flop)")
    ax.set_title("How Hit Songs Differ from Flops")
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    create_comparison_chart()

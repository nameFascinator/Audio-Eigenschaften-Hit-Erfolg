# Find audio features that really separate hits from flops
# and save them to a CSV for further analysis

import pandas as pd
from pathlib import Path

# Paths
data_dir = Path("data")
input_file = data_dir / "hit_vs_flop_stats.csv"
output_file = data_dir / "key_audio_features.csv"

# What counts as "significant"?
P_VALUE_CUTOFF = 0.001  # Very significant only
MIN_DIFF = 0.03  # At least 3% difference


def get_sig_stars(p_val):
    """Convert p-value to significance stars."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""


def main():
    print("Loading audio feature statistics...")
    df = pd.read_csv(input_file)

    # Filter for statistically significant AND practically meaningful differences
    print(f"Filtering for p < {P_VALUE_CUTOFF} and |difference| >= {MIN_DIFF}")

    is_significant = df["p_value"] < P_VALUE_CUTOFF
    is_meaningful = df["Δ (hit-flop)"].abs() >= MIN_DIFF

    key_features = df[is_significant & is_meaningful].copy()

    if len(key_features) == 0:
        print("No features passed our criteria - maybe try relaxing the thresholds?")
        return

    # Add significance markers
    key_features["sig"] = key_features["p_value"].apply(get_sig_stars)

    # Calculate effect size (standardized difference)
    pooled_std = max(key_features["hit_mean"].std(), 0.000001)  # avoid div by zero
    key_features["effect_size"] = key_features["Δ (hit-flop)"].abs() / pooled_std

    # Choose what to save
    output_cols = ["feature", "hit_mean", "flop_mean", "Δ (hit-flop)",
                   "sig", "p_value", "effect_size"]

    # Write results
    key_features[output_cols].to_csv(output_file, index=False, float_format="%.4f")

    print(f"\nFound {len(key_features)} key distinguishing features!")
    print(f"Results saved to: {output_file}")
    print("\nTop features that separate hits from flops:")
    print("-" * 55)
    for _, row in key_features.iterrows():
        print(f"{row['feature']:15} {row['Δ (hit-flop)']:+.3f} {row['sig']}")


if __name__ == "__main__":
    main()

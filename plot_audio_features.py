"""
Loads Spotify song data and creates two charts:
1. Histograms for audio features like danceability and energy.
2. A heatmap showing how these features relate to each other.
Saves charts as PNG files in the 'data/plots' folder.
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up file paths
data_dir = pathlib.Path("data")
csv_path = data_dir / "master_song_dataset_audio.csv"
plots_dir = data_dir / "plots"

# Audio features to analyze
features = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]

def load_data():
    """Load song data and drop rows missing danceability."""
    print("Loading song data...")
    return pd.read_csv(csv_path, low_memory=False).dropna(subset=["danceability"])

def plot_histograms(data):
    """Create histograms for each audio feature."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()  # Easier to loop through

    for i, feature in enumerate(features):
        sns.histplot(data[feature], bins=40, kde=True, ax=axes[i])
        axes[i].set_title(feature.capitalize())

    plt.tight_layout()
    output_path = plots_dir / "audio_histograms.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histograms to {output_path}")

def plot_correlation_heatmap(data):
    """Create a heatmap of feature correlations."""
    corr_matrix = data[features].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0)
    plt.title("Audio Feature Correlations")

    output_path = plots_dir / "audio_corr_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")

def main():
    """Load data and generate charts."""
    plots_dir.mkdir(exist_ok=True)  # Create plots folder if needed
    songs = load_data()
    plot_histograms(songs)
    plot_correlation_heatmap(songs)
    print("All done! Charts are in the plots folder.")

if __name__ == "__main__":
    main()
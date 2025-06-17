"""
Plots a scatter graph to compare hit and flop songs based on their audio features,
like danceability and energy, using a simplified 2D view.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# File paths
data_path = "data/master_song_dataset_audio.csv"
plot_folder = "data/plots"
os.makedirs(plot_folder, exist_ok=True)

# Audio features to analyze
audio_features = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]


def load_songs():
    """Loads and cleans the song dataset."""
    songs = pd.read_csv(data_path)
    songs = songs.dropna(subset=["target"] + audio_features)
    songs["target"] = songs["target"].astype(int)
    return songs


def simplify_features(songs):
    """Reduces audio features to 2D for plotting."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(songs[audio_features])

    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(scaled_data)


def plot_songs(coords, labels):
    """Creates a scatter plot of hits vs flops."""
    plt.figure(figsize=(8, 6))

    # Plot flops (red) and hits (blue)
    plt.scatter(coords[labels == 0, 0], coords[labels == 0, 1],
                s=10, alpha=0.5, color="red", label="Flop")
    plt.scatter(coords[labels == 1, 0], coords[labels == 1, 1],
                s=10, alpha=0.5, color="blue", label="Hit")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Hits vs Flops by Audio Features")
    plt.legend()

    # Save plot
    plot_path = os.path.join(plot_folder, "hit_flop_scatter.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    print("Loading songs...")
    songs = load_songs()

    print("Simplifying features...")
    coords = simplify_features(songs)

    print("Plotting...")
    plot_songs(coords, songs["target"])

    print("All done!")

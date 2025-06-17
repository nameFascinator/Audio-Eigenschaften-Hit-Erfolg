"""
Analyzes audio features to determine their importance in song classification.
Outputs a CSV with feature importance scores and a bar chart.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# File paths
DATA_FILE = "data/master_song_dataset_audio.csv"
OUTPUT_CSV = "data/logreg_feature_importance.csv"
OUTPUT_PLOT = "data/plots/logreg_feature_importance.png"

# Audio features to analyze
FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]


def load_data():
    """Load and clean the dataset."""
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["target"] + FEATURES)
    X = StandardScaler().fit_transform(df[FEATURES])
    y = df["target"].astype(int)
    return X, y


def train_model(X, y):
    """Train a logistic regression model and get feature coefficients."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model.coef_[0]


def make_importance_table(coefficients):
    """Create a sorted table of feature importance."""
    table = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": coefficients,
        "Odds Ratio": np.exp(coefficients),
        "Absolute Coefficient": np.abs(coefficients)
    })
    return table.sort_values("Absolute Coefficient", ascending=False)


def save_table(table):
    """Save the importance table to a CSV file."""
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved table to: {OUTPUT_CSV}")


def plot_importance(table):
    """Create and save a bar chart of feature importance."""
    os.makedirs("data/plots", exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(table["Feature"], table["Coefficient"], color="skyblue")
    plt.axhline(0, color="black", linewidth=0.8)

    # Add coefficient labels on bars
    for bar, coef in zip(bars, table["Coefficient"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            coef,
            f"{coef:+.2f}",
            ha="center",
            va="bottom" if coef > 0 else "top"
        )

    plt.ylabel("Coefficient Value")
    plt.title("Importance of Audio Features in Song Classification")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved plot to: {OUTPUT_PLOT}")


def main():
    print("Loading data...")
    X, y = load_data()

    print("Training model...")
    coefficients = train_model(X, y)

    print("Building importance table...")
    table = make_importance_table(coefficients)

    print("Saving results...")
    save_table(table)
    plot_importance(table)

    print("Done! Check the data folder for results.")


if __name__ == "__main__":
    main()
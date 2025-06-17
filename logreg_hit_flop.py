"""
Predicts if a song will be a hit or flop based on its audio features.
Trains a model and shows its accuracy with a ROC curve plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Define paths and features
DATA_FILE = "data/master_song_dataset_audio.csv"
PLOT_DIR = "data/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]


def predict_song_success():
    # Load and clean data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["target"] + FEATURES)

    # Prepare data for training
    X = df[FEATURES]
    y = df["target"].astype(int)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC: {auc:.3f}")

    # Plot and save ROC curve
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve: Hit vs Flop")
    plt.savefig(f"{PLOT_DIR}/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to {PLOT_DIR}/roc_curve.png")


if __name__ == "__main__":
    predict_song_success()

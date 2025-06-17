"""
Merges Spotify audio features (danceability, energy, etc.) into the core song
dataset, matching on track + first-artist names, then writes the enriched file.
"""

from pathlib import Path
import ast
import re
import pandas as pd

# File paths and settings
DATA_DIR = Path("data")
SONGS_FILE = DATA_DIR / "master_song_dataset.csv"
TRACKS_FILE = DATA_DIR / "tracks_features.csv"
TOP200_FILE = DATA_DIR / "spotify_top_songs_audio_features.csv"
OUTPUT_FILE = DATA_DIR / "master_song_dataset_audio.csv"

AUDIO_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
    "duration_ms", "key", "mode", "time_signature",
]

def clean_text(text):
    """Clean text for matching - lowercase, remove brackets/dashes, normalize spaces."""
    text = str(text).lower()
    text = re.sub(r"[()\[\]{}-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_first_artist(artists_str):
    """Extract first artist from stringified list, fallback to original string."""
    try:
        return ast.literal_eval(artists_str)[0]
    except:
        return artists_str

def load_tracks_sample():
    print("Reading large tracks file (taking sample)...")
    cols = ["name", "artists"] + AUDIO_FEATURES
    df = pd.read_csv(TRACKS_FILE, usecols=cols, low_memory=False)
    # Take a sample to make processing faster
    df = df.sample(frac=0.08, random_state=42)
    df["artist_name"] = df["artists"].apply(get_first_artist)
    df["match_key"] = df["name"].apply(clean_text) + "|" + df["artist_name"].apply(clean_text)
    return df[["match_key"] + AUDIO_FEATURES]

def load_top200_features():
    print("Reading Spotify Top 200 features...")
    df = pd.read_csv(TOP200_FILE, low_memory=False)
    # Split artist names and take first one
    df["artist_names"] = df["artist_names"].str.split(";").str[0]
    df["match_key"] = df["track_name"].apply(clean_text) + "|" + df["artist_names"].apply(clean_text)
    return df[["match_key"] + AUDIO_FEATURES]

# Main processing
print("Enriching master_song_dataset.csv with audio features...")

# Load main song dataset
songs = pd.read_csv(SONGS_FILE, low_memory=False)
songs["match_key"] = songs["track_name"].apply(clean_text) + "|" + songs["artist_name"].apply(clean_text)

# Combine audio features from both sources
print("Combining audio feature sources...")
audio_data = pd.concat([
    load_tracks_sample(),
    load_top200_features()
]).drop_duplicates("match_key")

print("Merging datasets...")
result = songs.merge(audio_data, on="match_key", how="left")

# Show matching stats
matched_count = result[AUDIO_FEATURES].notna().any(axis=1).sum()
total_count = len(result)
print(f"Added audio features to {matched_count:,} out of {total_count:,} songs")

# Save final result
result.drop(columns="match_key").to_csv(OUTPUT_FILE, index=False)
print(f"Saved enriched dataset to {OUTPUT_FILE}")
print("Done!")

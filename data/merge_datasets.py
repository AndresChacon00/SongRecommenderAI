import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# File paths
DATA_CSV = "../data.csv"
SPOTIFY_CSV = "../SpotifyFeatures.csv"
SPOTIFY_V2_CSV = "../SpotifyFeaturesV2.csv"
OUTPUT_CSV = "merged_songs.csv"

logging.info(f"Reading {DATA_CSV}...")
# Read data.csv and build a mapping from id to year
id_to_year = {}
with open(DATA_CSV, newline="", encoding="utf-8") as data_file:
    reader = csv.DictReader(data_file)
    for row in reader:
        id_to_year[row["id"]] = row["year"]
logging.info(f"Loaded {len(id_to_year)} song ids from data.csv.")

logging.info(f"Reading {SPOTIFY_CSV}...")
# Read SpotifyFeatures.csv and merge with year info
with open(SPOTIFY_CSV, newline="", encoding="utf-8") as spotify_file:
    reader = csv.DictReader(spotify_file)
    spotify_rows = list(reader)
    fieldnames = reader.fieldnames + ["release_year"]
logging.info(f"Loaded {len(spotify_rows)} rows from SpotifyFeatures.csv.")

logging.info(f"Reading {SPOTIFY_V2_CSV}...")
# Read SpotifyFeaturesV2.csv to get additional track info
with open(SPOTIFY_V2_CSV, newline="", encoding="utf-8") as spotify_v2_file:
    reader = csv.DictReader(spotify_v2_file)
    spotify_v2_rows = list(reader)
logging.info(f"Loaded {len(spotify_v2_rows)} rows from SpotifyFeaturesV2.csv.")

# Write merged rows to new CSV
match_count = 0
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_file:
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()
    merged_ids = set()
    for row in spotify_rows:
        track_id = row["track_id"]
        if track_id in id_to_year:
            row["release_year"] = id_to_year[track_id]
            merged_ids.add(track_id)
            writer.writerow(row)
            match_count += 1

    # If any songs on spotify_v2_rows are not in merged_ids, add them to the csv
    for row in spotify_v2_rows:
        track_id = row["track_id"]
        if (
            track_id not in merged_ids
            and track_id in id_to_year
        ):
            # remove album_name key from row
            row.pop("album_name", None)
            writer.writerow(row)
            match_count += 1

logging.info(f"Wrote {match_count} matched songs to {OUTPUT_CSV}.")

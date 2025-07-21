import csv
import os
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger()

# Log in
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

if not client_id or not client_secret:
    logger.error("Spotify client ID and secret must be set in environment variables.")
    exit(1)

client_credencials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
spotify_client = spotipy.Spotify(client_credentials_manager=client_credencials_manager)


output_path = "../SpotifyFeaturesV2.csv"
input_path = "../SpotifyFeatures.csv"

# Read existing track_ids from the output file (if it exists)
existing_track_ids = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as csv_output:
        reader = csv.DictReader(csv_output)
        for row in reader:
            existing_track_ids.add(row.get("track_id"))

with open(input_path, "r", encoding="utf-8") as csv_input:
    reader = csv.DictReader(csv_input)
    fieldnames = reader.fieldnames + ["release_year", "album_name"]

    # Write header only if file does not exist or is empty
    write_header = not os.path.exists(output_path) or os.stat(output_path).st_size == 0

    with open(output_path, "a", newline="", encoding="utf-8") as csv_output:
        writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in reader:
            track_id = row["track_id"]
            track_name = row["track_name"]

            if track_id in existing_track_ids:
                continue  # Skip if already processed

            try:
                track_data = spotify_client.track(track_id)
                album = track_data.get("album", None)
                if not album:
                    raise ValueError("Missing album")

                release_date = str(album.get("release_date", ""))
                row["release_year"] = release_date.split("-")[0]
                row["album_name"] = album.get("name", "")

            except Exception as e:
                logger.error(
                    f"Error fetching track ID for {track_name} with id {track_id}: {e}"
                )
                row["release_year"] = None
                row["album_name"] = None

            writer.writerow(row)
            existing_track_ids.add(track_id)

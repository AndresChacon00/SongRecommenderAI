import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()


class SpotifyClient:
    """Cliente para interactuar con la API de Spotify"""

    def __init__(self, client_id=None, client_secret=None):
        """Inicializa el cliente de Spotify"""
        if client_id is None or client_secret is None:
            client_id = os.getenv("SPOTIFY_CLIENT_ID")
            client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
        self.client = spotipy.Spotify(
            client_credentials_manager=client_credentials_manager
        )

    def search(self, query: str, limit=10):
        """Busca canciones en Spotify"""
        results = self.client.search(q=query, type="track", limit=limit)
        tracks = results.get("tracks", {}).get("items", [])
        return [
            {
                "name": track["name"],
                "artist": ", ".join(artist["name"] for artist in track["artists"]),
                "uri": track["uri"],
                "id": track["id"],
            }
            for track in tracks
        ]

    def get_audio_features(self, songs):
        """Obtiene las características de audio de una o varias canciones"""
        features = self.client.audio_features([song.id for song in songs])
        if features and len(features) > 0:
            return features
        return None

    def get_audio_features_offline(self, songs):
        """Obtiene las características de audio de una o varias canciones desde un archivo local"""
        df = pd.read_csv("../data/SpotifyFeatures.csv")
        matched_features = []
        for song in songs:
            match = df[
                (df["id"] == song.id)
                | (
                    (df["track_name"] == song.name)
                    & (df["artist_name"] == song.artist_name)
                )
            ]
            if not match.empty:
                matched_features.append(match.iloc[0].to_dict())
        return matched_features

    def get_audio_features_reccobeats(self, songs):
        """Obtiene las características de audio de una o varias canciones desde un archivo local"""
        url = "https://api.reccobeats.com/v1/track"
        url += "?ids=" + ",".join([song.id for song in songs])
        response = requests.get(url)
        if response.status_code == 200:
            content = response.json().get("content", [])
            reccobeats_ids = [c["id"] for c in content]

            features = []
            for recco_id, spotify_id in zip(
                reccobeats_ids, [song.id for song in songs]
            ):
                url = f"https://api.reccobeats.com/v1/track/{recco_id}/audio-features"
                feat_response = requests.get(url)
                if feat_response.status_code == 200:
                    feat_data = feat_response.json()
                    feat_data["id"] = spotify_id
                    features.append(feat_data)
                else:
                    print(
                        f"Error fetching features for {spotify_id}: {feat_response.status_code}"
                    )

            return features

        else:
            print(f"Error fetching audio features: {response.status_code}")
            return []

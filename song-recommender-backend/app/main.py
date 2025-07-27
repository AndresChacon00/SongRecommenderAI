from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from app.recommender import KNN_model, KMeans_model, DBSCAN_model, DEFAULT_FEATURES
from app.spotify_client import SpotifyClient

knn_model = None
kmeans_model = None
dbscan_model = None
spotify_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the robot on startup.
    """
    global knn_model, kmeans_model, dbscan_model, spotify_client
    # Init models
    csv_df = pd.read_csv("data/merged_songs.csv")
    knn_model = KNN_model()
    knn_model.train(csv_df)
    kmeans_model = KMeans_model()
    kmeans_model.train(csv_df)
    dbscan_model = DBSCAN_model()
    dbscan_model.train(csv_df)
    # Init Spotify client
    spotify_client = SpotifyClient()
    yield
    print("Exiting...")


app = FastAPI(title="Song Recommender API", lifespan=lifespan)


# class SongFeatures(BaseModel):
#     acousticness: float
#     danceability: float
#     energy: float
#     instrumentalness: float
#     liveness: float
#     loudness: float
#     speechiness: float
#     valence: float
#     tempo: float


class SongData(BaseModel):
    id: str
    name: str
    artist_name: str


class RecommendationRequest(BaseModel):
    songs: list[SongData]
    model_type: Literal["knn", "kmeans", "dbscan"]


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    """
    Recommend songs based on the provided features and model type.
    The request should contain a list of song IDs and the model type to use for recommendations.
    """
    # Get features
    if spotify_client is None:
        return {"error": "Spotify client not initialized."}
    song_features = spotify_client.get_audio_features_reccobeats(request.songs)
    if not song_features:
        return {"error": "No features found for the provided songs."}

    # Only keep fields specified in SongFeatures model
    filtered_features = [
        {**{k: s[k] for k in DEFAULT_FEATURES if k in s}, "track_id": song.id}
        for s, song in zip(song_features, request.songs)
    ]
    df = pd.DataFrame(filtered_features)
    if request.model_type == "knn":
        recs = knn_model.recomend_songs(df).to_dict(orient="records")
        # Filter out songs that are already in the request
        input_ids = {song.id for song in request.songs}
        recs = [rec for rec in recs if rec.get("track_id") not in input_ids]
    elif request.model_type == "kmeans":
        recs = kmeans_model.recommend_songs(df)
    else:
        recs = dbscan_model.recommend_songs(df)
    return {"recommendations": recs}


@app.get("/search")
def search(query: str):
    """
    Search for songs using the Spotify client.
    """
    if spotify_client is None:
        return {"error": "Spotify client not initialized."}
    results = spotify_client.search(query)
    return {"results": results}

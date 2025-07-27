from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from app.recommender import KNN_model, KMeans_model, DBSCAN_model

knn_model = None
kmeans_model = None
dbscan_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the robot on startup.
    """
    global knn_model, kmeans_model, dbscan_model
    csv_df = pd.read_csv("data/merged_songs.csv")
    knn_model = KNN_model()
    knn_model.train(csv_df)
    kmeans_model = KMeans_model()
    kmeans_model.train(csv_df)
    dbscan_model = DBSCAN_model()
    dbscan_model.train(csv_df)
    yield
    print("Exiting...")


app = FastAPI(title="Song Recommender API", lifespan=lifespan)


class SongFeatures(BaseModel):
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    valence: float
    tempo: float


class RecommendationRequest(BaseModel):
    songs: list[SongFeatures]
    model_type: Literal["knn", "kmeans", "dbscan"]


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    df = pd.DataFrame([s.dict() for s in request.songs])
    if request.model_type == "knn":
        recs = knn_model.recomend_songs(df).to_dict(orient="records")
    elif request.model_type == "kmeans":
        recs = kmeans_model.recommend_songs(df)
    else:
        recs = dbscan_model.recommend_songs(df)
    return {"recommendations": recs}

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.recommender import KNN_model

app = FastAPI()
model = KNN_model()
model.train(pd.read_csv("data/merged_songs.csv"))

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

@app.post("/recommend")
def recommend(songs: list[SongFeatures]):
    df = pd.DataFrame([s.dict() for s in songs])
    recs = model.recomend_songs(df).to_dict(orient="records")
    return {"recommendations": recs}
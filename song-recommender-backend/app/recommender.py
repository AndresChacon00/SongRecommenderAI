import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

class KNN_model:
    def __init__(self, features=None, n_neighbors=10):
        if features is None:
            self.features = [
                'acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'valence', 'tempo'
            ]
        else:
            self.features = features
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn = None
        self.train_data = None

    def train(self, data, test_size=0.2, random_state=42):
        self.train_data, _ = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        self.X_train = self.scaler.fit_transform(self.train_data[self.features])
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self.knn.fit(self.X_train)

    def recomend_songs(self, songs_df, n_recommendations=5):
        mean_vector = songs_df[self.features].mean().values.reshape(1, -1)
        mean_vector_scaled = self.scaler.transform(pd.DataFrame(mean_vector, columns=self.features))
        distances, indices = self.knn.kneighbors(mean_vector_scaled, n_neighbors=n_recommendations)
        recommended = self.train_data.iloc[indices[0]].copy()
        recommended['distance'] = distances[0]
        max_dist = distances[0].max() if distances[0].max() > 0 else 1
        recommended['similarity'] = 1 - (recommended['distance'] / max_dist)
        return recommended.sort_values('distance')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

DEFAULT_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "valence",
    "tempo",
]


class KNN_model:
    def __init__(self, features=None, n_neighbors=10):
        if features is None:
            self.features = DEFAULT_FEATURES
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
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        self.knn.fit(self.X_train)

    def recomend_songs(self, songs_df, n_recommendations=5):
        mean_vector = songs_df[self.features].mean().values.reshape(1, -1)
        mean_vector_scaled = self.scaler.transform(
            pd.DataFrame(mean_vector, columns=self.features)
        )
        distances, indices = self.knn.kneighbors(
            mean_vector_scaled, n_neighbors=n_recommendations
        )
        recommended = self.train_data.iloc[indices[0]].copy()
        recommended["distance"] = distances[0]
        max_dist = distances[0].max() if distances[0].max() > 0 else 1
        recommended["similarity"] = 1 - (recommended["distance"] / max_dist)
        return recommended.sort_values("distance")


class KMeans_model:
    """Clase para implementar el modelo de recomendación mediante clustering KMeans."""

    def __init__(self, n_clusters=8, random_state=42, features=DEFAULT_FEATURES):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.features = features

    def train(self, data, test_size=0.2):
        """Entrena el modelo KMeans con los datos proporcionados."""
        self.train_data, self.test_data = train_test_split(
            data, test_size=test_size, random_state=self.random_state
        )
        self.X_train = self.train_data[self.features].values
        self.fit(self.X_train)

    def fit(self, X):
        """Ajusta el modelo KMeans a los datos en X."""
        self.model = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        ).fit(X)
        return self.model

    def predict(self, X):
        """Predice las etiquetas de cluster para los datos en X."""
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise ValueError("El modelo no ha sido entrenado.")

    def _flatten_dict_list(self, dict_list: list):
        """Aplana una lista de diccionarios en un único diccionario"""
        flattened_dict = dict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []
        for dictionary in dict_list:
            for key, value in dictionary.items():
                flattened_dict[key].append(value)
        return flattened_dict

    def _get_mean_vector(self, song_list):
        """Calcula el vector medio representativo de una lista de canciones."""
        song_vectors = []
        for song in song_list:
            song_vector = song[self.features].values
            song_vectors.append(song_vector)
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis=0)

    def recommend_songs(self, song_list, spotify_data, n_songs=10):
        """
        Ofrece una lista de `n_songs` recomendaciones a partir de una lista de canciones escuchadas,
        recomendando solo canciones del mismo cluster.
        """
        metadata_cols = [
            "track_name",
            "artist_name",
            "track_id",
            "release_year",
            "distance",
        ]
        song_dict = self._flatten_dict_list(song_list)
        song_center = self._get_mean_vector(song_list)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(spotify_data[self.features])
        song_center_df = pd.DataFrame([song_center], columns=self.features)
        scaled_song_center = scaler.transform(song_center_df)

        # Ajustar KMeans si no está ajustado
        if self.model is None:
            self.fit(scaled_data)

        # Obtener el cluster del centro de las canciones escuchadas
        cluster_labels = self.model.labels_
        center_cluster = self.predict(scaled_song_center)[0]

        # Filtrar canciones del mismo cluster y no escuchadas
        mask_cluster = cluster_labels == center_cluster
        mask_not_listened = ~spotify_data["track_id"].isin(song_dict["track_id"])
        mask = mask_cluster & mask_not_listened.values
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            return []

        filtered_scaled_data = scaled_data[filtered_indices]
        distances = cdist(scaled_song_center, filtered_scaled_data, "cosine")[0]
        top_indices = np.argsort(distances)[:n_songs]
        selected_indices = filtered_indices[top_indices]

        rec_songs = spotify_data.iloc[selected_indices].copy()
        rec_songs["distance"] = distances[top_indices]
        return rec_songs[metadata_cols].to_dict(orient="records")

    def evaluate(self, test_data, train_data, n_songs=10):
        """
        Evalúa el modelo usando métricas de recomendación.
        """
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[self.features])
        if self.model is None:
            self.fit(X_train)
        precisiones = []
        recalls = []
        average_precisions = []
        ndcgs = []
        distancias = []
        for i, row in test_data.iterrows():
            recommendations = self.recommend_songs([row], train_data, n_songs=n_songs)
            if not recommendations:
                continue
            recommended_global_idx = [
                train_data[train_data["track_id"] == rec["track_id"]].index[0]
                for rec in recommendations
            ]
            recommended_idx = [
                train_data.index.get_loc(idx) for idx in recommended_global_idx
            ]
            test_vector = scaler.transform(
                pd.DataFrame([row[self.features]], columns=self.features)
            )
            dists = np.linalg.norm(X_train - test_vector, axis=1)
            sorted_idx = np.argsort(dists)
            relevancias = np.zeros(len(train_data))
            relevancias[sorted_idx[:n_songs]] = 1
            recommended_relevancias = relevancias[recommended_idx]
            num_relevantes = n_songs
            distancias.append(recommendations[0]["distance"])
            precision = np.sum(recommended_relevancias) / n_songs
            precisiones.append(precision)
            recall = (
                np.sum(recommended_relevancias) / num_relevantes
                if num_relevantes > 0
                else 0
            )
            recalls.append(recall)
            if np.sum(recommended_relevancias) > 0:
                ap = 0
                hits = 0
                for idx, rel in enumerate(recommended_relevancias, 1):
                    if rel:
                        hits += 1
                        ap += hits / idx
                ap /= np.sum(recommended_relevancias)
            else:
                ap = 0
            average_precisions.append(ap)
            recommended_scores = 1 - dists[recommended_idx] / (
                dists[recommended_idx].max() if dists[recommended_idx].max() > 0 else 1
            )
            from sklearn.metrics import ndcg_score

            ndcg = ndcg_score([relevancias[recommended_idx]], [recommended_scores])
            ndcgs.append(ndcg)
        distancia_media = np.mean(distancias) if distancias else 0
        print(
            f"Distancia euclidiana promedio entre test y recomendación: {distancia_media:.4f}"
        )
        print(f"Precisión@{n_songs}: {np.mean(precisiones):.4f}")
        print(f"Recall@{n_songs}: {np.mean(recalls):.4f}")
        print(f"MAP: {np.mean(average_precisions):.4f}")
        print(f"NDCG@{n_songs}: {np.mean(ndcgs):.4f}")


class DBSCAN_model:
    """Clase para implementar el modelo de recomendación mediante clustering DBSCAN."""

    def __init__(self, eps=0.5, min_samples=5, features=DEFAULT_FEATURES):
        """Inicializa el modelo con los parámetros de DBSCAN."""
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.features = features

    def train(self, data, test_size=0.2):
        """Entrena el modelo DBSCAN con los datos proporcionados."""
        self.train_data, self.test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        self.X_train = self.train_data[self.features].values
        self.fit(self.X_train)

    def fit(self, X):
        """Ajusta el modelo DBSCAN a los datos en X."""
        self.model = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="euclidean"
        ).fit(X)
        return self.model

    def predict(self, X):
        """Predice las etiquetas de cluster para los datos en X."""
        if self.model is not None:
            return self.model.fit_predict(X)
        else:
            raise ValueError("El modelo no ha sido entrenado.")

    def _flatten_dict_list(self, dict_list: list[dict]):
        """Aplana una lista de diccionarios en un único diccionario"""
        flattened_dict = dict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []

        for dictionary in dict_list:
            for key, value in dictionary.items():
                flattened_dict[key].append(value)

        return flattened_dict

    def _get_mean_vector(self, song_list):
        """Calcula el vector medio representativo de una lista de canciones."""
        song_vectors = []

        for song in song_list:
            song_vector = song[self.features].values
            song_vectors.append(song_vector)

        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis=0)

    def recommend_songs(self, song_list, spotify_data, n_songs=10):
        """Ofrece una lista de `n_songs` recomendaciones a partir de una lista de canciones escuchadas."""
        metadata_cols = [
            "track_name",
            "artist_name",
            "track_id",
            "release_year",
            "distance",
        ]
        song_dict = self._flatten_dict_list(song_list)

        song_center = self._get_mean_vector(song_list)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(spotify_data[self.features])
        song_center_df = pd.DataFrame([song_center], columns=self.features)
        scaled_song_center = scaler.transform(song_center_df)
        # scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, "cosine")[0]

        # Exclude already listened songs
        mask = ~spotify_data["track_id"].isin(song_dict["track_id"])
        filtered_indices = np.where(mask)[0]
        filtered_distances = distances[filtered_indices]

        # Get indices of the n_songs closest songs
        top_indices = filtered_indices[np.argsort(filtered_distances)[:n_songs]]

        rec_songs = spotify_data.iloc[top_indices].copy()
        rec_songs["distance"] = distances[top_indices]
        return rec_songs[metadata_cols].to_dict(orient="records")

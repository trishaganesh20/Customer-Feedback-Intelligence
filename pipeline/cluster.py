import numpy as np
from sklearn.cluster import KMeans

def cluster_embeddings(embeddings: np.ndarray, k: int = 8, seed: int = 42) -> np.ndarray:
    """
    KMeans clustering returning cluster_id per row.
    """
    if len(embeddings) < k:
        k = max(2, min(len(embeddings), k))

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels

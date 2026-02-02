import os
import numpy as np
from openai import OpenAI

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Returns embeddings as (n, d) numpy array.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Batch requests to avoid giant payloads
    batch_size = 100
    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in resp.data])

    return np.array(vectors, dtype=np.float32)

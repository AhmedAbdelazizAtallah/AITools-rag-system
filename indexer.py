import faiss
import numpy as np
from typing import List, Dict, Any


class FaissIndex:
    """Vector Similarity Search Acceleration using FAISS (Facebook AI Similarity Search)."""
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        dimension = self.embeddings.shape[1]
        
        # Using Inner Product (IP) index. Since vectors are normalized, 
        # IP is mathematically equivalent to Cosine Similarity but computationally faster.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index for the most similar vector representations."""
        q_vec = query_embedding.astype(np.float32)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "doc_id": int(idx),
                    "score": float(score)
                })
        return results
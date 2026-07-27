import re
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import RRF_K, EMBEDDING_MODEL_NAME


def tokenize(text: str) -> List[str]:
    """Tokenize text by extracting lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Lexical Search Engine (BM25) for exact keyword matching and term frequency weighting."""
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.tokenized_corpus = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "doc_id": int(idx),
                "score": float(scores[idx]),
                "rank": rank + 1
            })
        return results


class DenseRetriever:
    """Semantic Search Engine (Dense Embeddings) for context and meaning understanding."""
    def __init__(self, corpus: List[str], model_name: str = EMBEDDING_MODEL_NAME):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        # Normalize embeddings to enable fast dot-product similarity (equivalent to Cosine)
        self.embeddings = self.model.encode(
            corpus, convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "doc_id": int(idx),
                "score": float(scores[idx]),
                "rank": rank + 1
            })
        return results


class HybridRRFRetriever:
    """Hybrid Search Engine combining Lexical and Semantic results via Reciprocal Rank Fusion (RRF)."""
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    def search(self, query: str, top_k: int = 5, min_dense_score: float = 0.25) -> List[Dict[str, Any]]:
        fetch_k = top_k * 2
        
        # 1. فحص أعلى تشابه دلالي للسؤال مع الملفات
        dense_res = self.dense.search(query, top_k=fetch_k)
        
        # إذا كانت أعلى نتيجة تشابه دلالي أقل من الحد الأدنى، نرجع قائمة فارغة فوراً
        if not dense_res or dense_res[0]["score"] < min_dense_score:
            return []

        bm25_res = self.bm25.search(query, top_k=fetch_k)
        rrf_scores = {}

        # حساب سكور RRF للبحث اللفظي
        for item in bm25_res:
            doc_id = item["doc_id"]
            rank = item["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank))

        # حساب سكور RRF للبحث الدلالي
        for item in dense_res:
            doc_id = item["doc_id"]
            rank = item["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in sorted_docs
        ]
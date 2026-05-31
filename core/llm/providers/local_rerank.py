"""
本地 Rerank（基于余弦相似度）

当没有百炼 Rerank API Key 时的 fallback 方案。
利用已有 embedding 模型计算 query 与文档向量的 cosine similarity，
按相似度降序重排。

注意：会额外调用 embedding API（仅对 top_k 条文档），
适合 top_k 较小（≤20）的场景。
"""

import math
from typing import List, Tuple

from langchain_core.embeddings import Embeddings


class LocalReranker:
    """
    本地余弦相似度 Reranker。

    用法:
        from core.llm.providers.local_rerank import LocalReranker
        from core.llm.providers.ollama_embedding import get_ollama_embedding

        reranker = LocalReranker(top_n=5, embedder=get_ollama_embedding())
        scores = reranker.rerank(query="解雇赔偿", documents=[doc1, doc2, ...])
    """

    def __init__(self, top_n: int = 5, embedder=None):
        self.top_n = top_n
        self.embedder = embedder
        if self.embedder is None:
            raise ValueError("LocalReranker 需要提供 embedder 实例")

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def rerank(
        self, query: str, documents: List[str]
    ) -> List[Tuple[str, float]]:
        """
        对文档列表进行重排序。

        返回: [(document_text, score), ...] 按 score 降序排列
        """
        if not documents:
            return []

        query_vec = self.embedder.embed_query(query)
        doc_vecs = self.embedder.embed_documents(documents)

        scored = [
            (doc, self._cosine_similarity(query_vec, vec))
            for doc, vec in zip(documents, doc_vecs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_n]

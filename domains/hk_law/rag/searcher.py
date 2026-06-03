"""
向量检索器

ES kNN 向量检索 + 可选 rerank 重排序。
"""
from typing import List

from langchain_core.documents import Document
from core.utils.logger import get_logger
from domains.hk_law.rag.config import RAGConfig

logger = get_logger(__name__)

_default_config = RAGConfig()


def search(
    domain: str,
    query: str,
    top_k: int = 10,
    rerank_top_n: int = 5,
    use_rerank: bool = True,
    config: RAGConfig = None,
) -> List[Document]:
    """
    检索流程：
    1. ES kNN 向量检索 top_k 条
    2. （可选）百炼 rerank 重排序，取 rerank_top_n 条

    返回: Document 列表
    """
    cfg = config or _default_config
    es = cfg.get_es_client()
    index = cfg.index_name(domain)

    # 获取 query embedding
    embedder = cfg.get_embedder()
    query_vector = embedder.embed_query(query)

    # kNN 检索
    resp = es.search(
        index=index,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "_source": ["content", "source", "domain"],
        },
    )

    hits = resp["hits"]["hits"]
    logger.debug(f"[ES] {index}: kNN 召回 {len(hits)} 条")
    if not hits:
        logger.warning(f"[ES] {index}: 检索无结果")
        return []

    docs = [
        Document(
            page_content=h["_source"]["content"],
            metadata={
                "source": h["_source"]["source"],
                "domain": h["_source"]["domain"],
                "es_score": h["_score"],
            },
        )
        for h in hits
    ]

    # Rerank
    if use_rerank and len(docs) > 1:
        logger.debug(f"[RAG] 开始 rerank (provider={cfg.rerank_provider})")
        reranker = cfg.get_reranker(top_n=rerank_top_n)
        texts = [d.page_content for d in docs]
        reranked = reranker.rerank(query, texts)
        reranked_docs = []
        for idx, score in reranked:
            d = docs[idx]
            d.metadata["rerank_score"] = score
            reranked_docs.append(d)
        docs = reranked_docs

    return docs[:rerank_top_n]

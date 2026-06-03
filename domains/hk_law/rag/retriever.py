"""
法域专用检索器包装类

为 Agent 提供简洁的检索接口，封装召回数量调整和分数返回。
"""
from typing import List, Tuple

from langchain_core.documents import Document
from domains.hk_law.rag.config import RAGConfig
from domains.hk_law.rag.searcher import search
from domains.hk_law.rag.indexer import index_documents
from domains.hk_law.rag.es_client import delete_index


class DomainRetriever:
    """
    法域专用检索器。
    支持传入自定义 RAGConfig，实现同一进程中多组不同配置并存。
    """

    def __init__(self, domain: str, top_k: int = 5, config: RAGConfig = None):
        self.domain = domain
        self.top_k = top_k
        self.config = config

    def search(self, query: str) -> List[Document]:
        """检索与 query 相关的法律条文片段"""
        return search(
            domain=self.domain,
            query=query,
            top_k=self.top_k * 2,  # 先多召回一些，rerank 后取 top_k
            rerank_top_n=self.top_k,
            use_rerank=True,
            config=self.config,
        )

    def search_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """检索并返回相似度分数"""
        docs = self.search(query)
        return [(d, d.metadata.get("rerank_score", d.metadata.get("es_score", 0))) for d in docs]

    def rebuild(self):
        """重新构建索引（文档更新后调用）"""
        delete_index(self.domain, config=self.config)
        index_documents(self.domain, config=self.config)

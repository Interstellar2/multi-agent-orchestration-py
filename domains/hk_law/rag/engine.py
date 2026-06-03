"""
RAG 引擎 —— 向后兼容入口

原 engine.py 已拆分为以下专注模块：
  - config.py          : RAGConfig 配置对象
  - es_client.py       : Elasticsearch 客户端与索引管理
  - document_loader.py : 文件系统文档加载
  - indexer.py         : 文档切分、embedding、批量写入
  - searcher.py        : kNN 向量检索 + rerank
  - retriever.py       : DomainRetriever 包装类

此文件保留旧 import 路径的向后兼容，新代码建议直接导入子模块。
"""
from domains.hk_law.rag.config import RAGConfig
from domains.hk_law.rag.document_loader import load_documents
from domains.hk_law.rag.es_client import (
    get_es_client,
    index_name,
    ensure_index,
    delete_index,
    check_es_health,
)
from domains.hk_law.rag.indexer import index_documents
from domains.hk_law.rag.searcher import search
from domains.hk_law.rag.retriever import DomainRetriever

# 兼容旧导入：DOCUMENTS_DIR
DOCUMENTS_DIR = RAGConfig().documents_dir

__all__ = [
    "RAGConfig",
    "load_documents",
    "get_es_client",
    "index_name",
    "ensure_index",
    "delete_index",
    "check_es_health",
    "index_documents",
    "search",
    "DomainRetriever",
    "DOCUMENTS_DIR",
]

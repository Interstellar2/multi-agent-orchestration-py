"""
索引构建器

将文档切分、生成 embedding、批量写入 Elasticsearch 的完整流程。
"""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.utils.logger import get_logger
from domains.hk_law.rag.config import RAGConfig
from domains.hk_law.rag.document_loader import load_documents
from domains.hk_law.rag.es_client import ensure_index, index_name

logger = get_logger(__name__)

_default_config = RAGConfig()


def index_documents(
    domain: str,
    docs: Optional[List[Document]] = None,
    config: RAGConfig = None,
):
    """
    为指定法域构建向量索引。
    如果 docs 为 None，则从 documents/<domain>/ 自动加载。
    """
    cfg = config or _default_config

    # 加载文档
    if docs is None:
        docs = load_documents(domain, config=cfg)

    if not docs:
        raise ValueError(
            f"法域 '{domain}' 没有文档。"
            f"请在 {cfg.documents_dir / domain} 目录下添加法律文档。"
        )

    # 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"[RAG] {domain}: 加载 {len(docs)} 篇文档，切分为 {len(chunks)} 个片段")

    # 确保 index 存在
    ensure_index(domain, config=cfg)

    # embedding
    embedder = cfg.get_embedder()
    logger.info(f"[RAG] {domain}: 开始生成 embedding (provider={cfg.embedding_provider})")
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    logger.info(f"[RAG] {domain}: embedding 生成完成 ({len(embeddings)} 条)")

    # 批量写入 ES
    es = cfg.get_es_client()
    index = cfg.index_name(domain)

    from elasticsearch.helpers import bulk

    actions = []
    for chunk, vector in zip(chunks, embeddings):
        actions.append({
            "_index": index,
            "_source": {
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "domain": domain,
                "embedding": vector,
            },
        })

    success, errors = bulk(es, actions, refresh=True)
    if errors:
        logger.error(f"[ES] {domain}: 索引失败 {len(errors)} 条")
    else:
        logger.info(f"[ES] {domain}: 索引 {success} 条文档")

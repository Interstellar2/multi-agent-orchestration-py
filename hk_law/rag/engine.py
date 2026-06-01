"""
RAG 引擎 —— Elasticsearch 版
为每个法域 Agent 提供独立的向量检索能力。

设计：
  - 每个法域对应一个 ES index: hk_law_<domain>
  - Embedding: 阿里云百炼 text-embedding-v3（默认）或 Ollama 本地模型
  - Rerank: 阿里云百炼 gte-rerank（默认）或本地余弦相似度 fallback
  - ES 地址从环境变量读取，默认 localhost:9200

环境变量切换 provider：
  EMBEDDING_PROVIDER=bailian|ollama   默认 bailian
  RERANK_PROVIDER=bailian|local       默认 bailian
  OLLAMA_HOST=http://localhost:11434  Ollama 服务地址
  OLLAMA_EMBED_MODEL=nomic-embed-text Ollama embedding 模型
  OLLAMA_EMBED_DIMS=768               Ollama 模型维度
"""
import os
from pathlib import Path
from typing import List, Optional

from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.utils.logger import get_logger

logger = get_logger(__name__)

# ES 连接配置
ES_HOST = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX_PREFIX = "hk_law"

# Provider 选择
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bailian")
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "bailian")


def _safe_int_env(var_name: str, default: int) -> int:
    """安全读取整型环境变量，无效时返回默认值"""
    try:
        return int(os.getenv(var_name, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {var_name} 格式无效，使用默认值 {default}")
        return default


# Embedding 维度（按 provider 给默认值）
_DEFAULT_DIMS = {"bailian": 1536, "ollama": _safe_int_env("OLLAMA_EMBED_DIMS", 768)}
EMBEDDING_DIMS = _safe_int_env("EMBEDDING_DIMS", _DEFAULT_DIMS.get(EMBEDDING_PROVIDER, 1536))

# 文档根目录
# 文档根目录
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"


def get_embedder():
    """获取当前配置的 embedding 实例"""
    if EMBEDDING_PROVIDER == "ollama":
        from core.llm.providers.ollama_embedding import get_ollama_embedding

        return get_ollama_embedding()
    else:
        from core.llm.providers.bailian_embedding import get_bailian_embedding

        return get_bailian_embedding()


def get_reranker(top_n: int = 5):
    """获取当前配置的 reranker 实例"""
    if RERANK_PROVIDER == "local":
        from core.llm.providers.local_rerank import LocalReranker

        return LocalReranker(top_n=top_n, embedder=get_embedder())
    else:
        from core.llm.providers.bailian_rerank import get_bailian_reranker

        return get_bailian_reranker(top_n=top_n)


def _get_es_client() -> Elasticsearch:
    """获取 ES 客户端"""
    return Elasticsearch([ES_HOST])


def _index_name(domain: str) -> str:
    return f"{ES_INDEX_PREFIX}_{domain}"


def _ensure_index(domain: str):
    """确保法域 index 存在，不存在则创建"""
    es = _get_es_client()
    index = _index_name(domain)

    if not es.indices.exists(index=index):
        es.indices.create(
            index=index,
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "domain": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIMS,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                },
            },
        )
        logger.info(f"[ES] 创建 index: {index} dims={EMBEDDING_DIMS}")


def load_documents(domain: str) -> List[Document]:
    """
    从 documents/<domain>/ 目录加载法律文档。
    支持 .txt, .md, .pdf（需安装 PyPDF2）
    """
    domain_dir = DOCUMENTS_DIR / domain
    if not domain_dir.exists():
        return []

    docs: List[Document] = []

    for ext, loader_cls in [
        ("*.txt", "langchain_community.document_loaders.TextLoader"),
        ("*.md", "langchain_community.document_loaders.TextLoader"),
    ]:
        for file_path in domain_dir.glob(ext):
            try:
                module_path, class_name = loader_cls.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                loader = getattr(module, class_name)(str(file_path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata.update({
                        "source": str(file_path.name),
                        "domain": domain,
                    })
                docs.extend(loaded)
            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {e}")

    # PDF
    try:
        from langchain_community.document_loaders import PyPDFLoader
        for file_path in domain_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata.update({
                        "source": str(file_path.name),
                        "domain": domain,
                    })
                docs.extend(loaded)
            except Exception as e:
                logger.warning(f"加载 PDF 失败 {file_path}: {e}")
    except ImportError:
        logger.debug("PyPDFLoader 未安装，跳过 PDF 加载")

    return docs


def index_documents(domain: str, docs: Optional[List[Document]] = None):
    """
    为指定法域构建向量索引。
    如果 docs 为 None，则从 documents/<domain>/ 自动加载。
    """
    # 加载文档
    if docs is None:
        docs = load_documents(domain)

    if not docs:
        raise ValueError(
            f"法域 '{domain}' 没有文档。"
            f"请在 {DOCUMENTS_DIR / domain} 目录下添加法律文档。"
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
    _ensure_index(domain)

    # embedding
    embedder = get_embedder()
    logger.info(f"[RAG] {domain}: 开始生成 embedding (provider={EMBEDDING_PROVIDER})")
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    logger.info(f"[RAG] {domain}: embedding 生成完成 ({len(embeddings)} 条)")

    # 批量写入 ES
    es = _get_es_client()
    index = _index_name(domain)

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


def search(
    domain: str,
    query: str,
    top_k: int = 10,
    rerank_top_n: int = 5,
    use_rerank: bool = True,
) -> List[Document]:
    """
    检索流程：
    1. ES kNN 向量检索 top_k 条
    2. （可选）百炼 rerank 重排序，取 rerank_top_n 条

    返回: Document 列表
    """
    es = _get_es_client()
    index = _index_name(domain)

    # 获取 query embedding
    embedder = get_embedder()
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
        logger.debug(f"[RAG] 开始 rerank (provider={RERANK_PROVIDER})")
        reranker = get_reranker(top_n=rerank_top_n)
        texts = [d.page_content for d in docs]
        reranked = reranker.rerank(query, texts)
        # 按 rerank 结果重新排序
        reranked_docs = []
        for idx, score in reranked:
            d = docs[idx]
            d.metadata["rerank_score"] = score
            reranked_docs.append(d)
        docs = reranked_docs

    return docs[:rerank_top_n]


def delete_index(domain: str):
    """删除法域 index（用于重建）"""
    es = _get_es_client()
    index = _index_name(domain)
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
        logger.info(f"[ES] 删除 index: {index}")


def check_es_health() -> bool:
    """检查 ES 连接是否正常"""
    try:
        es = _get_es_client()
        return es.ping()
    except Exception:
        return False


class DomainRetriever:
    """
    法域专用检索器（兼容旧接口）。
    底层从 Chroma 切换到 Elasticsearch。
    """

    def __init__(self, domain: str, top_k: int = 5):
        self.domain = domain
        self.top_k = top_k

    def search(self, query: str) -> List[Document]:
        """检索与 query 相关的法律条文片段"""
        return search(
            domain=self.domain,
            query=query,
            top_k=self.top_k * 2,  # 先多召回一些，rerank 后取 top_k
            rerank_top_n=self.top_k,
            use_rerank=True,
        )

    def search_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """检索并返回相似度分数"""
        docs = self.search(query)
        return [(d, d.metadata.get("rerank_score", d.metadata.get("es_score", 0))) for d in docs]

    def rebuild(self):
        """重新构建索引（文档更新后调用）"""
        delete_index(self.domain)
        index_documents(self.domain)

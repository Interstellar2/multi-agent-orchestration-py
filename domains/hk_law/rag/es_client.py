"""
Elasticsearch 客户端与索引管理
"""
from elasticsearch import Elasticsearch
from core.utils.logger import get_logger
from domains.hk_law.rag.config import RAGConfig

logger = get_logger(__name__)

_default_config = RAGConfig()


def get_es_client(config: RAGConfig = None) -> Elasticsearch:
    """获取 ES 客户端"""
    cfg = config or _default_config
    return cfg.get_es_client()


def index_name(domain: str, config: RAGConfig = None) -> str:
    cfg = config or _default_config
    return cfg.index_name(domain)


def ensure_index(domain: str, config: RAGConfig = None):
    """确保法域 index 存在，不存在则创建"""
    cfg = config or _default_config
    es = cfg.get_es_client()
    index = cfg.index_name(domain)

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
                            "dims": cfg.embedding_dims,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                },
            },
        )
        logger.info(f"[ES] 创建 index: {index} dims={cfg.embedding_dims}")


def delete_index(domain: str, config: RAGConfig = None):
    """删除法域 index（用于重建）"""
    cfg = config or _default_config
    es = cfg.get_es_client()
    index = cfg.index_name(domain)
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
        logger.info(f"[ES] 删除 index: {index}")


def check_es_health(config: RAGConfig = None) -> bool:
    """检查 ES 连接是否正常"""
    cfg = config or _default_config
    try:
        es = cfg.get_es_client()
        return es.ping()
    except Exception:
        return False

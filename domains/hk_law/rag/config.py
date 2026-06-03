"""
RAG 配置对象

将原本分散在 engine.py 中的模块级全局变量（embedding provider、rerank provider、
ES 地址等）封装为可实例化的配置类，支持同一进程中多组不同配置并存。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_int_env(var_name: str, default: int) -> int:
    """安全读取整型环境变量，无效时返回默认值"""
    try:
        return int(os.getenv(var_name, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {var_name} 格式无效，使用默认值 {default}")
        return default


@dataclass
class RAGConfig:
    """
    RAG 引擎配置。

    默认从环境变量读取，也可手动构造以覆盖：
        config = RAGConfig(
            embedding_provider="ollama",
            es_host="http://other-es:9200",
            index_prefix="custom_law",
        )
    """

    # ES 连接
    es_host: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    index_prefix: str = "hk_law"

    # Embedding 配置
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "bailian"))
    embedding_dims: int = field(default=0)  # 0 表示自动推导

    # Ollama 专用
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    ollama_embed_dims: int = field(default_factory=lambda: _safe_int_env("OLLAMA_EMBED_DIMS", 768))

    # Rerank 配置
    rerank_provider: str = field(default_factory=lambda: os.getenv("RERANK_PROVIDER", "bailian"))

    # 文档目录（默认相对本文件）
    documents_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "documents")

    def __post_init__(self):
        if self.embedding_dims == 0:
            defaults = {"bailian": 1536, "ollama": self.ollama_embed_dims}
            env_dims = os.getenv("EMBEDDING_DIMS")
            if env_dims:
                self.embedding_dims = _safe_int_env("EMBEDDING_DIMS", defaults.get(self.embedding_provider, 1536))
            else:
                self.embedding_dims = defaults.get(self.embedding_provider, 1536)

    def get_embedder(self):
        """获取当前配置的 embedding 实例"""
        if self.embedding_provider == "ollama":
            from core.llm.providers.ollama_embedding import get_ollama_embedding

            return get_ollama_embedding()
        else:
            from core.llm.providers.bailian_embedding import get_bailian_embedding

            return get_bailian_embedding()

    def get_reranker(self, top_n: int = 5):
        """获取当前配置的 reranker 实例"""
        if self.rerank_provider == "local":
            from core.llm.providers.local_rerank import LocalReranker

            return LocalReranker(top_n=top_n, embedder=self.get_embedder())
        else:
            from core.llm.providers.bailian_rerank import get_bailian_reranker

            return get_bailian_reranker(top_n=top_n)

    def get_es_client(self):
        """获取 ES 客户端"""
        from elasticsearch import Elasticsearch

        return Elasticsearch([self.es_host])

    def index_name(self, domain: str) -> str:
        return f"{self.index_prefix}_{domain}"

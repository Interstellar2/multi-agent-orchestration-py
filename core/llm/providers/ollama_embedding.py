"""
Ollama 本地 Embedding
通过调用 Ollama HTTP API 实现，无需额外 SDK 依赖。

推荐模型:
  - nomic-embed-text  (768 维，轻量，推荐)
  - mxbai-embed-large (1024 维，效果更好)
  - bge-m3            (1024 维，多语言支持好)

使用前确保 Ollama 已启动并拉取模型:
  ollama pull nomic-embed-text
"""

import os
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
_DEFAULT_DIMS = int(os.getenv("OLLAMA_EMBED_DIMS", "768"))


class OllamaEmbeddings(Embeddings):
    """
    Ollama Embedding 封装，兼容 LangChain Embeddings 接口。

    用法:
        from core.llm.providers.ollama_embedding import OllamaEmbeddings
        embedder = OllamaEmbeddings()
        vectors = embedder.embed_documents(["hello", "world"])
    """

    def __init__(self, host: str = None, model: str = None, dims: int = None):
        self.host = (host or _OLLAMA_HOST).rstrip("/")
        self.model = model or _DEFAULT_MODEL
        self.dims = dims or _DEFAULT_DIMS
        self._api_url = f"{self.host}/api/embeddings"

    def _embed(self, text: str) -> List[float]:
        """调用 Ollama embedding API"""
        resp = requests.post(
            self._api_url,
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "embedding" not in data:
            raise RuntimeError(f"Ollama embedding 返回异常: {data}")
        return data["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入。Ollama 本地调用，逐条请求即可。"""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self._embed(text)


# 全局单例
_ollama_embedding_instance: Optional[OllamaEmbeddings] = None


def get_ollama_embedding() -> OllamaEmbeddings:
    """获取全局 Ollama embedding 实例"""
    global _ollama_embedding_instance
    if _ollama_embedding_instance is None:
        _ollama_embedding_instance = OllamaEmbeddings()
    return _ollama_embedding_instance

"""
阿里云百炼 Embedding
使用百炼 text-embedding-v3 模型。

API 文档: https://help.aliyun.com/zh/model-studio/developer-reference/text-embedding-api-details
"""
import os
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

from core.llm.config import config, ProviderConfig

# 读取配置
_cfg = config.providers.get("bailian") or ProviderConfig()
_api_key = _cfg.api_key or os.getenv("DASHSCOPE_API_KEY", "")

_DEFAULT_MODEL = "text-embedding-v3"
_API_URL = "https://dashscope.aliyuncs.com/api/v1/embeddings"


class BailianEmbeddings(Embeddings):
    """
    百炼 Embedding 封装，兼容 LangChain Embeddings 接口。

    用法:
        from llm.providers.bailian_embedding import BailianEmbeddings
        embedder = BailianEmbeddings()
        texts = ["hello world", "你好世界"]
        vectors = embedder.embed_documents(texts)
    """

    def __init__(self, api_key: str = None, model: str = _DEFAULT_MODEL):
        self.api_key = api_key or _api_key
        if not self.api_key:
            raise ValueError("百炼 API Key 未配置。请在 .env 中设置 DASHSCOPE_API_KEY")
        self.model = model
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call(self, texts: List[str]) -> List[List[float]]:
        """调用百炼 embedding API"""
        payload = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {"text_type": "document"},
        }
        resp = requests.post(_API_URL, headers=self._headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "output" not in data or "embeddings" not in data["output"]:
            raise RuntimeError(f"Embedding API 返回异常: {data}")

        # 按 text_index 排序返回
        embeddings = sorted(data["output"]["embeddings"], key=lambda x: x["text_index"])
        return [e["embedding"] for e in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档。百炼单次最多 25 条，这里自动分批。"""
        all_embeddings = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._call(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        payload = {
            "model": self.model,
            "input": {"texts": [text]},
            "parameters": {"text_type": "query"},
        }
        resp = requests.post(_API_URL, headers=self._headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["output"]["embeddings"][0]["embedding"]


# 全局单例
_bailian_embedding_instance: Optional[BailianEmbeddings] = None


def get_bailian_embedding() -> BailianEmbeddings:
    """获取全局百炼 embedding 实例"""
    global _bailian_embedding_instance
    if _bailian_embedding_instance is None:
        _bailian_embedding_instance = BailianEmbeddings()
    return _bailian_embedding_instance

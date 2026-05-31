"""
阿里云百炼 Rerank
使用百炼 gte-rerank 模型对检索结果重排序。

API 文档: https://help.aliyun.com/zh/model-studio/developer-reference/rerank-api-details
"""
import os
from typing import List, Optional, Tuple

import requests

from core.llm.config import config, ProviderConfig

# 读取配置
_cfg = config.providers.get("bailian") or ProviderConfig()
_api_key = _cfg.api_key or os.getenv("DASHSCOPE_API_KEY", "")

_DEFAULT_MODEL = "gte-rerank"
_API_URL = "https://dashscope.aliyuncs.com/api/v1/rerank"


class BailianReranker:
    """
    百炼 Rerank 封装。

    用法:
        from llm.providers.bailian_rerank import BailianReranker
        reranker = BailianReranker()
        scores = reranker.rerank(query="解雇赔偿", documents=[doc1, doc2, ...])
        # 返回按分数排序的 (document_index, score) 列表
    """

    def __init__(self, api_key: str = None, model: str = _DEFAULT_MODEL, top_n: int = 5):
        self.api_key = api_key or _api_key
        if not self.api_key:
            raise ValueError("百炼 API Key 未配置。请在 .env 中设置 DASHSCOPE_API_KEY")
        self.model = model
        self.top_n = top_n
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def rerank(
        self, query: str, documents: List[str]
    ) -> List[Tuple[int, float]]:
        """
        对文档列表进行重排序。

        返回: [(document_index, score), ...] 按 score 降序排列
        """
        if not documents:
            return []

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {"top_n": min(self.top_n, len(documents)), "return_documents": False},
        }
        resp = requests.post(_API_URL, headers=self._headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "output" not in data or "results" not in data["output"]:
            raise RuntimeError(f"Rerank API 返回异常: {data}")

        results = data["output"]["results"]
        # results 格式: [{"index": 0, "relevance_score": 0.95}, ...]
        sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        return [(r["index"], r["relevance_score"]) for r in sorted_results]


# 全局单例
_bailian_rerank_instance: Optional[BailianReranker] = None


def get_bailian_reranker(top_n: int = 5) -> BailianReranker:
    """获取全局百炼 rerank 实例"""
    global _bailian_rerank_instance
    if _bailian_rerank_instance is None:
        _bailian_rerank_instance = BailianReranker(top_n=top_n)
    return _bailian_rerank_instance

import json
import math
import os
from typing import List, Dict, Any, Optional

from biz.utils.log import logger


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    """
    简易本地向量库，默认读取 data/vector_store.json，结构示例:
    {
      "model": "text-embedding-3-small",
      "dimension": 1536,
      "items": [
        {"id":"pkg:numpy", "name":"numpy", "text":"Numpy 的常见约定...", "embedding":[0.01, ...]},
        {"id":"func:requests.get", "name":"requests.get", "text":"Requests GET 使用注意事项...", "embedding":[-0.02, ...]}
      ]
    }
    """
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("VECTOR_STORE_PATH", "data/vector_store.json")
        self.data: Dict[str, Any] = {"items": []}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                logger.warning(f"向量库文件不存在，将跳过检索: {self.path}")
        except Exception as e:
            logger.warning(f"加载向量库失败: {e}")
            self.data = {"items": []}

    def _embed_terms(self, terms: List[str]) -> Optional[List[float]]:
        """
        使用 OpenAI Embedding 对 terms 做平均向量，若不可用则返回 None 表示降级。
        """
        try:
            from biz.llm.embeddings import EmbeddingProvider
            provider = EmbeddingProvider()
            # 对每个 term 做 embedding，取平均作为查询向量
            vecs = provider.get_embeddings(terms)
            if not vecs:
                return None
            dim = len(vecs[0])
            avg = [0.0] * dim
            for v in vecs:
                for i, x in enumerate(v):
                    avg[i] += x
            n = float(len(vecs))
            avg = [x / n for x in avg]
            return avg
        except Exception as e:
            logger.warning(f"Embedding 失败，降级为关键词匹配: {e}")
            return None

    def _keyword_match(self, terms: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        不使用 embedding 时的简单降级策略：按名称包含计分。
        """
        scored = []
        items = self.data.get("items", [])
        for it in items:
            name = (it.get("name") or "").lower()
            score = 0
            for t in terms:
                t = (t or "").lower()
                if t and t in name:
                    score += 1
            if score > 0:
                scored.append({"id": it.get("id"), "name": it.get("name"), "text": it.get("text"), "score": float(score)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_similar(self, terms: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        terms = [t for t in terms if t]
        if not terms:
            return []
        items = self.data.get("items", [])
        if not items:
            return []

        query_vec = self._embed_terms(terms)
        if query_vec is None:
            # 降级：关键词匹配
            return self._keyword_match(terms, top_k)

        # 正常：基于余弦相似度排序
        results = []
        for it in items:
            emb = it.get("embedding")
            if isinstance(emb, list) and emb:
                score = _cosine_sim(query_vec, emb)
                results.append({
                    "id": it.get("id"),
                    "name": it.get("name"),
                    "text": it.get("text"),
                    "score": float(score),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
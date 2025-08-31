import os
from typing import List, Optional

from openai import OpenAI


class EmbeddingProvider:
    """
    简单的 Embedding 提供器：使用 OpenAI Embeddings。
    需要环境变量：
      - OPENAI_API_KEY
      - OPENAI_API_BASE_URL（可选，默认 https://api.openai.com）
      - EMBEDDING_MODEL（可选，默认 text-embedding-3-small）
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 OPENAI_API_KEY，无法使用向量检索。")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        res = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        # OpenAI SDK 返回与 input 顺序一致
        return [d.embedding for d in res.data]
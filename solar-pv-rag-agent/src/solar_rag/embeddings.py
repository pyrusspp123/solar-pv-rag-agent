"""Embedding 后端：本地 sentence-transformers 或 OpenAI 兼容接口。"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import Config


def get_embeddings(cfg: Config) -> Embeddings:
    """返回一个 LangChain Embeddings 实例。

    - local: 使用 sentence-transformers 本地模型（默认，无需 API key）
    - openai: 使用 OpenAI 兼容 embedding 接口
    """
    if cfg.embedding_backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=cfg.openai_api_key or None,
            base_url=cfg.openai_base_url or None,
        )

    # 默认本地模型
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    return HuggingFaceBgeEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

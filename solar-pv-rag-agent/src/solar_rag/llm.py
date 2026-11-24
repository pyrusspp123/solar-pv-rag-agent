"""LLM 后端：OpenAI 兼容接口（可接 DeepSeek / Qwen / OpenAI 等）。"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Config


def get_llm(cfg: Config) -> BaseChatModel:
    if not cfg.llm_enabled:
        raise RuntimeError(
            "未配置 LLM。请复制 .env.example 为 .env 并填写 OPENAI_API_KEY（可接 DeepSeek/Qwen/OpenAI 兼容接口）。"
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        temperature=cfg.llm_temperature,
    )

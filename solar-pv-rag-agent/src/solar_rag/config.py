"""集中配置：从环境变量 / .env 读取运行参数。

用法：
    from solar_rag.config import get_config
    cfg = get_config()
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（solar-pv-rag-agent/）
ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """加载 .env 文件（若存在），不覆盖已有环境变量。"""
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)


@dataclass
class Config:
    # LLM（OpenAI 兼容接口）
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Embedding
    embedding_backend: str = "local"  # local | openai
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # 检索
    chunk_size: int = 500
    chunk_overlap: int = 60
    top_k: int = 6
    hybrid_weight: float = 0.5  # 向量检索与 BM25 的融合权重（0=纯BM25, 1=纯向量）

    # 数据路径
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / "data")
    raw_pdfs_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "raw_pdfs")
    index_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "vector_index")
    materials_csv: Path = field(default_factory=lambda: ROOT_DIR / "data" / "materials.csv")

    # Materials Project（可选）
    materials_project_api_key: str = ""

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_config() -> Config:
    _load_env()
    return Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "local"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        materials_project_api_key=os.getenv("MATERIALS_PROJECT_API_KEY", ""),
    )

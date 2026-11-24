#!/usr/bin/env python3
"""构建检索索引：加载 data/raw_pdfs 下文档 → 切分 → embedding → 构建 FAISS + BM25。

用法：
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_rag.config import get_config  # noqa: E402
from solar_rag.embeddings import get_embeddings  # noqa: E402
from solar_rag.index import build_index  # noqa: E402
from solar_rag.loader import load_documents  # noqa: E402
from solar_rag.splitter import split_documents  # noqa: E402


def main() -> None:
    cfg = get_config()

    print(f"[1/4] 加载文档：{cfg.raw_pdfs_dir}")
    docs = load_documents(cfg.raw_pdfs_dir)
    print(f"      读取到 {len(docs)} 个文档片段")

    print(f"[2/4] 切分（chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}）")
    chunks = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
    print(f"      得到 {len(chunks)} 个 chunk")

    print(f"[3/4] 初始化 embedding（{cfg.embedding_backend}: {cfg.embedding_model}）")
    embeddings = get_embeddings(cfg)

    print("[4/4] 构建并保存 FAISS + BM25 索引...")
    n = build_index(chunks, embeddings, cfg.index_dir)
    print(f"完成！索引已保存到 {cfg.index_dir}（共 {n} 个 chunk）")


if __name__ == "__main__":
    main()

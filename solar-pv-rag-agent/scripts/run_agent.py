#!/usr/bin/env python3
"""交互式运行 AI Agent（需先运行 build_index.py 构建索引，并配置 LLM）。

用法：
    python scripts/run_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_rag.agent import build_agent  # noqa: E402
from solar_rag.config import get_config  # noqa: E402
from solar_rag.embeddings import get_embeddings  # noqa: E402
from solar_rag.index import load_index  # noqa: E402
from solar_rag.llm import get_llm  # noqa: E402
from solar_rag.retrieval import HybridRetriever  # noqa: E402


def main() -> None:
    cfg = get_config()
    llm = get_llm(cfg)
    embeddings = get_embeddings(cfg)

    print("加载索引...")
    vectorstore, chunks, bm25 = load_index(cfg.index_dir, embeddings)
    retriever = HybridRetriever(vectorstore, chunks, bm25, cfg.hybrid_weight)
    agent = build_agent(llm, retriever)

    print("AI Agent 已就绪。输入问题开始，输入 exit / 退出 结束。")
    chat_history: list[tuple[str, str]] = []
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "退出"}:
            break

        result = agent.invoke({"input": q, "chat_history": chat_history})
        chat_history.append(("human", q))
        chat_history.append(("ai", str(result["output"])))
        print("\n助手：", result["output"])


if __name__ == "__main__":
    main()

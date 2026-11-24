#!/usr/bin/env python3
"""RAGAS 评测：读取 data/eval_set.json 问答集，跑一遍 RAG 生成答案后评估。

用法：
    pip install -r requirements-eval.txt
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from solar_rag.config import get_config  # noqa: E402
from solar_rag.embeddings import get_embeddings  # noqa: E402
from solar_rag.evaluation import run_ragas_evaluation  # noqa: E402
from solar_rag.index import load_index  # noqa: E402
from solar_rag.llm import get_llm  # noqa: E402
from solar_rag.retrieval import HybridRetriever  # noqa: E402


def _generate(llm, question: str, contexts: list[str]) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是光伏材料研究助手。请仅根据给定的上下文回答问题，并在句末用括号注明来源编号；信息不足时如实说明。"),
            ("human", "上下文：\n\n{context}\n\n问题：{question}"),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke(
        {"context": "\n\n".join(contexts), "question": question}
    )
    return resp.content if hasattr(resp, "content") else str(resp)


def main() -> None:
    cfg = get_config()
    eval_file = cfg.data_dir / "eval_set.json"
    if not eval_file.exists():
        print(f"未找到评测集：{eval_file}")
        return

    qa = json.loads(eval_file.read_text(encoding="utf-8"))
    print(f"评测集共 {len(qa)} 条问题")

    llm = get_llm(cfg)
    embeddings = get_embeddings(cfg)
    vectorstore, chunks, bm25 = load_index(cfg.index_dir, embeddings)
    retriever = HybridRetriever(vectorstore, chunks, bm25, cfg.hybrid_weight)

    # 跑一遍 RAG：检索 + 生成
    samples = []
    for i, item in enumerate(qa, 1):
        contexts = [
            d.page_content for d in retriever.retrieve(item["question"], top_k=cfg.top_k)
        ]
        answer = _generate(llm, item["question"], contexts)
        samples.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.get("ground_truth", ""),
            }
        )
        print(f"  [{i}/{len(qa)}] {item['question'][:36]}...")

    print("开始 RAGAS 评测（faithfulness / answer relevancy / context precision / recall）...")
    result = run_ragas_evaluation(samples, llm, embeddings)

    print("\n平均指标：")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    out = cfg.data_dir / "eval_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()

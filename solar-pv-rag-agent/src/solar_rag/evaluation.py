"""RAGAS 评测：评估 RAG 问答的忠实度、相关性、上下文精确率与召回率。"""
from __future__ import annotations

from typing import Any


def run_ragas_evaluation(samples: list[dict], llm: Any, embeddings: Any) -> dict:
    """对 RAG 问答样本做 RAGAS 评测。

    参数
    ----
    samples : list[dict]
        每个元素形如：
        {
            "question": str,          # 用户问题
            "answer": str,            # 模型生成的回答
            "contexts": list[str],    # 检索到的上下文片段
            "ground_truth": str,      # 参考答案（可选）
        }
    llm / embeddings : LangChain 的 LLM 与 Embeddings 实例

    返回
    ----
    dict : {"scores": [每行指标...], "metrics": 平均分}
    """
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "缺少 RAGAS 评测依赖，请先执行：pip install -r requirements-eval.txt"
        ) from exc

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    sample_objs = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=list(s.get("contexts", [])),
            reference=s.get("ground_truth", ""),
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=sample_objs)

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    return {
        "scores": df.to_dict(orient="records"),
        "metrics": df.mean(numeric_only=True).to_dict(),
    }

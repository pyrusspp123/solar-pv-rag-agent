"""混合检索：向量检索（FAISS）与稀疏检索（BM25）加权融合。"""
from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .index import tokenize


def _minmax_norm(mapping: dict) -> dict:
    """对分数做 min-max 归一化到 [0, 1]。"""
    if not mapping:
        return {}
    values = list(mapping.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in mapping}
    return {k: (v - lo) / (hi - lo) for k, v in mapping.items()}


class HybridRetriever:
    """向量 + BM25 混合检索器。

    - vectorstore: FAISS 实例（已加载）
    - chunks: 与索引一致的 Document 列表（下标即 chunk_id）
    - bm25: BM25Okapi 实例
    - weight: 向量检索权重（0~1，越大越偏向向量检索）
    """

    def __init__(self, vectorstore, chunks: list[Document], bm25: BM25Okapi, weight: float = 0.5):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.bm25 = bm25
        self.weight = weight

    def retrieve(self, query: str, top_k: int = 6) -> list[Document]:
        cand = min(top_k * 2, len(self.chunks))

        # 1. 向量检索：FAISS 返回 L2 距离（越小越相似），取负转为“越大越好”
        vec_hits = self.vectorstore.similarity_search_with_score(query, k=cand)
        vec_scores: dict[int, float] = {}
        for doc, score in vec_hits:
            cid = doc.metadata.get("chunk_id")
            if cid is not None:
                vec_scores[int(cid)] = -float(score)

        # 2. BM25 检索：分数越大越相关
        toks = tokenize(query)
        bm25_arr = self.bm25.get_scores(toks)
        top_idx = sorted(range(len(bm25_arr)), key=lambda i: bm25_arr[i], reverse=True)[:cand]
        bm25_scores: dict[int, float] = {int(i): float(bm25_arr[i]) for i in top_idx}

        # 3. 分别归一化后加权融合
        n_vec = _minmax_norm(vec_scores)
        n_bm25 = _minmax_norm(bm25_scores)
        fused: dict[int, float] = {}
        for cid in set(n_vec) | set(n_bm25):
            fused[cid] = self.weight * n_vec.get(cid, 0.0) + (1.0 - self.weight) * n_bm25.get(cid, 0.0)

        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [self.chunks[cid] for cid in ranked]

    def format(self, query: str, top_k: int = 6) -> str:
        """返回给 LLM 使用的检索上下文文本。"""
        docs = self.retrieve(query, top_k)
        blocks = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page", "")
            blocks.append(f"[{i}] 来源：{src}{' 第' + str(page) + '页' if page else ''}\n{d.page_content.strip()}")
        return "\n\n".join(blocks)

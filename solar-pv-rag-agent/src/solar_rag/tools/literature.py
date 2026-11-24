"""文献检索工具：基于本地 RAG 混合检索器查询光伏材料知识库。"""
from __future__ import annotations

from langchain_core.tools import tool

_retriever = None


def set_retriever(retriever) -> None:
    global _retriever
    _retriever = retriever


@tool
def literature_search(query: str) -> str:
    """在光伏材料文献知识库中检索与 query 相关的内容，返回带来源的文本片段。

    当需要查找文献中的具体知识（如材料组分、制备工艺、稳定性、效率数据、
    机理等）时调用此工具。参数 query 为自然语言检索语句。
    """
    if _retriever is None:
        return "错误：检索器尚未初始化，请先构建索引。"
    return _retriever.format(query, top_k=6)

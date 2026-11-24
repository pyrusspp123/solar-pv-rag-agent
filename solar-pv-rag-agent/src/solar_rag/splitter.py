"""文本切分：将长文档切分为语义相对完整的 chunk。"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_splitter(chunk_size: int = 500, chunk_overlap: int = 60) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""],
        length_function=len,
    )


def split_documents(docs: list[Document], chunk_size: int = 500, chunk_overlap: int = 60) -> list[Document]:
    splitter = make_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)

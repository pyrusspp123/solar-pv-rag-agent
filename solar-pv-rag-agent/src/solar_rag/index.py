"""索引构建与加载：FAISS 向量索引 + BM25 稀疏索引，二者共用同一批 chunk。"""
from __future__ import annotations

import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """中英混合分词：中文用 jieba，英文按空白切分。"""
    import jieba

    jieba.setLogLevel(60)  # 关闭 jieba 的加载日志
    tokens: list[str] = []
    for word in jieba.cut(text.lower()):
        word = word.strip()
        if word:
            tokens.append(word)
    return tokens


def build_index(docs: list[Document], embeddings: Embeddings, index_dir: Path) -> int:
    """构建并保存 FAISS 与 BM25 索引，返回 chunk 数量。"""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # 给每个 chunk 打上全局编号，供检索结果回映射
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(index_dir / "faiss"))

    tokenized = [tokenize(d.page_content) for d in docs]
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(
            {
                "texts": [d.page_content for d in docs],
                "tokenized": tokenized,
                "metadatas": [d.metadata for d in docs],
            },
            f,
        )
    return len(docs)


def load_index(index_dir: Path, embeddings: Embeddings):
    """加载 FAISS 与 BM25 索引，返回 (vectorstore, chunks, bm25)。"""
    index_dir = Path(index_dir)
    faiss_dir = index_dir / "faiss"
    bm25_file = index_dir / "bm25.pkl"
    if not faiss_dir.exists() or not bm25_file.exists():
        raise FileNotFoundError(f"索引不存在，请先运行 scripts/build_index.py：{index_dir}")

    vectorstore = FAISS.load_local(
        str(faiss_dir), embeddings, allow_dangerous_deserialization=True
    )
    with open(bm25_file, "rb") as f:
        data = pickle.load(f)

    bm25 = BM25Okapi(data["tokenized"])
    chunks = [
        Document(page_content=t, metadata=m)
        for t, m in zip(data["texts"], data["metadatas"])
    ]
    return vectorstore, chunks, bm25

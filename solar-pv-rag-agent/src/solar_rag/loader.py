"""文档加载：从目录读取 PDF / TXT / Markdown，返回 LangChain Document 列表。"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": i + 1},
                )
            )
    return docs


def load_documents(directory: Path) -> list[Document]:
    """递归加载目录下所有 PDF / TXT / MD 文档。"""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")

    docs: list[Document] = []
    supported = {".pdf", ".txt", ".md", ".markdown"}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            if path.suffix.lower() == ".pdf":
                docs.extend(_load_pdf(path))
            else:
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = path.read_text(encoding="gbk", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": str(path)}))

    if not docs:
        raise ValueError(f"未在 {directory} 中找到任何支持的文档（.pdf/.txt/.md）")
    return docs

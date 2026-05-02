import logging
import re
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from pypdf import PdfReader

from app.config import Settings, get_settings
from app.utils import (
    build_embed_model,
    clean_text,
    detect_section_title,
    ensure_project_dirs,
    get_chroma_collection,
    list_source_files,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestStats:
    documents_loaded: int
    chunks_indexed: int
    storage_path: str


def load_documents(raw_docs_dir: Path) -> list[Document]:
    """Load supported source files and attach source metadata before chunking."""

    documents: list[Document] = []
    for path in list_source_files(raw_docs_dir):
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        elif path.suffix.lower() == ".txt":
            document = _load_text(path)
            if document is not None:
                documents.append(document)
    return documents


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            logger.warning("Skipping empty page %s in %s", page_index, path.name)
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "filename": path.name,
                    "page": page_index,
                    "section": detect_section_title(text) or "",
                    "file_type": "pdf",
                },
            )
        )

    return documents


def _load_text(path: Path) -> Document | None:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        logger.warning("Skipping empty text file %s", path.name)
        return None

    return Document(
        text=text,
        metadata={
            "filename": path.name,
            "page": 0,
            "section": detect_section_title(text) or "",
            "file_type": "txt",
        },
    )


def ingest_documents(settings: Settings | None = None) -> IngestStats:
    """Chunk documents, embed nodes, and persist the Chroma-backed vector index."""

    settings = settings or get_settings()
    ensure_project_dirs(settings)

    documents = load_documents(settings.raw_docs_dir)
    if not documents:
        raise FileNotFoundError(
            f"No PDF or .txt files found in {settings.raw_docs_dir}. "
            "Add 10-20 marine biology source files and run ingestion again."
        )

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    for index, node in enumerate(nodes, start=1):
        chunk_id = f"chunk_{index:04d}"
        filename = str(node.metadata.get("filename", "source"))
        node.metadata["chunk_id"] = chunk_id
        node.metadata["section"] = str(node.metadata.get("section") or "")
        node.metadata["page"] = int(node.metadata.get("page") or 0)
        node.id_ = _stable_node_id(filename, chunk_id)

    collection = get_chroma_collection(settings, reset=True)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = build_embed_model(settings)

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info(
        "Indexed %s chunks from %s document pages/files into %s",
        len(nodes),
        len(documents),
        settings.chroma_path,
    )
    return IngestStats(
        documents_loaded=len(documents),
        chunks_indexed=len(nodes),
        storage_path=str(settings.chroma_path),
    )


def _stable_node_id(filename: str, chunk_id: str) -> str:
    safe_filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", filename)
    return f"{safe_filename}:{chunk_id}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = ingest_documents()
    print(
        f"Ingested {stats.chunks_indexed} chunks from {stats.documents_loaded} "
        f"document pages/files into {stats.storage_path}"
    )


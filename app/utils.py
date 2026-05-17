import logging
import re
from pathlib import Path
from typing import Any

import chromadb

from app.config import Settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def clean_text(text: str) -> str:
    """Normalize extracted document text while preserving paragraph breaks."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section_title(text: str) -> str | None:
    """Best-effort section/title detection from the first few lines of a page or file."""

    for raw_line in text.splitlines()[:12]:
        line = raw_line.strip().strip("#").strip()
        if not line or len(line) > 120:
            continue
        looks_like_heading = (
            line.lower().startswith(("chapter ", "section ", "unit "))
            or line.isupper()
            or re.match(r"^\d+(\.\d+)*\s+[A-Z]", line) is not None
        )
        if looks_like_heading:
            return line
    return None


def make_excerpt(text: str, max_chars: int = 360) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def list_source_files(raw_docs_dir: Path) -> list[Path]:
    if not raw_docs_dir.exists():
        return []
    return sorted(
        path
        for path in raw_docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ensure_project_dirs(settings: Settings) -> None:
    settings.raw_docs_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    if not settings.chroma_host:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)


def get_chroma_collection(settings: Settings, reset: bool = False) -> Any:
    ensure_project_dirs(settings)
    if settings.chroma_host:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            ssl=settings.chroma_ssl,
        )
    else:
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass
    return client.get_or_create_collection(settings.chroma_collection)


def build_embed_model(settings: Settings) -> Any:
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OpenAI embeddings require MARINE_RAG_OPENAI_API_KEY or OPENAI_API_KEY."
            )
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=settings.local_embedding_model)


def build_llm(settings: Settings) -> Any:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("Anthropic requires MARINE_RAG_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.")
        from llama_index.llms.anthropic import Anthropic

        return Anthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
        )

    if not settings.openai_api_key:
        raise RuntimeError("OpenAI requires MARINE_RAG_OPENAI_API_KEY or OPENAI_API_KEY.")

    from llama_index.llms.openai import OpenAI

    return OpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )

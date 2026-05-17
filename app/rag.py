import logging
from hashlib import sha256
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import Settings, get_settings
from app.schemas import AskResponse, SourceCitation
from app.utils import build_embed_model, build_llm, get_chroma_collection, make_excerpt

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a marine biology tutor answering questions using only the provided textbook excerpts. "
    "Do not use outside knowledge. Cite the source filename and page number for claims when possible. "
    "If the answer is not supported by the excerpts, say: I don't know based on the provided documents."
)

UNKNOWN_ANSWER = "I don't know based on the provided documents."


@dataclass
class RerankedNode:
    node: NodeWithScore
    rerank_score: float | None = None


class RAGService:
    """RAG pipeline: retrieve candidates, rerank them, answer with cited context."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._index: VectorStoreIndex | None = None
        self._llm = None
        self._reranker = None
        self._redis = None

    def ask(self, question: str) -> AskResponse:
        cache_key = self._cache_key(question)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        index = self._get_index()
        retriever = index.as_retriever(similarity_top_k=self.settings.similarity_top_k)
        candidates = retriever.retrieve(question)
        if not candidates:
            return AskResponse(answer=UNKNOWN_ANSWER, sources=[])

        final_nodes = self._rerank(question, candidates)[: self.settings.final_top_k]
        context = self._build_context(final_nodes)
        prompt = self._build_user_prompt(question, context)
        llm = self._get_llm()
        response = llm.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]
        )
        answer = (response.message.content or "").strip() or UNKNOWN_ANSWER
        result = AskResponse(
            answer=answer,
            sources=[self._citation_from_node(item.node) for item in final_nodes],
        )
        self._cache_set(cache_key, result)
        return result

    def index_ready(self) -> bool:
        try:
            collection = get_chroma_collection(self.settings)
            return collection.count() > 0
        except Exception:
            logger.exception("Unable to inspect Chroma collection")
            return False

    def reload_index(self) -> None:
        self._index = None

    def _get_index(self) -> VectorStoreIndex:
        if self._index is not None:
            return self._index

        collection = get_chroma_collection(self.settings)
        if collection.count() == 0:
            raise RuntimeError("The vector store is empty. Add documents and run POST /ingest first.")

        vector_store = ChromaVectorStore(chroma_collection=collection)
        embed_model = build_embed_model(self.settings)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
        return self._index

    def _get_llm(self):
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    def _get_redis(self):
        if not self.settings.redis_url:
            return None
        if self._redis is None:
            import redis

            self._redis = redis.Redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        return self._redis

    def _cache_key(self, question: str) -> str:
        raw_key = "|".join(
            [
                self.settings.chroma_collection,
                str(self.settings.similarity_top_k),
                str(self.settings.final_top_k),
                question.strip().lower(),
            ]
        )
        return "marine-rag:ask:" + sha256(raw_key.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> AskResponse | None:
        try:
            redis_client = self._get_redis()
            if redis_client is None:
                return None
            cached = redis_client.get(key)
            if cached:
                logger.info("Returning cached RAG response")
                return AskResponse.model_validate_json(cached)
        except Exception:
            logger.exception("Redis cache read failed; continuing without cache")
        return None

    def _cache_set(self, key: str, response: AskResponse) -> None:
        try:
            redis_client = self._get_redis()
            if redis_client is None:
                return
            redis_client.setex(
                key,
                self.settings.cache_ttl_seconds,
                response.model_dump_json(),
            )
        except Exception:
            logger.exception("Redis cache write failed; continuing without cache")

    def _rerank(self, question: str, candidates: list[NodeWithScore]) -> list[RerankedNode]:
        if not self.settings.enable_reranking or len(candidates) <= 1:
            return [RerankedNode(node=candidate) for candidate in candidates]

        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.settings.reranker_model)

            pairs = [
                (question, candidate.node.get_content(metadata_mode=MetadataMode.NONE))
                for candidate in candidates
            ]
            scores = self._reranker.predict(pairs)
            reranked = [
                RerankedNode(node=candidate, rerank_score=float(score))
                for candidate, score in zip(candidates, scores, strict=False)
            ]
            return sorted(reranked, key=lambda item: item.rerank_score or 0.0, reverse=True)
        except Exception:
            logger.exception("Reranking failed; falling back to vector similarity order")
            return [RerankedNode(node=candidate) for candidate in candidates]

    def _build_context(self, nodes: list[RerankedNode]) -> str:
        blocks: list[str] = []
        for rank, item in enumerate(nodes, start=1):
            metadata = item.node.node.metadata
            filename = metadata.get("filename", "unknown")
            page = _page_or_unknown(metadata.get("page"))
            chunk_id = metadata.get("chunk_id", f"chunk_{rank:04d}")
            text = item.node.node.get_content(metadata_mode=MetadataMode.NONE)
            blocks.append(
                f"[Source {rank}]\n"
                f"filename: {filename}\n"
                f"page: {page}\n"
                f"chunk_id: {chunk_id}\n"
                f"excerpt:\n{text}"
            )
        return "\n\n".join(blocks)

    def _build_user_prompt(self, question: str, context: str) -> str:
        return (
            "Use the following textbook excerpts as the complete evidence base.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer in a clear tutoring style. Include filename/page citations inline when the excerpts support the claim. "
            f"If the excerpts do not support an answer, reply exactly: {UNKNOWN_ANSWER}"
        )

    def _citation_from_node(self, node: NodeWithScore) -> SourceCitation:
        metadata = node.node.metadata
        return SourceCitation(
            filename=str(metadata.get("filename", "unknown")),
            page=_page_or_none(metadata.get("page")),
            chunk_id=str(metadata.get("chunk_id", node.node.node_id)),
            excerpt=make_excerpt(node.node.get_content(metadata_mode=MetadataMode.NONE)),
        )


def _page_or_none(value: object) -> int | None:
    try:
        page = int(value)  # Chroma metadata cannot store null, so text files use 0.
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _page_or_unknown(value: object) -> str:
    page = _page_or_none(value)
    return str(page) if page is not None else "unknown"

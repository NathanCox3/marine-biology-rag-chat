import logging

from fastapi import FastAPI, HTTPException, status

from app.config import get_settings
from app.ingest import ingest_documents
from app.rag import RAGService
from app.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse
from app.utils import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="RAG API for answering questions from marine biology textbook excerpts.",
    version="1.0.0",
)
rag_service = RAGService(settings)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        app_name=settings.app_name,
        status="ok",
        index_ready=rag_service.index_ready(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return rag_service.ask(request.question)
    except RuntimeError as exc:
        logger.warning("RAG request cannot be completed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while answering question")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to answer the question right now.",
        ) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    try:
        stats = ingest_documents(settings)
        rag_service.reload_index()
        return IngestResponse(
            status="indexed",
            documents_loaded=stats.documents_loaded,
            chunks_indexed=stats.chunks_indexed,
            storage_path=stats.storage_path,
        )
    except FileNotFoundError as exc:
        logger.warning("Ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("Ingestion configuration error: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected ingestion failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to ingest documents right now.",
        ) from exc


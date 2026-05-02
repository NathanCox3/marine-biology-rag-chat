from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    app_name: str
    status: str
    index_ready: bool


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What role do coral reefs play in marine ecosystems?"])


class SourceCitation(BaseModel):
    filename: str
    page: int | None = None
    chunk_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int
    storage_path: str


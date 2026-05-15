"""
api/routes/summary.py
──────────────────────
GET  /summary/{source_name}   — summarize an ingested document
POST /summary/text            — summarize arbitrary text
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tools.document_summary import summarize_document, summarize_text
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/summary", tags=["Summarization"])


class TextSummaryRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=20000)
    max_length: int = Field(300, ge=50, le=1000)


@router.get("/{source_name}", summary="Summarize an ingested document by source name")
async def summarize_source(source_name: str, max_chunks: int = 10):
    """
    Generate a concise summary of a document already in the vector store.
    Uses map-reduce for long documents automatically.
    """
    try:
        summary = summarize_document(source_name, max_chunks=max_chunks)
        return {"source": source_name, "summary": summary}
    except Exception as e:
        logger.error(f"Summary failed for '{source_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text", summary="Summarize arbitrary text")
async def summarize_raw_text(request: TextSummaryRequest):
    """Summarize any text string without needing it to be ingested first."""
    try:
        summary = summarize_text(request.text, max_length=request.max_length)
        return {"summary": summary, "original_length": len(request.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

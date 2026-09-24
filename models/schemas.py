"""Sch?mas de donn?es structur?s et valid?s avec Pydantic."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Message de conversation persistant dans Supabase ou dans Streamlit."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "model"]
    content: str = Field(..., min_length=1)


class DocumentSummary(BaseModel):
    """R?sum? synth?tique d'un document pr?sent dans la base de connaissance."""

    model_config = ConfigDict(extra="ignore")

    file_name: str
    chunks: int = Field(default=0, ge=0)


class DocumentChunk(BaseModel):
    """Fragment de texte vectoris? pour la recherche RAG."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    session_id: str | None = None
    file_name: str | None = None
    content: str
    embedding: list[float] | None = None
    similarity: float | None = None

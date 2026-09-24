"""Tests unitaires pour les mod?les Pydantic."""

import pytest
from pydantic import ValidationError

from models.schemas import ChatMessage, DocumentChunk, DocumentSummary


def test_chat_message_validation():
    msg = ChatMessage(role="user", content="Hello world")
    assert msg.role == "user"
    assert msg.content == "Hello world"

    # R?le invalide
    with pytest.raises(ValidationError):
        ChatMessage(role="inconnu", content="test")  # type: ignore[arg-type]

    # Contenu vide
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


def test_document_summary_validation():
    doc = DocumentSummary(file_name="doc.pdf", chunks=10)
    assert doc.file_name == "doc.pdf"
    assert doc.chunks == 10

    # Chunks n?gatifs
    with pytest.raises(ValidationError):
        DocumentSummary(file_name="doc.pdf", chunks=-1)


def test_document_chunk_validation():
    chunk = DocumentChunk(
        session_id="s1", file_name="a.pdf", content="Extrait", similarity=0.85
    )
    assert chunk.session_id == "s1"
    assert chunk.similarity == 0.85
    dump = chunk.model_dump(exclude_none=True)
    assert "id" not in dump
    assert dump["file_name"] == "a.pdf"

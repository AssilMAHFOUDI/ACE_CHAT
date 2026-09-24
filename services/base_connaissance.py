"""Base de connaissance d'une session : ingestion, liste et suppression."""

import logging
from collections.abc import Callable
from typing import Any

from models.schemas import DocumentSummary
from modules.database import clear_document_chunks, list_session_documents
from modules.document_processor import (
    extract_text_from_file,
    process_and_store_document,
)

logger = logging.getLogger(__name__)


def documents_indexes(supabase_client: Any, session_id: str) -> list[dict[str, Any]]:
    """Liste les documents de la session valid?s avec Pydantic."""
    bruts = list_session_documents(supabase_client, session_id)
    documents: list[dict[str, Any]] = []
    for doc in bruts:
        try:
            summary = DocumentSummary.model_validate(doc)
            documents.append(summary.model_dump())
        except Exception as err:
            logger.warning("Document corrompu ignor? : %s", err)
    return documents


def document_deja_indexe(documents: list[dict[str, Any]], file_name: str) -> bool:
    """True si un document du m?me nom est d?j? pr?sent dans la liste."""
    return any(doc.get("file_name") == file_name for doc in documents)


def indexer_document(
    fichier_upload: Any,
    session_id: str,
    supabase_client: Any,
    ai_client: Any,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Lit le fichier, le vectorise et remplace sa version pr?c?dente."""
    texte = extract_text_from_file(fichier_upload)
    return process_and_store_document(
        text=texte,
        file_name=fichier_upload.name,
        session_id=session_id,
        supabase_client=supabase_client,
        ai_client=ai_client,
        progress_callback=progress_callback,
    )


def supprimer_document(supabase_client: Any, session_id: str, file_name: str) -> bool:
    """Supprime un document de la base. Retourne True en cas de succ?s."""
    return clear_document_chunks(supabase_client, session_id, file_name)

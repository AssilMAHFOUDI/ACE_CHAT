"""Base de connaissance d une session : ingestion, liste et suppression."""

import logging

from modules.database import clear_document_chunks, list_session_documents
from modules.document_processor import (
    extract_text_from_file,
    process_and_store_document,
)

logger = logging.getLogger(__name__)


def documents_indexes(supabase_client, session_id):
    """Liste les documents de la session avec leur nombre de morceaux."""
    return list_session_documents(supabase_client, session_id)


def document_deja_indexe(documents, file_name):
    """True si un document du meme nom est deja present dans la liste."""
    return any(document["file_name"] == file_name for document in documents)


def indexer_document(fichier_upload, session_id, supabase_client, ai_client,
                     progress_callback=None):
    """Lit le fichier, le vectorise et remplace sa version precedente.

    Leve une erreur si la lecture ou l ingestion echoue ; dans ce cas la base
    n a pas ete modifiee.
    """
    texte = extract_text_from_file(fichier_upload)
    return process_and_store_document(
        text=texte,
        file_name=fichier_upload.name,
        session_id=session_id,
        supabase_client=supabase_client,
        ai_client=ai_client,
        progress_callback=progress_callback,
    )


def supprimer_document(supabase_client, session_id, file_name):
    """Supprime un document de la base. Retourne True en cas de succes."""
    return clear_document_chunks(supabase_client, session_id, file_name)

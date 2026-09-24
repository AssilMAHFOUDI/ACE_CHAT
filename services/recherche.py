"""Recherche RAG : question, vecteur, RPC, prompt enrichi et sources."""

import logging
from typing import Any

from models.schemas import DocumentChunk
from modules.ai_engine import generate_rag_prompt, get_embedding
from modules.database import filtre_document_disponible, search_relevant_chunks

logger = logging.getLogger(__name__)


def chercher_passages(
    question: str,
    session_id: str,
    supabase_client: Any,
    ai_client: Any,
    file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Embed la question puis interroge la base de la session."""
    vecteur_question = get_embedding(question, ai_client)
    bruts = search_relevant_chunks(
        supabase_client=supabase_client,
        query_embedding=vecteur_question,
        session_id=session_id,
        file_name=file_name,
    )
    valides: list[dict[str, Any]] = []
    for b in bruts:
        try:
            chunk = DocumentChunk.model_validate(b)
            valides.append(chunk.model_dump(exclude_none=True))
        except Exception as err:
            logger.warning("Extrait RAG invalide ignor? : %s", err)
    return valides


def extraire_sources(chunks: list[dict[str, Any]]) -> list[str]:
    """Liste tri?e des documents r?ellement utilis?s pour r?pondre."""
    return sorted({c["file_name"] for c in chunks if c.get("file_name")})


def construire_prompt(chunks: list[dict[str, Any]], question: str) -> str:
    """Construit le prompt RAG, ou l'avis de repli si aucun extrait."""
    if not chunks:
        logger.info("Aucun extrait pertinent pour la question : %s", question)
        return (
            f"L'utilisateur pose cette question : {question}, mais aucun extrait "
            "pertinent n'a ?t? trouv? dans la base."
        )
    return generate_rag_prompt(chunks, question)


def filtre_applique_par_base() -> bool:
    """True si le filtrage par document est fait par la base de donn?es."""
    return filtre_document_disponible()

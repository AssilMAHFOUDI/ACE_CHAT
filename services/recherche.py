"""Recherche RAG : question, vecteur, RPC, prompt enrichi et sources."""

import logging

from modules.ai_engine import generate_rag_prompt, get_embedding
from modules.database import filtre_document_disponible, search_relevant_chunks

logger = logging.getLogger(__name__)


def chercher_passages(question, session_id, supabase_client, ai_client,
                      file_name=None):
    """Embed la question puis interroge la base de la session."""
    vecteur_question = get_embedding(question, ai_client)
    return search_relevant_chunks(
        supabase_client=supabase_client,
        query_embedding=vecteur_question,
        session_id=session_id,
        file_name=file_name,
    )


def extraire_sources(chunks):
    """Liste triee des documents reellement utilises pour repondre."""
    return sorted(
        {chunk["file_name"] for chunk in chunks if chunk.get("file_name")}
    )


def construire_prompt(chunks, question):
    """Construit le prompt RAG, ou l avis de repli si aucun extrait."""
    if not chunks:
        logger.info("Aucun extrait pertinent pour la question : %s", question)
        return (
            f"L'utilisateur pose cette question : {question}, mais aucun extrait "
            "pertinent n'a ete trouve dans la base."
        )
    return generate_rag_prompt(chunks, question)


def filtre_applique_par_base():
    """True si le filtrage par document est fait par la base de donnees."""
    return filtre_document_disponible()

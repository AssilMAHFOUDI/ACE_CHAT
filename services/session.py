"""Session Streamlit et historique de conversation."""

import logging
import uuid

from modules.database import (
    clear_chat_history,
    clear_document_chunks,
    get_chat_history,
    save_message,
)

logger = logging.getLogger(__name__)


def nouvel_identifiant():
    """Genere un nouvel identifiant de session (UUID v4)."""
    return str(uuid.uuid4())


def charger_historique(supabase_client, session_id):
    """Renvoie l historique persiste, au format attendu par l interface."""
    historique = get_chat_history(supabase_client, session_id)
    return [
        {"role": ligne["role"], "content": ligne["content"]}
        for ligne in historique
    ]


def enregistrer_message(supabase_client, session_id, role, contenu):
    """Persiste un message de la conversation."""
    save_message(supabase_client, session_id, role, contenu)
    logger.debug("Message %s enregistre pour la session %s", role, session_id[:8])


def reinitialiser_session(supabase_client, session_id):
    """Efface chunks et historique, renvoie un nouvel identifiant de session."""
    clear_document_chunks(supabase_client, session_id)
    clear_chat_history(supabase_client, session_id)
    nouveau_id = nouvel_identifiant()
    logger.info("Session reinitialisee : %s -> %s", session_id[:8], nouveau_id[:8])
    return nouveau_id

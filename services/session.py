"""Session Streamlit et historique de conversation."""

import logging
import uuid
from typing import Any

from models.schemas import ChatMessage
from modules.database import (
    clear_chat_history,
    clear_document_chunks,
    get_chat_history,
    save_message,
)

logger = logging.getLogger(__name__)


def nouvel_identifiant() -> str:
    """G?n?re un nouvel identifiant de session (UUID v4)."""
    return str(uuid.uuid4())


def charger_historique(supabase_client: Any, session_id: str) -> list[dict[str, str]]:
    """Renvoie l'historique persist?, valid? avec Pydantic au format attendu par l'interface."""
    historique = get_chat_history(supabase_client, session_id)
    valides: list[dict[str, str]] = []
    for ligne in historique:
        try:
            msg = ChatMessage.model_validate(ligne)
            valides.append({"role": msg.role, "content": msg.content})
        except Exception as err:
            logger.warning(
                "Message invalide ignor? (session %s) : %s", session_id[:8], err
            )
    return valides


def enregistrer_message(
    supabase_client: Any, session_id: str, role: str, contenu: str
) -> None:
    """Valide et persiste un message de la conversation."""
    msg = ChatMessage(role=role, content=contenu)  # type: ignore[arg-type]
    save_message(supabase_client, session_id, msg.role, msg.content)
    logger.debug("Message %s enregistr? pour la session %s", msg.role, session_id[:8])


def reinitialiser_session(supabase_client: Any, session_id: str) -> str:
    """Efface chunks et historique, renvoie un nouvel identifiant de session."""
    clear_document_chunks(supabase_client, session_id)
    clear_chat_history(supabase_client, session_id)
    nouveau_id = nouvel_identifiant()
    logger.info("Session r?initialis?e : %s -> %s", session_id[:8], nouveau_id[:8])
    return nouveau_id

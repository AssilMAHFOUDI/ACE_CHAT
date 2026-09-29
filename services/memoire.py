"""Mémoire de conversation : borne la taille du contexte envoyé au modèle.

Sans mémoire, l'historique complet est renvoyé à Gemini à chaque tour : la
requête grossit jusqu'à saturer la fenêtre du modèle (et la facture). La
stratégie ici garde les derniers messages mot pour mot et remplace les plus
anciens par un résumé, réinjecté dans l'appel suivant pour que la synthèse
reste incrémentale au lieu de repartir de zéro à chaque tour.

L'historique complet n'est jamais modifié : ni l'affichage, ni la persistance.
Seule la vue envoyée au modèle est compressée, et le résumé n'est jamais écrit
en base (il n'est qu'un état de session, recalculable).
"""

import logging
from typing import Any

from modules.ai_engine import CONFIG_SANS_OUTILS
from modules.config import (
    CHAT_MODEL,
    MEMORY_SUMMARY_MAX_CHARS,
    MEMORY_WINDOW_SIZE,
)

logger = logging.getLogger(__name__)

# Rôles désignant une réponse du modèle (selon la source : Streamlit ou base).
ROLES_MODELE = ("assistant", "model")

# Rôle d'un message de l'utilisateur dans l'historique envoyé au modèle.
ROLES_UTILISATEUR = ("user",)

# Marqueur visible par le modèle : sans lui, le résumé serait pris pour une
# nouvelle question de l'utilisateur.
EN_TETE_RESUME = "[Résumé des échanges précédents]"

PROMPT_RESUME = (
    "Résume la conversation suivante en quelques phrases, dans la langue de "
    "l'utilisateur. Conserve les faits établis, les décisions prises, les "
    "contraintes et préférences données, les chiffres cités, ainsi que les "
    "questions restées sans réponse. N'invente rien et n'ajoute aucun "
    "commentaire : renvoie uniquement le résumé."
)


def _formater_echanges(messages: list[dict[str, str]]) -> str:
    """Convertit des messages en texte brut pour l'appel de synthèse."""
    lignes = []
    for message in messages:
        qui = "Assistant" if message["role"] in ROLES_MODELE else "Utilisateur"
        lignes.append(f"{qui} : {message['content']}")
    return "\n".join(lignes)


def _tronquer(resume: str) -> str:
    """Borne la longueur du résumé (la taille du prompt doit rester maîtrisée)."""
    if len(resume) <= MEMORY_SUMMARY_MAX_CHARS:
        return resume
    return resume[: MEMORY_SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def _nouveau_resume(
    ia_client: Any, resume_existant: str | None, messages: list[dict[str, str]]
) -> str:
    """Résume les messages qui sortent de la fenêtre, en prolongeant le résumé.

    Le résumé déjà établi est fourni au modèle pour que la synthèse reste
    incrémentale : un tour ne renvoie que les échanges qui viennent de sortir de
    la fenêtre, sans réécrire tout le passé de la conversation.
    """
    portions = [PROMPT_RESUME, ""]
    if resume_existant:
        portions += [f"Résumé déjà établi :\n{resume_existant}", ""]
    portions.append(f"Échanges à intégrer :\n{_formater_echanges(messages)}")

    chat = ia_client.chats.create(
        model=CHAT_MODEL, history=[], config=CONFIG_SANS_OUTILS
    )
    resume = (chat.send_message("\n".join(portions)).text or "").strip()
    if not resume:
        raise ValueError("le modèle n'a produit aucun résumé")
    return _tronquer(resume)


def _aligner(historique: list[dict[str, str]], index: int, roles: tuple) -> int:
    """Avance l'index jusqu'au premier message dont le rôle figure dans `roles`.

    Gemini exige que les tours de la conversation alternent. Comme le résumé est
    injecté comme un tour de l'utilisateur, la fenêtre qui suit doit commencer
    sur une réponse du modèle ; sans résumé, l'historique doit lui commencer sur
    un message de l'utilisateur. Un message ainsi déplacé n'est pas perdu : il
    est intégré au résumé.
    """
    while index < len(historique) and historique[index]["role"] not in roles:
        index += 1
    return index


def _vue_modele(
    resume: str | None, fenetre: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Assemble la vue envoyée au modèle : résumé (tour utilisateur) + fenêtre."""
    if not resume:
        return list(fenetre)
    resume_vu = {"role": "user", "content": f"{EN_TETE_RESUME}\n{resume}"}
    return [resume_vu] + list(fenetre)


def compresser_historique(
    historique: list[dict[str, str]],
    ia_client: Any,
    resume: str | None = None,
    deja_resume: int = 0,
) -> tuple[list[dict[str, str]], str | None, int]:
    """Compresse l'historique au-delà de MEMORY_WINDOW_SIZE messages récents.

    Args:
        historique: messages de la session, du plus ancien au plus récent.
        ia_client: client Gemini utilisé pour rédiger le résumé.
        resume: résumé déjà établi lors d'un tour précédent (None s'il n'y en a
            pas encore).
        deja_resume: nombre de messages les plus anciens déjà couverts par ce
            résumé. Mémorisé côté session pour ne jamais résumer deux fois.

    Returns:
        Le tuple (messages à envoyer au modèle, résumé à mémoriser, nombre de
        messages couverts par ce résumé). Aucun appel réseau n'a lieu tant
        qu'aucun message ne sort de la fenêtre.
    """
    total = len(historique)
    if deja_resume > total:
        # Des lignes invalides ont pu être filtrées au rechargement : l'index
        # mémorisé ne désigne plus rien de valide, on repart d'un état propre.
        logger.warning(
            "Index de résumé périmé (%d pour %d messages) : mémoire réinitialisée.",
            deja_resume,
            total,
        )
        resume, deja_resume = None, 0

    limite = max(total - MEMORY_WINDOW_SIZE, deja_resume)
    if limite <= deja_resume:
        # Rien de nouveau ne sort de la fenêtre : le résumé suffit, aucun appel.
        return _vue_modele(resume, historique[deja_resume:]), resume, deja_resume

    limite = _aligner(historique, limite, ROLES_MODELE)
    if limite >= total:
        # Cas dégénéré (aucune réponse du modèle dans la fenêtre) : plutôt qu'une
        # requête réduite à son résumé, on renvoie tout sans compresser.
        logger.warning("Aucune réponse du modèle à conserver : compression reportée.")
        return _vue_modele(resume, historique[deja_resume:]), resume, deja_resume

    try:
        resume_a_jour = _nouveau_resume(
            ia_client, resume, historique[deja_resume:limite]
        )
    except Exception as erreur:
        # Pas de résumé possible. L'index n'avance pas : les mêmes messages
        # seront proposés à la synthèse au tour suivant, rien n'est oublié.
        logger.warning(
            "Échec du résumé de mémoire (%s) : seuls les récents sont envoyés.",
            erreur,
        )
        if resume:
            # Le résumé déjà établi reste utile, même non mis à jour.
            return _vue_modele(resume, historique[limite:]), resume, deja_resume
        debut = _aligner(historique, limite, ROLES_UTILISATEUR)
        return list(historique[debut:]), None, deja_resume

    logger.info(
        "Mémoire compressée : %d message(s) résumé(s), %d caractère(s).",
        limite - deja_resume,
        len(resume_a_jour),
    )
    return _vue_modele(resume_a_jour, historique[limite:]), resume_a_jour, limite

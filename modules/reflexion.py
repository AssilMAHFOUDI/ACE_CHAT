"""Réflexion de l'agent : planification, auto-critique et observations.

La boucle ReAct de `modules.ai_engine` agit dès le premier tour : elle ne
s'interroge ni sur la stratégie à suivre, ni sur la qualité de ce qu'elle
produit. Ce module apporte les briques de ce contrôle : le plan avant d'agir,
la critique de la réponse avant de l'envoyer, et le formatage des observations.

Il n'appelle jamais le modèle lui-même : l'appel est fourni par l'appelant
(`appeler_modele`), ce qui rend toute cette logique testable sans réseau ni
client Gemini. Comme `modules.tools`, il reste de la logique pure (stdlib et
constantes seules) : ni Streamlit, ni `google.genai`. Placé dans `modules` et
non dans `services` pour que le sens de dépendance reste unique
(`services` -> `modules`) et évite toute dépendance circulaire.
"""

import logging

from modules.config import (
    CRITIQUE_BROUILLON_MAX_CHARS,
    CRITIQUE_MAX_CHARS,
    OBSERVATIONS_MAX_CHARS,
    PLAN_MAX_CHARS,
    PLAN_SEUIL_CARACTERES,
)

logger = logging.getLogger(__name__)

# Réponse exigée du modèle quand la question ne mérite aucune planification.
PLAN_SIMPLE = "PLAN SIMPLE"

# Préfixes tenant lieu de verdict négatif. Le protocole est volontairement
# textuel (plutôt qu'un JSON) : il tient dans un appel outils coupés et se
# teste sans schéma. Tout le reste est lu comme une validation (fail-open).
PREFIXES_REFUS = ("INSUFFISANT", "NON")

PROMPT_PLAN = (
    "Tu prépares le plan d'une aide qui répondra ensuite à la question posée, "
    "outils compris (calculatrice, météo, recherche web). Écris entre 2 et 4 "
    "étapes numérotées, une par ligne, au format « 1. ... ». Chaque étape dit "
    "quoi collecter ou calculer, sans jamais répondre à la question. Si la "
    "question se traite d'une seule traite, sans étape intermédiaire, réponds "
    f"uniquement : {PLAN_SIMPLE}."
)

PROMPT_CRITIQUE = (
    "Tu vérifies une réponse avant qu'elle soit envoyée à l'utilisateur. Elle "
    "est insuffisante si elle ignore une partie de la question, contredit une "
    "observation, avance un fait qu'aucune observation n'établit, ou reste sans "
    "conclusion exploitable. Sinon elle est correcte. Réponds uniquement par "
    "« OK », ou par « INSUFFISANT : » suivi d'une raison courte et précise "
    "disant ce qui manque ou ce qu'il faut corriger."
)


def _tronquer(texte, max_caracteres):
    """Borne la longueur d'un texte injecté dans un prompt."""
    if len(texte) <= max_caracteres:
        return texte
    return texte[: max_caracteres - 1].rstrip() + "…"


def formater_observations(observations):
    """
    Résume les observations des outils pour les rappels injectés au modèle.

    Les observations les plus récentes sont conservées en priorité : le rappel
    est tronqué à OBSERVATIONS_MAX_CHARS pour garder la main sur la taille du
    prompt, et la troncature est signalée au modèle.
    """
    if not observations:
        return ""

    blocs = []
    total = 0
    for observation in reversed(observations):
        if total + len(observation) > OBSERVATIONS_MAX_CHARS:
            blocs.append("(... observations plus anciennes tronquées ...)")
            break
        blocs.append(observation)
        total += len(observation)

    blocs.reverse()
    return "\n".join(blocs)


def merite_un_plan(question):
    """Vrai si la question est assez longue ou multiple pour justifier un plan.

    Le plan coûte un appel de modèle supplémentaire : il est réservé aux
    questions qui en ont besoin (longues, ou à plusieurs voix), jamais au
    « bonjour » d'ouverture ni à la question triviale.
    """
    nette = (question or "").strip()
    if not nette:
        return False
    return len(nette) >= PLAN_SEUIL_CARACTERES or nette.count("?") >= 2


def construire_plan(question, appeler_modele):
    """Établit le plan de recherche, ou None si la question n'en mérite pas.

    `appeler_modele` est la fonction qui envoie un texte au modèle et renvoie sa
    réponse. Une panne de cet appel n'interdit pas de répondre : elle supprime
    seulement le plan (fail-open).
    """
    nette = (question or "").strip()
    if not nette:
        return None

    try:
        invite = f"{PROMPT_PLAN}\n\nQuestion : {nette}"
        reponse = (appeler_modele(invite) or "").strip()
    except Exception as erreur:
        logger.warning("Échec du plan initial (%s) : l'agent agit sans plan.", erreur)
        return None

    if not reponse or reponse.upper().startswith(PLAN_SIMPLE):
        logger.info("Question jugée simple : aucun plan calculé.")
        return None
    return _tronquer(reponse, PLAN_MAX_CHARS)


def analyser_verdict(texte):
    """Interprète la réponse du modèle de critique : (validite, raison).

    Les verdicts non reconnus, vides ou mal formatés sont tenus pour une
    validation : une critique illisible ne doit jamais retenir une réponse.
    """
    net = (texte or "").strip()
    if not net:
        return True, ""

    haut = net.upper().lstrip("#*>-!?\"' \t")
    for refus in PREFIXES_REFUS:
        if haut.startswith(refus):
            _, separe, raison = net.partition(":")
            if not separe or not raison.strip():
                raison = "la réponse a été jugée insuffisante, sans précision"
            return False, _tronquer(raison.strip(), CRITIQUE_MAX_CHARS)
    return True, ""


def _prompt_critique(question, observations, brouillon):
    """Assemble la demande de vérification : question, faits, réponse à juger."""
    portions = [PROMPT_CRITIQUE, "", f"Question : {question}", ""]

    rappel = formater_observations(observations)
    if rappel:
        portions += ["Observations collectées auprès des outils :", rappel, ""]

    portions += [
        "Réponse à évaluer :",
        _tronquer(brouillon, CRITIQUE_BROUILLON_MAX_CHARS),
    ]
    return "\n".join(portions)


def evaluer_brouillon(question, observations, brouillon, appeler_modele):
    """Juge la réponse produite avant son envoi : (validite, raison a corriger).

    La vérification reste aveugle aux outils (l'appel est fait outils coupés par
    l'appelant) : elle ne fait qu'interroger la cohérence de la réponse au vu des
    faits déjà collectés. Une panne de l'appel se traduit par une validation.
    """
    try:
        verdict = appeler_modele(_prompt_critique(question, observations, brouillon))
    except Exception as erreur:
        logger.warning(
            "Échec de l'auto-critique (%s) : réponse envoyée telle quelle.", erreur
        )
        return True, ""

    valide, raison = analyser_verdict(verdict)
    logger.info(
        "Auto-critique : %s",
        "réponse validée" if valide else f"réponse à reprendre ({raison})",
    )
    return valide, raison

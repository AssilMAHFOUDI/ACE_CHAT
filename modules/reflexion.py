"""Réflexion de l'agent : planification, auto-critique et observations.

La boucle ReAct de `modules.ai_engine` agit dès le premier tour : elle ne
s'interroge ni sur la stratégie à suivre, ni sur la qualité de ce qu'elle
produit. Ce module apporte les briques de ce contrôle : la planification qui
précède la boucle (et qui peut répondre directement quand elle a la réponse), la
critique de la réponse avant de l'envoyer, et le formatage des observations.

Il n'appelle jamais le modèle lui-même : l'appel est fourni par l'appelant
(`appeler_modele`), ce qui rend toute cette logique testable sans réseau ni
client Gemini. Comme `modules.tools`, il reste de la logique pure (stdlib et
constantes seules) : ni Streamlit, ni `google.genai`. Placé dans `modules` et
non dans `services` pour que le sens de dépendance reste unique
(`services` -> `modules`) et évite toute dépendance circulaire.
"""

import logging
from dataclasses import dataclass

from modules.config import (
    CRITIQUE_BROUILLON_MAX_CHARS,
    CRITIQUE_MAX_CHARS,
    OBSERVATIONS_MAX_CHARS,
    PLAN_MARQUEUR_MAX_CHARS,
    PLAN_MAX_CHARS,
)

logger = logging.getLogger(__name__)

# Marqueurs exigés en première ligne de la réponse du planificateur : soit une
# réponse directe, soit un plan à suivre. Le protocole est volontairement
# textuel (plutôt qu'un JSON) : il tient dans un appel outils coupés, se lit en
# flux et se teste sans schéma.
MARQUEUR_PLAN = "PLAN"
MARQUEUR_REPONSE = "REPONSE"

# Décors tolérés en tête de la réponse : markdown, ponctuation, séparateurs.
DECORS = " \t\r\n#*-_.:>\"'"

# Préfixes tenant lieu de verdict négatif. Tout le reste est lu comme une
# validation (fail-open).
PREFIXES_REFUS = ("INSUFFISANT", "NON")

PROMPT_PLAN = (
    "Tu es le planificateur d'une aide qui répondra ensuite à la question, "
    "outils compris (calculatrice, météo, recherche web). Ta réponse commence "
    f"toujours par une première ligne contenant uniquement {MARQUEUR_REPONSE} "
    f"ou {MARQUEUR_PLAN}.\n"
    f"- {MARQUEUR_REPONSE} : la réponse se trouve entièrement dans les échanges "
    "fournis ou dans la question elle-même (salutation, question de culture "
    "générale, calcul mental, demande de reformulation, suivi immédiat). "
    "Écris alors la réponse complète et définitive à l'utilisateur, juste après "
    "cette ligne.\n"
    f"- {MARQUEUR_PLAN} : la réponse exige une source extérieure (web, météo, "
    "document) ou plusieurs étapes enchaînées. Écris alors, une par ligne, de 2 "
    "à 4 étapes numérotées au format « 1. ... ». Chaque étape dit quoi "
    "collecter ou calculer, sans jamais répondre à la question.\n"
    f"Dans le doute, écris {MARQUEUR_PLAN}."
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


@dataclass(frozen=True)
class Planification:
    """Verdict du planificateur : un plan à suivre, ou une réponse directe.

    Les deux champs sont exclusifs. Vides tous les deux, la planification n'a
    rien apporté : l'agent décide alors seul.
    """

    plan: str | None = None
    reponse: str | None = None


def marqueur_en_tete(texte, definitif=False):
    """Lit le marqueur en tête du texte : réponse directe, plan, ou rien.

    Les morceaux arrivent un par un : tant que la première ligne n'est pas close
    (ni terminée par un retour à la ligne, ni assez longue pour trancher), la
    lecture reste indécise et renvoie None. Le flux est alors retenu : il ne doit
    jamais publier un plan comme s'il s'agissait d'une réponse. `definitif` force
    la lecture sur le texte reçu, quel qu'en soit l'état.

    Par défaut c'est MARQUEUR_PLAN : un texte inattendu conduit à la boucle
    ReAct, jamais à une réponse tronquée.
    """
    net = (texte or "").lstrip(DECORS)
    if not net:
        return None
    fin = net.find("\n")
    if fin == -1 and len(net) < PLAN_MARQUEUR_MAX_CHARS and not definitif:
        return None
    ligne = (net if fin == -1 else net[:fin]).upper()
    return MARQUEUR_REPONSE if MARQUEUR_REPONSE in ligne else MARQUEUR_PLAN


def retirer_marqueur(texte):
    """Sépare le texte en (marqueur, corps), le marqueur retiré du corps.

    Le retrait ne touche que la ligne du marqueur : « REPONSE : Bonjour » laisse
    « Bonjour », « PLAN » suivi des étapes laisse les étapes. Sans marqueur, tout
    le texte fait corps (et sera lu comme un plan).
    """
    net = (texte or "").strip()
    if not net:
        return None, ""

    marqueur = marqueur_en_tete(net, definitif=True)
    fin = net.find("\n")
    premiere = net if fin == -1 else net[:fin]
    reste = "" if fin == -1 else net[fin + 1 :]
    haut = premiere.upper()

    if MARQUEUR_REPONSE in haut:
        # « REPONSE : la suite » : la suite appartient déjà à la réponse.
        debut = haut.find(MARQUEUR_REPONSE) + len(MARQUEUR_REPONSE)
        amorce = premiere[debut:].lstrip(DECORS)
        return marqueur, "\n".join(partie for partie in (amorce, reste) if partie)
    if MARQUEUR_PLAN in haut:
        return marqueur, reste
    return marqueur, net


def analyser_planification(texte):
    """Interprète la réponse du planificateur : plan ou réponse directe.

    Un texte vide, ou réduit à son marqueur, ne renvoie rien : l'agent agit seul
    (fail-open). Un plan est borné à PLAN_MAX_CHARS ; une réponse directe est
    rendue telle quelle, car c'est elle que verra l'utilisateur.
    """
    if not (texte or "").strip():
        return Planification()

    marqueur, corps = retirer_marqueur(texte)
    corps = corps.strip()
    if not corps:
        return Planification()
    if marqueur == MARQUEUR_REPONSE:
        return Planification(reponse=corps)
    return Planification(plan=_tronquer(corps, PLAN_MAX_CHARS))


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

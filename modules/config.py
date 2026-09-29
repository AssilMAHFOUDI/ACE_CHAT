"""Configuration centralisée d'ACE CHAT.

Toutes les constantes ajustables (découpage, lots, seuils, modèles) et la
configuration du logging vivent ici, pour éviter les valeurs magiques
dispersées dans les modules. Ce module ne dépend de rien d'autre (stdlib seule) :
il reste testable sans Streamlit ni réseau.
"""

import logging

# --- Découpage des documents ---
CHUNK_SIZE = 1000  # taille d'un morceau de texte (caractères)
CHUNK_OVERLAP = 200  # chevauchement entre morceaux consécutifs
CHUNK_MIN_LENGTH = 10  # morceaux plus courts sont ignorés

# --- Batching ---
EMBEDDING_BATCH_SIZE = 20  # textes par requête d'embedding
INSERT_BATCH_SIZE = 50  # lignes par requête INSERT vers Supabase

# --- Recherche sémantique (RAG) ---
MATCH_THRESHOLD = 0.3  # similarité cosinus minimale
MATCH_COUNT = 4  # nombre d'extraits remontés par question

# --- Agent (ReAct) ---
MAX_ITERATIONS = 10  # boucle raison/act bornée
# Taille maximale du rappel des observations injecté dans le message de
# synthèse forcée (fin de boucle) : borne la taille du prompt.
OBSERVATIONS_MAX_CHARS = 2000

# --- Outils & Réseau ---
# Timeout en secondes appliqué à CHAQUE appel réseau : 1 pour la recherche web,
# 2 pour la météo (géocodage puis prévisions), soit 20 s au pire.
TOOL_NETWORK_TIMEOUT = 10

# --- Mémoire de conversation ---
# Nombre de messages les plus récents envoyés mot pour mot au modèle. Choisi
# pair pour que la fenêtre se termine sur une paire question/réponse et que
# l'historique reste aligné. En dessous de cette taille, rien n'est compressé.
MEMORY_WINDOW_SIZE = 6
# Longueur maximale du résumé qui remplace les messages plus anciens : borne la
# taille du prompt, quelle que soit la durée de la conversation.
MEMORY_SUMMARY_MAX_CHARS = 1500


# --- Modèles Gemini ---
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"
# Dimension des vecteurs produite par EMBEDDING_MODEL : doit correspondre
# exactement au vector(3072) de la colonne document_chunks.embedding.
EMBEDDING_DIMENSIONS = 3072


def configurer_logging(niveau=logging.INFO):
    """Configure le logging global une seule fois (appelé par app.py).

    Messages sans emoji : certains terminaux n'affichent pas les caractères
    non ASCII, ce qui rendait les logs illisibles (mojibake).
    """
    logging.basicConfig(
        level=niveau,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

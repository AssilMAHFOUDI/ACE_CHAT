"""Configuration centralisee d ACE CHAT.

Toutes les constantes ajustables (decoupage, lots, seuils, modeles) et la
configuration du logging vivent ici, pour eviter les valeurs magiques
dispersees dans les modules. Ce module ne depend de rien d autre (stdlib seule) :
il reste testable sans Streamlit ni reseau.
"""

import logging

# --- Decoupage des documents ---
CHUNK_SIZE = 1000        # taille d un morceau de texte (caracteres)
CHUNK_OVERLAP = 200      # chevauchement entre morceaux consecutifs
CHUNK_MIN_LENGTH = 10    # morceaux plus courts sont ignores

# --- Batching ---
EMBEDDING_BATCH_SIZE = 20    # textes par requete d embedding
INSERT_BATCH_SIZE = 50       # lignes par requete INSERT vers Supabase

# --- Recherche semantique (RAG) ---
MATCH_THRESHOLD = 0.3    # similarite cosinus minimale
MATCH_COUNT = 4          # nombre d extraits remontes par question

# --- Agent (ReAct) ---
MAX_ITERATIONS = 10      # boucle raison/act bornee

# --- Modeles Gemini ---
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"


def configurer_logging(niveau=logging.INFO):
    """Configure le logging global une seule fois (appele par app.py).

    Messages sans emoji : certains terminaux n affichent pas les caracteres
    non ASCII, ce qui rendait les logs illisibles (mojibake).
    """
    logging.basicConfig(
        level=niveau,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

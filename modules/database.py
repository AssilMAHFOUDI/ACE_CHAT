import logging

import streamlit as st
from supabase import create_client

from modules.config import MATCH_COUNT, MATCH_THRESHOLD

logger = logging.getLogger(__name__)


@st.cache_resource
def init_connection():
    """
    Initialise et met en cache la connexion à Supabase.
    """
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


def get_chat_history(supabase_client, session_id):
    """
    Récupère l'historique des messages pour une session donnée.

    Une base injoignable ne doit pas empêcher le démarrage de l'application :
    on journalise l'échec et on renvoie une liste vide (historique vide).
    """
    try:
        reponse_db = (
            supabase_client.table("chat_history")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        )
    except Exception as erreur:
        logger.error("Historique illisible pour la session %s : %s", session_id, erreur)
        return []
    return reponse_db.data


def save_message(supabase_client, session_id, role, content):
    """
    Enregistre un nouveau message dans la base de données.
    """
    supabase_client.table("chat_history").insert(
        {"session_id": session_id, "role": role, "content": content}
    ).execute()


def clear_chat_history(supabase_client, session_id):
    """
    Supprime tous les messages d'une session spécifique.
    """
    supabase_client.table("chat_history").delete().eq(
        "session_id", session_id
    ).execute()


# La base sait-elle filtrer par document (paramètre p_file_name de la fonction
# SQL) ? Tant que la migration SQL du README n'est pas exécutée, l'application
# filtre elle-même et l'interface le signale.
_filtre_document_par_sql = {"disponible": True}


def filtre_document_disponible():
    """
    True si le filtre par document a bien été appliqué par la base de données.
    """
    return _filtre_document_par_sql["disponible"]


def search_relevant_chunks(
    supabase_client,
    query_embedding,
    session_id,
    file_name=None,
    match_threshold=MATCH_THRESHOLD,
    match_count=MATCH_COUNT,
):
    """
    Appelle la fonction SQL Supabase pour trouver les morceaux de documents
    les plus proches sémantiquement de la question, filtrés par session_id.

    Si `file_name` est fourni, la recherche est en plus limitée à ce document :
    une session ne peut donc jamais répondre avec les extraits d'un ancien
    document qui traînerait encore en base.

    La fonction SQL doit accepter le paramètre `p_file_name`
    (voir la section « Semantic search function » du README).
    """
    try:
        # On appelle la fonction SQL match_document_chunks que l'on a créée dans Supabase
        parametres = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "p_session_id": session_id,
        }
        if file_name:
            parametres["p_file_name"] = file_name

        response = supabase_client.rpc("match_document_chunks", parametres).execute()

        if file_name:
            # Le filtre par document est bien pris en charge par la base
            _filtre_document_par_sql["disponible"] = True

        return response.data
    except Exception as e:
        message = str(e)
        if file_name and ("p_file_name" in message or "PGRST202" in message):
            # La fonction SQL n'a pas encore été migrée : on interroge la session
            # entière puis on filtre nous-mêmes, pour ne pas bloquer l'application
            # et pour respecter malgré tout le document choisi par l'utilisateur.
            _filtre_document_par_sql["disponible"] = False
            logger.warning(
                "⚠️ Filtre par document indisponible côté base : la fonction SQL "
                "match_document_chunks n'accepte pas encore p_file_name. "
                "Exécute la migration SQL du README. Filtrage côté application."
            )
            resultats = search_relevant_chunks(
                supabase_client,
                query_embedding,
                session_id,
                None,
                match_threshold,
                max(match_count * 5, 20),
            )
            if not any(ligne.get("file_name") for ligne in resultats):
                # La base ne renvoie même pas le nom du document : impossible de
                # filtrer proprement, on rend la recherche de la session entière.
                return resultats[:match_count]
            return [
                ligne for ligne in resultats if ligne.get("file_name") == file_name
            ][:match_count]
        logger.error("Erreur lors de la recherche sémantique : %s", e)
        return []


def list_session_documents(supabase_client, session_id):
    """
    Liste les documents indexés pour une session.

    Retourne une liste triée de dictionnaires `{"file_name": ..., "chunks": n}`.
    Elle sert à afficher la base de connaissance dans l'interface et à savoir si
    un fichier est déjà indexé. Le comptage se fait côté Python : le volume par
    session reste faible (quelques dizaines à quelques centaines de morceaux).
    """
    try:
        reponse = (
            supabase_client.table("document_chunks")
            .select("file_name")
            .eq("session_id", session_id)
            .execute()
        )
    except Exception as e:
        logger.error(
            "❌ Impossible de lister les documents de la session %s : %s",
            session_id[:8],
            e,
        )
        return []

    compteurs = {}
    for ligne in reponse.data or []:
        nom = ligne.get("file_name") or "(sans nom)"
        compteurs[nom] = compteurs.get(nom, 0) + 1

    return [
        {"file_name": nom, "chunks": nombre}
        for nom, nombre in sorted(compteurs.items())
    ]


def clear_document_chunks(supabase_client, session_id, file_name=None):
    """
    Supprime les chunks de documents d'une session dans Supabase et journalise
    le résultat.

    Sans `file_name`, tous les chunks de la session sont supprimés (bouton
    « Recommencer la discussion ») ; avec `file_name`, seuls ceux de ce
    document le sont.

    Retourne True si la suppression a réussi, False sinon : un échec silencieux
    laisserait les anciens morceaux en base et fausserait la recherche
    sémantique en mélangeant deux documents.
    """
    try:
        requete = (
            supabase_client.table("document_chunks")
            .delete()
            .eq("session_id", session_id)
        )
        if file_name:
            requete = requete.eq("file_name", file_name)
        requete.execute()
        logger.info(
            "🧹 Chunks supprimés pour la session %s%s",
            session_id[:8],
            f" (document {file_name})" if file_name else "",
        )
        return True
    except Exception as e:
        logger.error(
            "❌ Erreur lors de la suppression des chunks (session %s) : %s",
            session_id[:8],
            e,
        )
        return False

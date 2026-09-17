import streamlit as st
from supabase import create_client


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
    """
    reponse_db = (
        supabase_client.table("chat_history")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
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

def search_relevant_chunks(supabase_client, query_embedding, session_id, match_threshold=0.3, match_count=4):
    """
    Appelle la fonction SQL Supabase pour trouver les morceaux de documents
    les plus proches sémantiquement de la question, filtrés par session_id.
    """
    try:
        # On appelle la fonction SQL match_document_chunks que l'on a créée dans Supabase
        response = supabase_client.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "p_session_id": session_id
            }
        ).execute()
        
        return response.data
    except Exception as e:
        print(f"❌ Erreur lors de la recherche sémantique : {e}")
        return []

def clear_document_chunks(supabase_client, session_id):
    """
    Supprime tous les chunks de documents associés à une session dans Supabase 
    et affiche le résultat dans le terminal.
    """
    try:
        #print(f"🧹 Tentative de suppression des chunks pour la session : {session_id}")
        response = supabase_client.table("document_chunks").delete().eq("session_id", session_id).execute()
        print(f"🧹 Document supprimé de la base pour la session {session_id[:8]}...")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression des chunks : {e}")
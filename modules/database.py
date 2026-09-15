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

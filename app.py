"""
app.py module used to launch the ACE CHAT tool.

Note: the RAG (document analysis) mode is disabled for this test branch.
This keeps the app lightweight (no embedding model, no fastembed) so it fits
within the Alwaysdata disk quota.
"""

import json
import logging
import urllib.request
import uuid

import streamlit as st

# --- IMPORTATION DE NOS MODULES ---
from modules.database import (
    init_connection,
    get_chat_history,
    save_message,
    clear_chat_history,
)
from modules.ai_engine_groq import (
    init_ai_client,
    format_history_for_gemini,
    get_ai_response,
)

# On force Python à afficher les logs INFO dans le terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def log_egress_ip():
    """
    Affiche dans les logs l'IP publique sortante (et sa géolocalisation) telle
    que vue depuis la machine qui exécute réellement ce code.
    """
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=10) as response:
            data = json.loads(response.read().decode())
        logger.info(
            "🌍 [EGRESS IP] ip=%s country=%s city=%s org=%s",
            data.get("ip"),
            data.get("country"),
            data.get("city"),
            data.get("org"),
        )
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("⚠️ [EGRESS IP] Impossible de récupérer l'IP sortante : %s", exc)
        return {}


# On log l'IP sortante au démarrage pour qu'elle apparaisse dans les logs du run server
log_egress_ip()

# --- 1. INITIALISATION DES OUTILS ---
supabase = init_connection()
cle_api = st.secrets["GROQ_API_KEY"]
client = init_ai_client(cle_api)

# --- 2. GESTION DE LA SESSION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("🤖 ACE CHAT")

# --- 3. MÉMOIRE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # On délègue la récupération de l'historique au module database
    historique_db = get_chat_history(supabase, st.session_state.session_id)
    for row in historique_db:
        st.session_state.messages.append(
            {"role": row["role"], "content": row["content"]}
        )

# --- 4. BARRE LATÉRALE ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    st.info("ℹ️ Mode « Analyse de Document » (RAG) désactivé pour ce test.")
    st.divider()

    if st.button("🗑️ Recommencer la discussion"):
        # 1. On efface l'historique chat en base
        clear_chat_history(supabase, st.session_state.session_id)

        # 2. On génère un tout nouveau session_id pour repartir à zéro
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []

        st.rerun()

# --- 5. AFFICHAGE DES MESSAGES ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. GESTION D'UN NOUVEAU MESSAGE ---
if prompt := st.chat_input("Pose-moi une question..."):
    # A. Préparation de la question pour l'IA (pas de RAG -> prompt brut)
    prompt_pour_ia = prompt

    # B. Conversion de l'historique au format du moteur (OpenAI/Groq)
    gemini_history = format_history_for_gemini(st.session_state.messages)
    gemini_history.append({"role": "user", "content": prompt_pour_ia})

    # C. Affichage et Sauvegarde de la question utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(supabase, st.session_state.session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # D. Appel à l'IA et Sauvegarde de la réponse
    with st.chat_message("assistant"):
        try:
            with st.status(
                "L'Agent se met au travail...", expanded=True
            ) as status_box:

                def update_ui_status(message):
                    status_box.write(message)

                texte_reponse = get_ai_response(
                    client, gemini_history, status_callback=update_ui_status
                )
                status_box.update(
                    label="Réponse prête !", state="complete", expanded=True
                )
                st.markdown(texte_reponse)

            st.session_state.messages.append(
                {"role": "assistant", "content": texte_reponse}
            )
            save_message(
                supabase, st.session_state.session_id, "assistant", texte_reponse
            )

        except Exception as e:
            st.error(f"Erreur : {e}")
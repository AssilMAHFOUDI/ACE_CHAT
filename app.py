"""
app.py module used to laucnh ace chat tool
"""

import streamlit as st
import uuid
import logging

# --- IMPORTATION DE NOS NOUVEAUX MODULES ---
from modules.database import (
    init_connection,
    get_chat_history,
    save_message,
    clear_chat_history,
)
from modules.document_processor import extract_text_from_file
from modules.ai_engine import (
    init_ai_client,
    format_history_for_gemini,
    generate_rag_prompt,
    get_ai_response,
)


# On force Python à afficher les logs INFO dans le terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- 1. INITIALISATION DES OUTILS ---
supabase = init_connection()
cle_api = st.secrets["GEMINI_API_KEY"]
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

# --- 4. BARRE LATÉRALE (INTERFACE SEULEMENT) ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    mode = st.radio(
        "Choisis le mode :", ["💬 Chat Classique", "📄 Analyse de Document"]
    )
    st.divider()

    texte_document = ""
    if mode == "📄 Analyse de Document":
        fichier_upload = st.file_uploader("Charge ton document", type=["txt", "pdf"])
        # On délègue la lecture du fichier au module document_processor
        texte_document = extract_text_from_file(fichier_upload)
        if texte_document:
            st.success("Fichier chargé avec succès !")

    st.divider()
    if st.button("🗑️ Recommencer la discussion"):
        # On délègue la suppression au module database
        clear_chat_history(supabase, st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

# --- 5. AFFICHAGE DES MESSAGES ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. GESTION D'UN NOUVEAU MESSAGE ---
if prompt := st.chat_input("Pose-moi une question..."):
    # A. Préparation de la question pour l'IA
    if mode == "📄 Analyse de Document":
        if not texte_document:
            with st.chat_message("assistant"):
                st.warning(
                    "⚠️ Merci de charger un document dans le menu de gauche avant de poser une question."
                )
            st.stop()
        # On délègue la création du prompt caché au module ai_engine
        prompt_pour_ia = generate_rag_prompt(texte_document, prompt)
    else:
        prompt_pour_ia = prompt

    # B. Traduction de l'historique pour Gemini
    gemini_history = format_history_for_gemini(st.session_state.messages)
    gemini_history.append({"role": "user", "parts": [{"text": prompt_pour_ia}]})

    # C. Affichage et Sauvegarde de la question utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(
        supabase, st.session_state.session_id, "user", prompt
    )  # Délégation Database
    with st.chat_message("user"):
        st.markdown(prompt)

    # D. Appel à l'IA et Sauvegarde de la réponse
    with st.chat_message("assistant"):
        try:
            # 1. On crée la boîte de statut Streamlit
            with st.status("L'Agent se met au travail...", expanded=True) as status_box:
                # 2. On crée la fonction qui va écrire dans cette boîte
                def update_ui_status(message):
                    status_box.write(message)

                # On délègue la génération de texte au module ai_engine (AVEC LE CALLBACK !)
                texte_reponse = get_ai_response(
                    client, gemini_history, status_callback=update_ui_status
                )
                # 4. Quand c'est fini, on ferme et on met à jour le titre de la boîte
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

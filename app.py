<<<<<<< Updated upstream
"""
app.py module used to laucnh ace chat tool
"""
=======
import json
import logging
import urllib.request
>>>>>>> Stashed changes

import streamlit as st
import uuid
import logging

<<<<<<< Updated upstream
# --- IMPORTATION DE NOS NOUVEAUX MODULES ---
=======
logger = logging.getLogger(__name__)


def log_egress_ip():
    """
    Logs the public egress IP and its geolocation as seen from the machine
    that actually runs this code (e.g. the Alwaysdata 'run server').

    This is used to diagnose Gemini's
    '400 FAILED_PRECONDITION: User location is not supported for the API use'
    error, which depends on the *origin IP* of the API request.
    """
    try:
        with urllib.request.urlopen(
            "https://ipinfo.io/json", timeout=10
        ) as response:
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
        logger.warning("⚠️ [EGRESS IP] Could not fetch egress IP: %s", exc)
        return {}


# Log the egress IP once at startup so it appears in the run-server logs
log_egress_ip()


# 1. L'en-tête de la page
st.title("🤖 ACE CHAT ")
>>>>>>> Stashed changes

from modules.database import (
    init_connection,
    get_chat_history,
    save_message,
    clear_chat_history,
    search_relevant_chunks,
    clear_document_chunks,
)
from modules.document_processor import (
    extract_text_from_file,
    process_and_store_document  # 💡 NOUVEAU
)
from modules.ai_engine import (
    init_ai_client,
    format_history_for_gemini,
    generate_rag_prompt,
    get_ai_response,
    get_embedding,
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
        
        if fichier_upload:
            # 1. On extrait le texte (comme avant)
            texte_document = extract_text_from_file(fichier_upload)
            
            # 2. On vérifie si ce fichier a DÉJÀ été traité dans cette session
            if "fichier_traite" not in st.session_state or st.session_state.fichier_traite != fichier_upload.name:
                
                # On affiche un petit spinner pendant que Gemini calcule les vecteurs
                with st.spinner("🧠 Découpage et vectorisation du document en cours..."):
                    process_and_store_document(
                        text=texte_document,
                        file_name=fichier_upload.name,
                        session_id=st.session_state.session_id,
                        supabase_client=supabase,
                        ai_client=client
                    )
                # On marque le fichier comme "traité" pour ne pas le refaire au prochain message
                st.session_state.fichier_traite = fichier_upload.name
                
            st.success("✅ Fichier prêt et mémorisé dans Supabase !")

    st.divider()
    if st.button("🗑️ Recommencer la discussion"):
        # 1. On nettoie les chunks dans Supabase avant de changer de session
        
        clear_document_chunks(supabase, st.session_state.session_id)
        
        # 2. On efface l'historique chat en base
        clear_chat_history(supabase, st.session_state.session_id)
        
        # 3. On génère un tout nouveau session_id pour repartir à zéro
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        
        # 4. On oublie le fichier traité
        if "fichier_traite" in st.session_state:
            del st.session_state["fichier_traite"]
            
        st.rerun()


    
# --- 5. AFFICHAGE DES MESSAGES ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. GESTION D'UN NOUVEAU MESSAGE ---
if prompt := st.chat_input("Pose-moi une question sur ton document..."):
    
    # A. Préparation de la question pour l'IA
    if mode == "📄 Analyse de Document":
        if "fichier_traite" not in st.session_state:
            with st.chat_message("assistant"):
                st.warning("⚠️ Merci de charger un document dans le menu de gauche avant de poser une question.")
            st.stop()
            
        # 1. On transforme la question de l'utilisateur en vecteur
        with st.spinner("🔍 Recherche des passages pertinents dans le document..."):
            
            question_vector = get_embedding(prompt, client)
            
            # 2. On interroge Supabase pour trouver les morceaux les plus proches
            relevant_chunks = search_relevant_chunks(
                supabase_client=supabase,
                query_embedding=question_vector,
                session_id=st.session_state.session_id
            )
            
        if not relevant_chunks:
            prompt_pour_ia = f"L'utilisateur pose cette question : {prompt}, mais aucun extrait pertinent n'a été trouvé dans le document."
        else:
            # 3. On génère le prompt RAG intelligent avec les extraits ciblés
            prompt_pour_ia = generate_rag_prompt(relevant_chunks, prompt)
    else:
        prompt_pour_ia = prompt

    # B. Traduction de l'historique pour Gemini
    gemini_history = format_history_for_gemini(st.session_state.messages)
    gemini_history.append({"role": "user", "parts": [{"text": prompt_pour_ia}]})

    # C. Affichage et Sauvegarde de la question utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(supabase, st.session_state.session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # D. Appel à l'IA et Sauvegarde de la réponse
    with st.chat_message("assistant"):
        try:
            with st.status("L'Agent analyse les extraits...", expanded=True) as status_box:
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
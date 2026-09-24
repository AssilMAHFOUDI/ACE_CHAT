"""
app.py: Streamlit entry point for ACE CHAT (UI only; logic lives in modules/ and services/)
"""

import logging

import streamlit as st

# --- IMPORTATION DE NOS NOUVEAUX MODULES ---
from modules.ai_engine import (
    format_history_for_gemini,
    get_ai_response,
    init_ai_client,
)
from modules.config import configurer_logging
from modules.database import init_connection
from services.base_connaissance import (
    document_deja_indexe,
    documents_indexes,
    indexer_document,
    supprimer_document,
)
from services.recherche import (
    chercher_passages,
    construire_prompt,
    extraire_sources,
    filtre_applique_par_base,
)
from services.session import (
    charger_historique,
    enregistrer_message,
    nouvel_identifiant,
    reinitialiser_session,
)

# On force Python à afficher les logs INFO dans le terminal
configurer_logging()
logger = logging.getLogger(__name__)

# --- 1. INITIALISATION DES OUTILS ---
supabase = init_connection()
cle_api = st.secrets["GEMINI_API_KEY"]
client = init_ai_client(cle_api)

# --- 2. GESTION DE LA SESSION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = nouvel_identifiant()

st.title("🤖 ACE CHAT")

# --- 3. MÉMOIRE ---
if "messages" not in st.session_state:
    st.session_state.messages = charger_historique(
        supabase, st.session_state.session_id
    )

# --- 4. BARRE LATÉRALE (INTERFACE SEULEMENT) ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    mode = st.radio(
        "Choisis le mode :", ["💬 Chat Classique", "📄 Analyse de Document"]
    )
    st.divider()

    # Base de connaissance de la session : relue depuis Supabase à chaque
    # exécution (source de vérité), donc toujours à jour après un ajout ou une
    # suppression de document.
    documents = []
    filtre_document = None

    if mode == "📄 Analyse de Document":
        documents = documents_indexes(supabase, st.session_state.session_id)
        fichier_upload = st.file_uploader("Charge ton document", type=["txt", "pdf"])

        if fichier_upload:
            # On n'indexe que les documents absents de la base : Streamlit
            # réexécute tout le script à chaque message et le flux du fichier est
            # déjà consommé après la première lecture (pypdf le verrait vide).
            # Recharger un document déjà présent est ignoré (sinon il serait
            # réindexé à chaque message) : pour le mettre à jour, le supprimer
            # avec 🗑️ dans la liste ci-dessous, puis le recharger.

            if not document_deja_indexe(documents, fichier_upload.name):
                # On affiche une barre de progression pendant que Gemini calcule les vecteurs
                barre = st.progress(0.0, text="🧠 Découpage du document en cours...")

                def maj_progression(done, total):
                    barre.progress(
                        done / total,
                        text=f"🧠 Vectorisation : {done}/{total} morceaux...",
                    )

                try:
                    # La lecture est dans le try : un PDF vide ou corrompu donne
                    # un message clair au lieu d'une trace dans l'interface.
                    nb_chunks = indexer_document(
                        fichier_upload,
                        st.session_state.session_id,
                        supabase,
                        client,
                        progress_callback=maj_progression,
                    )
                except Exception as erreur:
                    nb_chunks = None
                    logger.exception("Échec de l'ingestion du document")
                    barre.empty()
                    st.error(f"❌ Impossible de mémoriser ce document : {erreur}")

                if nb_chunks:
                    barre.progress(1.0, text=f"✅ {nb_chunks} morceaux vectorisés")
                    # Le document rejoint la base tout de suite dans l'affichage ;
                    # au rendu suivant la liste est relue depuis Supabase.
                    documents.append(
                        {"file_name": fichier_upload.name, "chunks": nb_chunks}
                    )
                    documents.sort(key=lambda document: document["file_name"])
                elif nb_chunks == 0:
                    barre.empty()
                    st.warning(
                        "⚠️ Aucun texte exploitable : ce document ne contient pas de "
                        "couche texte (PDF composé d'images ou scan)."
                    )

            else:
                st.caption("📄 Ce document est déjà dans la base de connaissance.")

        st.divider()
        st.subheader("📚 Base de connaissance")
        if documents:
            for document in documents:
                colonne_nom, colonne_suppression = st.columns([5, 1])
                colonne_nom.write(
                    f"📄 {document['file_name']} — {document['chunks']} morceaux"
                )
                if colonne_suppression.button(
                    "🗑️",
                    key=f"supprimer_{document['file_name']}",
                    help="Supprimer ce document de la base",
                ):
                    if supprimer_document(
                        supabase, st.session_state.session_id, document["file_name"]
                    ):
                        st.rerun()
                    else:
                        st.error("❌ Suppression impossible (voir les logs).")

            options_recherche = ["📚 Tous les documents"] + [
                document["file_name"] for document in documents
            ]
            if st.session_state.get("filtre_document") not in options_recherche:
                st.session_state.filtre_document = options_recherche[0]
            choix_recherche = st.selectbox(
                "🔎 Chercher dans :", options_recherche, key="filtre_document"
            )
            if choix_recherche != options_recherche[0]:
                filtre_document = choix_recherche
        else:
            st.caption("Aucun document dans cette session pour le moment.")

    st.divider()
    if st.button("🗑️ Recommencer la discussion"):
        # 1. On efface chunks et historique, puis nouvel identifiant de session.
        st.session_state.session_id = reinitialiser_session(
            supabase, st.session_state.session_id
        )
        st.session_state.messages = []

        st.rerun()


# --- 5. AFFICHAGE DES MESSAGES ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. GESTION D'UN NOUVEAU MESSAGE ---
if prompt := st.chat_input("Pose-moi une question sur tes documents..."):
    sources = []

    # A. Préparation de la question pour l'IA
    if mode == "📄 Analyse de Document":
        if not documents:
            with st.chat_message("assistant"):
                st.warning(
                    "⚠️ Merci de charger un document dans le menu de gauche avant de poser une question."
                )
            st.stop()

        # 1. On transforme la question de l'utilisateur en vecteur
        with st.spinner("🔍 Recherche des passages pertinents dans la base..."):
            relevant_chunks = chercher_passages(
                prompt,
                st.session_state.session_id,
                supabase,
                client,
                file_name=filtre_document,
            )

        # Le filtre par document est-il réellement appliqué par la base ? Sinon on
        # le signale une fois par session (il faut exécuter la migration SQL).
        if filtre_applique_par_base():
            st.session_state.pop("averti_filtre_document", None)
        elif not st.session_state.get("averti_filtre_document"):
            st.session_state["averti_filtre_document"] = True
            st.warning(
                "⚠️ Le filtre « Chercher dans » n'est pas appliqué par la base : "
                "la fonction SQL `match_document_chunks` n'accepte pas encore "
                "`p_file_name`. Exécute la migration SQL du README. En attendant, "
                "le filtrage est fait côté application (résultat approché)."
            )

        sources = extraire_sources(relevant_chunks)
        prompt_pour_ia = construire_prompt(relevant_chunks, prompt)
    else:
        prompt_pour_ia = prompt

    # B. Traduction de l'historique pour Gemini
    gemini_history = format_history_for_gemini(st.session_state.messages)
    gemini_history.append({"role": "user", "parts": [{"text": prompt_pour_ia}]})

    # C. Affichage et Sauvegarde de la question utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    enregistrer_message(supabase, st.session_state.session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # D. Appel à l'IA et Sauvegarde de la réponse
    with st.chat_message("assistant"):
        try:
            with st.status(
                "L'Agent analyse les extraits...", expanded=True
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
            enregistrer_message(
                supabase, st.session_state.session_id, "assistant", texte_reponse
            )
            if sources:
                st.caption("📎 Sources : " + ", ".join(sources))

        except Exception as e:
            st.error(f"Erreur : {e}")

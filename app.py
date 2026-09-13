import pypdf
import uuid
import streamlit as st
from google import genai
from supabase import create_client


# --- CONFIGURATION SUPABASE ---
@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


supabase = init_connection()

# --- GESTION DE LA SESSION ---
# On génère un identifiant unique (UUID) pour le visiteur s'il n'en a pas encore
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 1. L'en-tête de la page
st.title("🤖 ACE CHAT ")

# 2. Connexion à l'API
cle_api = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=cle_api)

# 3. Initialisation de la mémoire (session_state + Supabase)
if "messages" not in st.session_state:
    # On va chercher l'historique de cette session dans la base
    reponse_db = (
        supabase.table("chat_history")
        .select("*")
        .eq("session_id", st.session_state.session_id)
        .order("created_at")
        .execute()
    )

    st.session_state.messages = []

    # On réinjecte les anciens messages dans la mémoire de Streamlit
    for row in reponse_db.data:
        st.session_state.messages.append(
            {"role": row["role"], "content": row["content"]}
        )

# Barre latérale pour effacer l'historique
with st.sidebar:
    st.header("⚙️ Paramètres")

    # 1. Le choix du mode
    mode = st.radio(
        "Choisis le mode de discussion :",
        ["💬 Chat Classique", "📄 Analyse de Document"],
    )

    st.divider()  # Une petite ligne de séparation esthétique

    # 2. Si le mode Document est choisi, on affiche le bouton d'upload
    texte_document = ""
    if mode == "📄 Analyse de Document":
        # On autorise maintenant les fichiers pdf en plus de txt
        fichier_upload = st.file_uploader("Charge ton document", type=["txt", "pdf"])
        # Si un fichier est chargé, on lit son contenu
        if fichier_upload is not None:
            # Traitement selon le format de fichier
            if fichier_upload.name.endswith(".txt"):
                texte_document = fichier_upload.getvalue().decode("utf-8")

            elif fichier_upload.name.endswith(".pdf"):
                pdf_reader = pypdf.PdfReader(fichier_upload)
                # Extrait le texte de chaque page et assemble le tout
                texte_document = "\n".join(
                    [
                        page.extract_text()
                        for page in pdf_reader.pages
                        if page.extract_text()
                    ]
                )
            st.success("Fichier chargé avec succès !")

    st.divider()

    # Bouton pour effacer l'historique
    if st.button("🗑️ Recommencer la discussion"):
        # NOUVEAU : On supprime les messages de cette session dans Supabase
        supabase.table("chat_history").delete().eq(
            "session_id", st.session_state.session_id
        ).execute()

        st.session_state.messages = []
        st.rerun()

# --- AFFICHAGE DE L'HISTORIQUE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- GESTION DE L'ENVOI D'UN MESSAGE ---
if prompt := st.chat_input("Pose-moi une question..."):
    # 1. PRÉPARATION DU MESSAGE POUR L'IA (La magie du RAG opère ici)
    if mode == "📄 Analyse de Document":
        # Sécurité : on vérifie que l'utilisateur a bien chargé un fichier
        if texte_document == "":
            with st.chat_message("assistant"):
                st.warning(
                    "⚠️ Merci de charger un document dans le menu de gauche avant de poser une question."
                )
            st.stop()  # On arrête l'exécution ici pour ne pas interroger Gemini pour rien

        # On crée le "Super Prompt" secret qui contient les règles + le document + la question
        prompt_pour_ia = f"""
        Tu es un assistant expert en analyse de documents.
        
        Voici le document de référence :
        --- DÉBUT DU DOCUMENT ---
        {texte_document}
        --- FIN DU DOCUMENT ---
        
        Consignes strictes :
        1. Utilise le document ci-dessus pour trouver les informations factuelles.
        2. MÉMOIRE VITAL : Sers-toi impérativement de l'historique de notre conversation pour comprendre le contexte de ma question (ex: de quelle entreprise on parle, traduction d'une réponse précédente).
        3. Si je pose une question très courte ("quand ?", "et en anglais ?"), c'est que je fais référence à ta réponse précédente.
        4. Si l'information factuelle n'est ni dans le document ni déductible de la conversation, réponds : "L'information n'est pas dans le document."
        5. Reponds moi dans la langue de la question par exemple si je pause la question en anglais reponds en anglais
        
        Ma question actuelle : {prompt}
        """
    else:
        # En mode Chat Classique, l'IA reçoit juste la question normale
        prompt_pour_ia = prompt

    # 2. TRADUCTION DE L'HISTORIQUE POUR GEMINI (Sans le -1 !)
    gemini_history = []
    for msg in st.session_state.messages:  # Plus besoin de couper le dernier message
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})

    # On ajoute la nouvelle question formatée pour l'IA
    gemini_history.append({"role": "user", "parts": [{"text": prompt_pour_ia}]})

    # 3. AFFICHAGE ET SAUVEGARDE VISUELLE (Déplacé ici, juste avant l'appel API)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # NOUVEAU : Sauvegarde du message utilisateur dans Supabase
    supabase.table("chat_history").insert(
        {"session_id": st.session_state.session_id, "role": "user", "content": prompt}
    ).execute()

    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. ENVOI À L'API ET AFFICHAGE DE LA RÉPONSE
    with st.chat_message("assistant"):
        try:
            reponse = client.models.generate_content(
                model="gemini-3.5-flash-lite", contents=gemini_history
            )
            texte_reponse = reponse.text
            st.markdown(texte_reponse)

            st.session_state.messages.append(
                {"role": "assistant", "content": texte_reponse}
            )

            # NOUVEAU : Sauvegarde de la réponse IA dans Supabase
            supabase.table("chat_history").insert(
                {
                    "session_id": st.session_state.session_id,
                    "role": "assistant",
                    "content": texte_reponse,
                }
            ).execute()

        except Exception as e:
            st.error(f"Erreur : {e}")

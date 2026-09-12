import streamlit as st
from google import genai

# 1. L'en-tête de la page
st.title("🤖 ACE CHAT ")

# 2. Connexion à l'API
cle_api = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=cle_api)

# 3. Initialisation de la mémoire (session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Barre latérale pour effacer l'historique
with st.sidebar:
    if st.button("🗑️ Recommencer la discussion"):
        st.session_state.messages = []
        st.rerun()

# 4. Afficher l'historique des messages précédents à l'écran
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. La zone où tu tapes ton texte
if prompt := st.chat_input("Pose-moi une question..."):
    # A. On ajoute et affiche le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. On formate l'historique pour l'API Gemini ("model" au lieu de "assistant")
    gemini_history = []
    for msg in st.session_state.messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})

    # C. On interroge Gemini en lui envoyant toute la conversation
    with st.chat_message("assistant"):
        try:
            reponse = client.models.generate_content(
                model="gemini-3.5-flash-lite", contents=gemini_history
            )
            texte_reponse = reponse.text
            st.markdown(texte_reponse)

            # D. On sauvegarde la réponse de l'assistant dans la mémoire
            st.session_state.messages.append(
                {"role": "assistant", "content": texte_reponse}
            )

        except Exception as e:
            st.error(f"Erreur : {e}")

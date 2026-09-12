import streamlit as st
from google import genai

# 1. L'en-tête de la page
st.title("🤖 ACE CHAT Amnesia")

# 2. Connexion à l'API (avec la clé de ton fichier secrets.toml)
cle_api = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=cle_api)

# 3. La zone où tu tapes ton texte
if prompt := st.chat_input("Pose-moi une question..."):
    # A. On affiche ta question à l'écran avec une icône utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. On interroge Gemini en lui envoyant juste ta question
    with st.chat_message("assistant"):
        reponse = client.models.generate_content(
            model="gemini-3.5-flash-lite", contents=prompt
        )
        # C. On affiche le texte de la réponse à l'écran
        st.markdown(reponse.text)

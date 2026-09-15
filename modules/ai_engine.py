from google import genai


def init_ai_client(api_key):
    """
    Initialise la connexion à l'API Gemini.
    """
    return genai.Client(api_key=api_key)


def format_history_for_gemini(st_messages):
    """
    Convertit l'historique de Streamlit au format attendu par l'API Gemini.
    """
    gemini_history = []
    for msg in st_messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
    return gemini_history


def generate_rag_prompt(texte_document, question):
    """
    Crée le prompt formaté avec les consignes strictes et le contexte du document.
    """
    prompt = f"""
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
    
    Ma question actuelle : {question}
    """
    return prompt


def get_ai_response(client, gemini_history):
    """
    Envoie l'historique complet au modèle Gemini et retourne le texte généré.
    """
    reponse = client.models.generate_content(
        model="gemini-3.5-flash-lite", contents=gemini_history
    )
    return reponse.text

import logging
import datetime
from google import genai
from google.genai import types
from modules.tools import calculatrice, meteo, recherche_web

logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = {
    "calculatrice": calculatrice,
    "meteo": meteo,
    "recherche_web": recherche_web,
}


def init_ai_client(api_key):
    return genai.Client(api_key=api_key)


def format_history_for_gemini(st_messages):
    gemini_history = []
    for msg in st_messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
    return gemini_history


# 💡 NOUVEAU : Ajout du paramètre status_callback
def get_ai_response(client, gemini_history, status_callback=None):
    last_msg = gemini_history[-1]
    latest_user_message = last_msg["parts"][0]["text"]
    history_for_session = gemini_history[:-1]

    # 💡 NOUVEAU : On récupère la date du jour
    date_du_jour = datetime.datetime.now().strftime("%A %d %B %Y")
    instruction = f"Tu es ACE CHAT. Nous sommes aujourd'hui le {date_du_jour}. Utilise cette date comme référence absolue pour toutes tes recherches temporelles."

    config = types.GenerateContentConfig(
        system_instruction=instruction,  # 💡 NOUVEAU : On donne l'instruction au modèle
        tools=[calculatrice, meteo, recherche_web],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    logger.info("🧠 Démarrage d'une nouvelle session Agent ReAct.")
    chat_session = client.chats.create(
        model="gemini-3.5-flash-lite", history=history_for_session, config=config
    )

    current_message = latest_user_message
    max_iterations = 10

    for iteration in range(max_iterations):
        logger.info(f"🔄 ReAct Loop - Itération {iteration + 1}")

        # 💡 Mise à jour de l'interface
        if status_callback:
            status_callback(f"🔄 Analyse et réflexion (Étape {iteration + 1})...")

        response = chat_session.send_message(current_message)

        if response.function_calls:
            tool_responses = []
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = function_call.args

                # 💡 Des messages personnalisés pour l'UI selon l'outil !
                if status_callback:
                    if tool_name == "recherche_web":
                        status_callback(
                            f"🌍 Recherche sur le web : `{tool_args.get('requete', '')}`"
                        )
                    elif tool_name == "meteo":
                        status_callback(
                            f"🌦️ Consultation de la météo pour `{tool_args.get('ville', '')}`"
                        )
                    elif tool_name == "calculatrice":
                        status_callback(
                            f"🧮 Calcul en cours : `{tool_args.get('expression', '')}`"
                        )
                    else:
                        status_callback(f"🛠️ Utilisation de l'outil : `{tool_name}`")

                logger.info(
                    f"🛠️  L'Agent décide d'utiliser : {tool_name} avec les arguments : {tool_args}"
                )

                if tool_name in AVAILABLE_TOOLS:
                    try:
                        tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=tool_name, response={"result": str(tool_result)}
                            )
                        )
                    except Exception as e:
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=tool_name, response={"error": str(e)}
                            )
                        )
                else:
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name, response={"error": "Tool not found"}
                        )
                    )
            current_message = tool_responses
        else:
            if status_callback:
                status_callback("💬 Rédaction de la réponse finale...")
            logger.info("💬 L'Agent a terminé son raisonnement et fournit une réponse.")
            return response.text

    return "Je suis désolé, le raisonnement était trop complexe et j'ai dû m'arrêter avant de trouver la réponse."


def generate_rag_prompt(relevant_chunks, user_question):
    """
    Crée un prompt enrichi en combinant uniquement les extraits pertinents 
    trouvés dans la base de données et la question de l'utilisateur.
    """
    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
    
    prompt = f"""
    Tu es un assistant IA professionnel. Tu dois répondre à la question de l'utilisateur en te basant **uniquement** sur le contexte fourni ci-dessous. 
    Si la réponse ne se trouve pas dans le contexte, dis honnêtement que tu ne sais pas, n'invente rien.

    --- CONTEXTE ---
    {context}
    --- FIN DU CONTEXTE ---

    Question de l'utilisateur : {user_question}
    """
    return prompt

def get_embedding(text, client):
    """
    Transforme un texte en vecteur (embedding) de 768 dimensions 
    en utilisant le modèle d'embedding de Gemini.
    """
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=text
    )
    # On retourne la liste des 768 nombres
    return response.embeddings[0].values
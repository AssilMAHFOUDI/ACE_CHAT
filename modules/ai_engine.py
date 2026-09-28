import datetime
import logging

from google import genai
from google.genai import types

from modules.config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    MAX_ITERATIONS,
    OBSERVATIONS_MAX_CHARS,
)
from modules.tools import calculatrice, meteo, recherche_web

logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = {
    "calculatrice": calculatrice,
    "meteo": meteo,
    "recherche_web": recherche_web,
}

# Configuration utilisée pour l'appel de synthèse forcée en fin de boucle :
# les outils y sont explicitement désactivés, de sorte que le modèle ne puisse
# plus lancer de recherche et doive produire du texte.
CONFIG_SANS_OUTILS = types.GenerateContentConfig(
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="NONE")
    )
)


def _formater_observations(observations):
    """
    Résume les observations des outils pour le message de synthèse forcée.

    Les observations les plus récentes sont conservées en priorité : le rappel
    est tronqué à OBSERVATIONS_MAX_CHARS pour garder la main sur la taille du
    prompt, et la troncature est signalée au modèle.
    """
    if not observations:
        return ""

    blocs = []
    total = 0
    for observation in reversed(observations):
        if total + len(observation) > OBSERVATIONS_MAX_CHARS:
            blocs.append("(... observations plus anciennes tronquées ...)")
            break
        blocs.append(observation)
        total += len(observation)

    blocs.reverse()
    return "\n".join(blocs)


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
    instruction = (
        f"Tu es ACE CHAT. Nous sommes aujourd'hui le {date_du_jour}. "
        "Utilise cette date comme référence absolue pour toutes tes recherches "
        "temporelles. Réponds toujours dans la langue de la question."
    )

    config = types.GenerateContentConfig(
        system_instruction=instruction,  # 💡 NOUVEAU : On donne l'instruction au modèle
        tools=[calculatrice, meteo, recherche_web],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    logger.info("Démarrage d'une nouvelle session Agent ReAct.")
    chat_session = client.chats.create(
        model=CHAT_MODEL, history=history_for_session, config=config
    )

    current_message = latest_user_message
    max_iterations = MAX_ITERATIONS
    observations = []  # résultats déjà obtenus des outils, réutilisés en synthèse

    for iteration in range(max_iterations):
        logger.info(f"ReAct Loop - Itération {iteration + 1}")

        # 💡 Mise à jour de l'interface
        if status_callback:
            status_callback(f"🔄 Analyse et réflexion (Étape {iteration + 1})...")

        response = chat_session.send_message(current_message)

        if response.function_calls:
            # Si on atteint la dernière itération autorisée, on force l'agent à synthétiser
            # au lieu de relancer des outils indéfiniment
            if iteration == max_iterations - 1:
                logger.warning(
                    "Limite d'itérations atteinte (%d) : forçage de synthèse finale.",
                    max_iterations,
                )
                if status_callback:
                    status_callback(
                        "⚠️ Limite de recherche atteinte : rédaction d'une synthèse avec les éléments disponibles..."
                    )

                message_forcage = (
                    "Limite d'itérations atteinte. N'appelle plus aucun outil. "
                    "Rédige immédiatement une synthèse claire et directe répondant au mieux "
                    "à la question initiale de l'utilisateur à partir des éléments collectés jusqu'ici, "
                    "en précisant les éventuelles incertitudes ou données manquantes."
                )
                rappel = _formater_observations(observations)
                if rappel:
                    message_forcage += (
                        "\n\nObservations déjà collectées auprès des outils :\n"
                        + rappel
                    )
                try:
                    reponse_synthese = chat_session.send_message(
                        message_forcage, config=CONFIG_SANS_OUTILS
                    )
                    if reponse_synthese.text:
                        return (
                            f"{reponse_synthese.text}\n\n"
                            "*(Note : Cette réponse est une synthèse établie après avoir atteint la limite de recherche.)*"
                        )
                except Exception as err_synthese:
                    logger.error("Échec de la synthèse forcée : %s", err_synthese)

                return (
                    "Je n'ai pas pu finaliser l'ensemble des étapes de recherche "
                    "dans la limite impartie, mais voici ce qui a pu être identifié."
                )

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
                        observations.append(f"{tool_name} → {tool_result}")
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=tool_name, response={"result": str(tool_result)}
                            )
                        )
                    except Exception as e:
                        observations.append(f"{tool_name} → erreur : {e}")
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=tool_name, response={"error": str(e)}
                            )
                        )
                else:
                    observations.append(f"{tool_name} → outil inconnu")
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name, response={"error": "Tool not found"}
                        )
                    )
            current_message = tool_responses
        else:
            if response.text:
                if status_callback:
                    status_callback("💬 Rédaction de la réponse finale...")
                logger.info(
                    "L'Agent a terminé son raisonnement et fournit une réponse."
                )
                return response.text

            # Réponse sans texte ni appel d'outil : on relance le modèle au lieu
            # de renvoyer None à l'interface.
            logger.warning("Réponse finale vide : relance du modèle.")
            current_message = (
                "Ta dernière réponse ne contenait aucun texte. "
                "Rédige maintenant la réponse finale à la question initiale."
            )

    return "Je suis désolé, le raisonnement était trop complexe et j'ai dû m'arrêter avant de trouver la réponse."


def generate_rag_prompt(relevant_chunks, user_question):
    """
    Crée un prompt enrichi en combinant uniquement les extraits pertinents
    trouvés dans la base de données et la question de l'utilisateur.
    """
    # On étiquette chaque extrait avec son document source : la session peut
    # contenir plusieurs documents et le modèle peut ainsi citer ses sources.
    blocs = []
    for chunk in relevant_chunks:
        source = chunk.get("file_name") if hasattr(chunk, "get") else None
        entete = f"[Extrait de {source}]" if source else "[Extrait]"
        blocs.append(f"{entete}\n{chunk['content']}")
    context = "\n\n".join(blocs)

    prompt = f"""
    Tu es un assistant IA professionnel. Tu dois répondre à la question de l'utilisateur en te basant **uniquement** sur le contexte fourni ci-dessous.
    Si la réponse ne se trouve pas dans le contexte, dis honnêtement que tu ne sais pas, n'invente rien.
    Quand plusieurs documents sont fournis, précise de quel extrait provient l'information.

    --- CONTEXTE ---
    {context}
    --- FIN DU CONTEXTE ---

    Question de l'utilisateur : {user_question}
    """
    return prompt


def get_embedding(text, client):
    """
    Transforme un texte en vecteur (embedding) de 3072 dimensions
    en utilisant le modèle d'embedding de Gemini.
    """
    return get_embeddings([text], client)[0]


def get_embeddings(texts, client, batch_size=20):
    """
    Vectorise plusieurs textes en un minimum d'appels à l'API Gemini.

    On envoie les morceaux par paquets de `batch_size` au lieu de faire un
    appel par morceau, ce qui accélère fortement l'ingestion d'un gros
    document.

    Attention : pour obtenir plusieurs vecteurs, il faut passer une liste
    d'objets `types.Content` explicites. Une simple liste de chaînes
    (`contents=["a", "b"]`) est interprétée par l'API comme UN SEUL contenu
    composé de deux parties, et un seul vecteur est alors renvoyé.

    Retourne la liste des vecteurs, dans le même ordre que `texts`.
    """
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[types.Content(parts=[types.Part(text=text)]) for text in batch],
        )
        if len(response.embeddings) != len(batch):
            # Sécurité : sans alignement, on associerait un vecteur au mauvais
            # morceau de texte dans la base.
            raise RuntimeError(
                f"L'API d'embedding a renvoyé {len(response.embeddings)} vecteurs "
                f"pour {len(batch)} textes : alignement impossible."
            )
        vectors.extend(embedding.values for embedding in response.embeddings)
    return vectors

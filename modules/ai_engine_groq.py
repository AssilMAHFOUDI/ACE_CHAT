"""Groq-based AI engine (OpenAI-compatible API) with a ReAct tool-calling loop.

This module mirrors the public interface of ``modules.ai_engine`` so it can be
swapped in by changing a single import in ``app.py``:

    from modules.ai_engine_groq import ( ... )

It uses:
- Groq (https://api.groq.com/openai/v1) for chat + tool calling.
- fastembed (local ONNX) for 768-dim embeddings, so the Supabase
  ``vector(768)`` schema is preserved without any embedding API.
"""

import json
import logging
import datetime

from openai import OpenAI
from fastembed import TextEmbedding

from modules.tools import calculatrice, meteo, recherche_web

logger = logging.getLogger(__name__)

# --- Chat model (Groq) ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"

# --- Embedding model (local, 768 dims) ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768 dims
FALLBACK_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --- Tool registry + OpenAI-style JSON schemas ---
AVAILABLE_TOOLS = {
    "calculatrice": calculatrice,
    "meteo": meteo,
    "recherche_web": recherche_web,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculatrice",
            "description": "Evalue une expression mathematique, ex: '(45 * 12) / 3'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "L'expression mathematique a evaluer.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meteo",
            "description": "Donne la meteo actuelle pour une ville dans le monde.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ville": {
                        "type": "string",
                        "description": "Le nom de la ville, ex: 'Paris', 'Tokyo'.",
                    }
                },
                "required": ["ville"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recherche_web",
            "description": (
                "Recherche sur Internet des actualites, scores ou infos recentes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {
                        "type": "string",
                        "description": "La requete de recherche web.",
                    }
                },
                "required": ["requete"],
            },
        },
    },
]

# Lazy singleton for the local embedding model (downloaded on first use)
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Chargement du modele d'embedding local : %s", EMBEDDING_MODEL)
        try:
            _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Modele %s indisponible (%s), repli sur %s",
                EMBEDDING_MODEL,
                exc,
                FALLBACK_EMBEDDING_MODEL,
            )
            _embedding_model = TextEmbedding(model_name=FALLBACK_EMBEDDING_MODEL)
    return _embedding_model


def init_ai_client(api_key):
    """Initialise le client Groq (compatible OpenAI)."""
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30)


def get_embedding(text, client=None):
    """Retourne un vecteur de 768 dimensions via fastembed (local, sans API).

    Le parametre ``client`` est conserve pour rester compatible avec les
    appels existants (``get_embedding(text, client)``) mais n'est pas utilise.
    """
    model = _get_embedding_model()
    vector = next(iter(model.embed([text])))
    return vector.tolist()


def format_history_for_gemini(st_messages):
    """Convertit l'historique Streamlit au format messages OpenAI.

    (Le nom est conserve pour rester compatible avec app.py.)
    """
    messages = []
    for msg in st_messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "content": msg["content"]})
    return messages


def _extract_last_user_message(history):
    """Recupere le dernier message utilisateur d'un historique OpenAI."""
    for msg in reversed(history):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_ai_response(client, history, status_callback=None):
    """Boucle ReAct avec Groq : le modele decide d'appeler des outils ou non."""
    messages = list(history)
    system_prompt = _build_system_prompt()
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})

    tools_list = list(AVAILABLE_TOOLS.keys())
    max_iterations = 10

    for iteration in range(max_iterations):
        logger.info("ReAct Loop - Iteration %s", iteration + 1)
        if status_callback:
            status_callback(f"Analyse et reflexion (Etape {iteration + 1})...")

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.3,
        )

        choice = response.choices[0].message

        if choice.tool_calls:
            messages.append(choice)
            for tool_call in choice.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_args = {}

                _notify_tool_use(status_callback, tool_name, tool_args)
                logger.info("L'Agent utilise l'outil %s : %s", tool_name, tool_args)

                tool_result = _run_tool(tool_name, tool_args, tools_list)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(tool_result),
                    }
                )
        else:
            if status_callback:
                status_callback("Redaction de la reponse finale...")
            logger.info("L'Agent a termine son raisonnement.")
            return choice.content

    return (
        "Je suis desole, le raisonnement etait trop complexe et j'ai du "
        "m'arreter avant de trouver la reponse."
    )


def _build_system_prompt():
    date_du_jour = datetime.datetime.now().strftime("%A %d %B %Y")
    return (
        "Tu es ACE CHAT, un assistant utile. "
        f"Nous sommes aujourd'hui le {date_du_jour}. "
        "Utilise cette date comme reference absolue pour toute recherche "
        "temporelle. Reponds dans la langue de la question."
    )


def _run_tool(tool_name, tool_args, tools_list):
    if tool_name not in tools_list:
        return f"Erreur : outil inconnu '{tool_name}'."
    try:
        return AVAILABLE_TOOLS[tool_name](**tool_args)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erreur d'execution de l'outil %s : %s", tool_name, exc)
        return f"Erreur lors de l'execution de l'outil {tool_name} : {exc}"


def _notify_tool_use(status_callback, tool_name, tool_args):
    if not status_callback:
        return
    if tool_name == "recherche_web":
        status_callback(f"Recherche sur le web : `{tool_args.get('requete', '')}`")
    elif tool_name == "meteo":
        status_callback(f"Consultation de la meteo pour `{tool_args.get('ville', '')}`")
    elif tool_name == "calculatrice":
        status_callback(f"Calcul en cours : `{tool_args.get('expression', '')}`")
    else:
        status_callback(f"Utilisation de l'outil : `{tool_name}`")


def generate_rag_prompt(relevant_chunks, user_question):
    """Construit un prompt RAG a partir des extraits pertinents."""
    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])

    prompt = f"""
    Tu es un assistant IA professionnel. Tu dois repondre a la question de
    l'utilisateur en te basant **uniquement** sur le contexte fourni ci-dessous.
    Si la reponse ne se trouve pas dans le contexte, dis honnetement que tu ne
    sais pas, n'invente rien.

    --- CONTEXTE ---
    {context}
    --- FIN DU CONTEXTE ---

    Question de l'utilisateur : {user_question}
    """
    return prompt
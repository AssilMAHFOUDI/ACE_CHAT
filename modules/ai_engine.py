import datetime
import json
import logging

from google import genai
from google.genai import types

from modules.config import (
    AGENT_CRITIQUE_ACTIVEE,
    AGENT_FLUX_ACTIVE,
    AGENT_PLAN_ACTIVEE,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    MAX_ITERATIONS,
    STAGNATION_MAX_APPELS_IDENTIQUES,
)
from modules.reflexion import (
    MARQUEUR_REPONSE,
    PROMPT_PLAN,
    Planification,
    analyser_planification,
    evaluer_brouillon,
    formater_observations,
    retirer_marqueur,
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


# Réponse de dernier recours quand la synthèse forcée échoue elle-même :
# l'utilisateur doit toujours recevoir quelque chose.
REPLI_SYNTHESE = (
    "Je n'ai pas pu finaliser l'ensemble des étapes de recherche "
    "dans la limite impartie, mais voici ce qui a pu être identifié."
)


def _statut_outil(nom, arguments):
    """Message d'étape affiché dans l'interface, propre à chaque outil."""
    if nom == "recherche_web":
        return f"🌍 Recherche sur le web : `{arguments.get('requete', '')}`"
    if nom == "meteo":
        return f"🌦️ Consultation de la météo pour `{arguments.get('ville', '')}`"
    if nom == "calculatrice":
        return f"🧮 Calcul en cours : `{arguments.get('expression', '')}`"
    return f"🛠️ Utilisation de l'outil : `{nom}`"


def _cle_appel(nom, arguments):
    """Clé stable d'un appel d'outil : mêmes nom et arguments, même appel.

    Les arguments viennent du modèle sous forme de mapping : l'ordre des clés est
    ignoré (tri) et toute valeur non sérialisable est repliée en texte, pour que
    deux appels identiques produisent toujours la même clé.
    """
    try:
        args = json.dumps(dict(arguments or {}), sort_keys=True, ensure_ascii=False)
    except TypeError:
        args = repr(arguments)
    return f"{nom} {args}"


class _ReponseFlux:
    """Réponse reconstruite à partir des morceaux d'un flux.

    Elle n'expose que ce que la boucle ReAct consomme (`text`, `function_calls`) :
    inutile de reconstruire les objets du SDK, et le flux devient testable avec
    des morceaux factices.
    """

    def __init__(self, texte, appels):
        self.text = texte
        self.function_calls = appels


def _envoyer(session, message, texte_callback=None, config=None):
    """Envoie un message à une session et renvoie la réponse complète.

    Sans `texte_callback`, l'appel est un `send_message` classique : rien n'est
    publié avant la réponse entière. Avec, la réponse est lue en flux et le texte
    déjà reçu est publié au fur et à mesure ; le callback reçoit le texte
    **accumulé du tour courant**, et un texte vide signifie « efface » (le tour a
    débouché sur un appel d'outil : son texte d'accompagnement n'est pas la
    réponse).

    La réponse renvoyée porte la même information qu'une réponse d'un seul bloc :
    la boucle ReAct n'a donc rien à savoir du mode d'envoi.
    """
    if texte_callback is None:
        return session.send_message(message, config=config)

    textes = []
    appels = []
    for morceau in session.send_message_stream(message, config=config):
        if morceau.text:
            textes.append(morceau.text)
        appels.extend(morceau.function_calls or [])
        texte_callback("" if appels else "".join(textes))

    return _ReponseFlux("".join(textes), appels)


def _planifier(client, history, question, texte_callback=None):
    """Établit le plan, ou récupère la réponse directe du planificateur.

    Le planificateur travaille dans sa propre session, outils coupés : il ne fait
    qu'arbitrer entre « je réponds » et « voici les étapes ». Le flux est routé au
    fil de l'eau : une réponse directe est publiée immédiatement (sans son
    marqueur), un plan n'est jamais publié comme réponse.

    Une panne n'interdit pas de répondre : elle renvoie une planification vide et
    l'agent agit alors seul (fail-open).
    """

    def router(texte_partiel):
        # Le plan est une étape de raisonnement, pas une réponse : tant que le
        # flux n'a pas désigné une réponse directe, la zone de réponse reste vide.
        if texte_callback is None:
            return
        marqueur, corps = retirer_marqueur(texte_partiel)
        texte_callback(corps if marqueur == MARQUEUR_REPONSE else "")

    # La session reçoit une copie de l'historique : le planificateur hérite du
    # contexte de la conversation (indispensable pour juger une question de
    # suivi) sans polluer la session de l'agent, qui partagera la liste d'origine.
    chat_plan = client.chats.create(
        model=CHAT_MODEL, history=list(history or []), config=CONFIG_SANS_OUTILS
    )
    invite = f"{PROMPT_PLAN}\n\nQuestion : {question}"

    try:
        reponse = _envoyer(chat_plan, invite, texte_callback=router)
    except Exception as erreur:
        logger.warning("Échec du plan (%s) : l'agent agit sans plan.", erreur)
        if texte_callback:
            texte_callback("")
        return Planification()

    texte = reponse.text or ""
    if texte_callback:
        router(texte)  # état final : publie la réponse, efface un flux d'étape
    return analyser_planification(texte)


def init_ai_client(api_key):
    return genai.Client(api_key=api_key)


def format_history_for_gemini(st_messages):
    gemini_history = []
    for msg in st_messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
    return gemini_history


def get_ai_response(client, gemini_history, status_callback=None, texte_callback=None):
    """Répond à la dernière question de l'historique, planification comprise.

    `status_callback` reçoit les étapes de raisonnement (ce que fait l'agent),
    `texte_callback` reçoit le texte de la réponse au fur et à mesure. Sans lui,
    ou si AGENT_FLUX_ACTIVE est coupé, la réponse est livrée d'un seul bloc.
    """
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

    def notifier(message):
        """Fait défiler une étape dans la zone de statut de l'interface."""
        if status_callback:
            status_callback(message)

    # Le flux n'est publié que si l'interface le demande et que le drapeau
    # l'autorise : sans callback, la réponse est livrée d'un seul bloc, comme
    # avant.
    publier_flux = texte_callback if AGENT_FLUX_ACTIVE else None

    def synthese_forcee(motif):
        """Contraint le modèle à conclure, outils coupés, avec le rappel des faits.

        Deux chemins mènent ici : les itérations épuisées sans conclusion, ou
        l'agent répété à l'identique (stagnation). Renvoie None si aucun texte
        n'obtient, à l'appelant de choisir le message de repli.
        """
        logger.warning("%s : forçage de synthèse finale.", motif)
        notifier(
            "⚠️ Limite de recherche atteinte : rédaction d'une synthèse avec les éléments disponibles..."
        )

        message = (
            "N'appelle plus aucun outil. Rédige immédiatement une synthèse claire "
            "et directe répondant au mieux à la question initiale de l'utilisateur "
            "à partir des éléments collectés jusqu'ici, en précisant les "
            "éventuelles incertitudes ou données manquantes."
        )
        rappel = formater_observations(observations)
        if rappel:
            message += "\n\nObservations déjà collectées auprès des outils :\n" + rappel
        if plan:
            message += f"\n\nPlan initial à couvrir :\n{plan}"

        try:
            texte = (
                _envoyer(
                    chat_session,
                    message,
                    texte_callback=publier_flux,
                    config=CONFIG_SANS_OUTILS,
                ).text
                or ""
            ).strip()
        except Exception as err_synthese:
            logger.error("Échec de la synthèse forcée : %s", err_synthese)
            return None
        if not texte:
            return None
        return (
            f"{texte}\n\n"
            "*(Note : Cette réponse est une synthèse établie après avoir atteint la limite de recherche.)*"
        )

    # Planification : elle est systématique et précède la boucle. Le
    # planificateur peut aussi répondre directement, quand la réponse se trouve
    # déjà dans la question ou dans le contexte fourni : la boucle ne s'ouvre
    # alors pas du tout et la question ne coûte qu'un appel.
    question_initiale = latest_user_message
    plan = None
    if AGENT_PLAN_ACTIVEE:
        notifier("🧠 Création du plan de réponse...")
        planification = _planifier(
            client,
            history_for_session,
            question_initiale,
            texte_callback=publier_flux,
        )
        if planification.reponse:
            notifier("💬 Réponse directe : aucun outil n'était nécessaire.")
            logger.info("Le planificateur a répondu directement à la question.")
            return planification.reponse

        plan = planification.plan
        if plan:
            notifier(f"🗺️ Plan retenu :\n{plan}")
            latest_user_message = (
                f"[Plan à suivre]\n{plan}\n\nQuestion : {question_initiale}"
            )
        else:
            notifier("🗺️ Aucun plan retenu : l'agent décide seul.")
            logger.info("Planificateur sans verdict : l'agent décide seul.")

    logger.info("Démarrage d'une nouvelle session Agent ReAct.")
    chat_session = client.chats.create(
        model=CHAT_MODEL, history=history_for_session, config=config
    )

    current_message = latest_user_message
    max_iterations = MAX_ITERATIONS
    observations = []  # résultats déjà obtenus des outils, réutilisés en synthèse
    # Résultats déjà obtenus et nombre de fois que le même appel a été demandé :
    # un modèle qui répète mot pour mot sa requête ne doit pas repayer le réseau.
    resultats_par_appel = {}
    repetitions = {}
    critique_faite = False  # une seule auto-critique par question

    for iteration in range(max_iterations):
        logger.info(f"ReAct Loop - Itération {iteration + 1}")

        notifier(f"🔄 Analyse et réflexion (Étape {iteration + 1})...")

        response = _envoyer(chat_session, current_message, texte_callback=publier_flux)

        if response.function_calls:
            # Dernière itération autorisée : on force l'agent à synthétiser au
            # lieu de relancer des outils indéfiniment.
            if iteration == max_iterations - 1:
                synthese = synthese_forcee(
                    f"Limite d'itérations atteinte ({max_iterations})"
                )
                return synthese or REPLI_SYNTHESE

            tool_responses = []
            stagnation = False
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = function_call.args or {}
                cle = _cle_appel(tool_name, tool_args)
                repetitions[cle] = repetitions.get(cle, 0) + 1

                # 💡 Des messages personnalisés pour l'UI selon l'outil !
                notifier(_statut_outil(tool_name, tool_args))
                logger.info(
                    f"🛠️  L'Agent décide d'utiliser : {tool_name} avec les arguments : {tool_args}"
                )

                if cle in resultats_par_appel:
                    # Même outil, mêmes arguments, résultat déjà en main : aucune
                    # nouvelle requête réseau. Le modèle retrouve le résultat,
                    # plus une pichenette pour qu'il change d'angle au lieu de
                    # tourner en rond jusqu'à la fin du budget.
                    resultat = resultats_par_appel[cle]
                    stagnation = stagnation or (
                        repetitions[cle] > STAGNATION_MAX_APPELS_IDENTIQUES
                    )
                    observations.append(f"{tool_name} → (déjà appelé) {resultat}")
                    notifier("♻️ Résultat déjà obtenu : nouvelle requête évitée.")
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={
                                "result": resultat,
                                "avertissement": (
                                    "Cet appel a déjà été effectué avec exactement "
                                    "ces arguments. Change d'approche ou rédige ta "
                                    "réponse."
                                ),
                            },
                        )
                    )
                    continue

                if tool_name in AVAILABLE_TOOLS:
                    try:
                        tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    except Exception as e:
                        # Échec volontairement non mis en cache : une panne réseau
                        # mérite d'être retentée au tour suivant.
                        observations.append(f"{tool_name} → erreur : {e}")
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=tool_name, response={"error": str(e)}
                            )
                        )
                        continue
                    observations.append(f"{tool_name} → {tool_result}")
                    resultats_par_appel[cle] = str(tool_result)
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name, response={"result": str(tool_result)}
                        )
                    )
                else:
                    observations.append(f"{tool_name} → outil inconnu")
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name, response={"error": "Tool not found"}
                        )
                    )

            if stagnation:
                # L'agent répète le même appel sans avancer : plutôt que d'épuiser
                # le budget en requêtes inutiles, il conclut avec ce qu'il sait.
                synthese = synthese_forcee("Agent bloqué sur un appel déjà exécuté")
                return synthese or REPLI_SYNTHESE

            current_message = tool_responses
        else:
            if response.text:
                brouillon = response.text
                # Auto-critique : une réponse obtenue grâce aux outils est relue
                # avant envoi, une seule fois par question et outils coupés. Sa
                # panne ne retient jamais la réponse (fail-open) : elle ne peut
                # que demander une reprise, jamais bloquer une conversation.
                if AGENT_CRITIQUE_ACTIVEE and observations and not critique_faite:
                    critique_faite = True
                    notifier("🧪 Vérification de la réponse avant envoi...")
                    valide, raison = evaluer_brouillon(
                        question_initiale,
                        observations,
                        brouillon,
                        lambda invite: (
                            chat_session.send_message(
                                invite, config=CONFIG_SANS_OUTILS
                            ).text
                        ),
                    )
                    if not valide:
                        notifier(f"🧪 Réponse jugée insuffisante : {raison}")
                        current_message = (
                            f"Auto-critique : {raison}. Reprends le raisonnement et "
                            "corrige la réponse en t'appuyant sur les observations "
                            "déjà obtenues, en appelant l'outil qui manque au besoin."
                        )
                        continue
                if publier_flux is None:
                    # La réponse part d'un bloc : on annonce sa rédaction. En
                    # flux, elle s'écrit déjà sous les yeux de l'utilisateur.
                    notifier("💬 Rédaction de la réponse finale...")
                logger.info(
                    "L'Agent a terminé son raisonnement et fournit une réponse."
                )
                return brouillon

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

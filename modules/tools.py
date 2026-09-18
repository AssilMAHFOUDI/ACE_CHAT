import logging
import urllib.request
import urllib.parse
import json

from ddgs import DDGS
from simpleeval import simple_eval, InvalidExpression

logger = logging.getLogger(__name__)



def calculatrice(expression: str) -> str:
    """
    Évalue une expression mathématique de manière sécurisée grâce à simpleeval.
    Empêche toute injection de code malveillant.
    """
    try:
        # Nettoyage basique de l'expression
        expression = expression.strip()
        
        # Évaluation sécurisée
        resultat = simple_eval(expression)
        return str(resultat)
        
    except (InvalidExpression, ZeroDivisionError, SyntaxError, TypeError) as e:
        return f"Erreur de calcul : Expression invalide ou non supportée ({e})"
    except Exception as e:
        return f"Erreur critique lors du calcul : {str(e)}"


def recherche_web(requete: str) -> str:
    """
    Effectue une recherche sur Internet via DuckDuckGo pour trouver des actualités, des scores sportifs ou des informations récentes.
    Exemple de requête : 'score dernier match Real Madrid', 'actualité tech du jour'
    """
    logger.info(f"🛠️ [TOOL EXECUTED] Recherche web appelée pour : {requete}")
    try:
        # On demande les 3 premiers résultats du web
        results = list(DDGS().text(requete, max_results=3))

        if not results:
            return f"Aucun résultat trouvé pour '{requete}'."

        # On formate les résultats de manière propre pour le LLM
        extrait_resultats = []
        for res in results:
            extrait_resultats.append(
                f"- Titre : {res.get('title')}\n  Résumé : {res.get('body')}\n  Lien : {res.get('href')}"
            )

        reponse_formatee = "\n\n".join(extrait_resultats)
        logger.info("✅ Recherche web exécutée avec succès.")
        return reponse_formatee

    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche web pour {requete} : {e}")
        return f"Désolé, je n'ai pas réussi à faire la recherche sur Internet pour {requete}."




def meteo(ville: str) -> str:
    """
    Donne la météo actuelle pour une ville donnée dans le monde.
    Exemple de ville : 'Paris', 'Madrid', 'Tokyo', 'Dakar'
    """
    logger.info(f"🛠️ [TOOL EXECUTED] Météo appelée pour la ville : {ville}")
    try:
        # 1. On cherche d'abord les coordonnées géographiques (latitude/longitude) de la ville
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(ville)}&count=1&language=fr&format=json"

        with urllib.request.urlopen(geo_url) as response:
            geo_data = json.loads(response.read().decode())

        if not geo_data.get("results"):
            return f"Je n'ai pas trouvé la ville de '{ville}'."

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        nom_officiel = location.get("name", ville)
        pays = location.get("country", "")

        # 2. On interroge la météo actuelle avec ces coordonnées
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code"

        with urllib.request.urlopen(weather_url) as response:
            weather_data = json.loads(response.read().decode())

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")

        # Petit dictionnaire pour traduire les codes météo de l'API en français
        wmo_codes = {
            0: "Ciel dégagé",
            1: "Principalement dégagé",
            2: "Partiellement nuageux",
            3: "Couvert",
            45: "Brouillard",
            51: "Bruine légère",
            61: "Pluie légère",
            63: "Pluie modérée",
            95: "Orage",
        }
        description = wmo_codes.get(
            current.get("weather_code", 0), "Conditions variables"
        )

        resultat_meteo = f"Météo à {nom_officiel} ({pays}) : {temp}°C, {description}. Humidité : {humidity}%."
        logger.info(f"✅ Météo récupérée avec succès : {resultat_meteo}")
        return resultat_meteo

    except Exception as e:
        logger.error(
            f"❌ Erreur lors de la récupération de la météo pour {ville} : {e}"
        )
        return f"Désolé, je n'ai pas pu récupérer la météo pour {ville}."

import logging
import urllib.request
import urllib.parse
import json

logger = logging.getLogger(__name__)


def calculatrice(expression: str) -> str:
    """
    Évalue une expression mathématique.
    Exemple d'expression : '2 + 2', '(45 * 12) / 3'
    """
    logger.info(
        f"🛠️ [TOOL EXECUTED] Calculatrice appelée avec l'expression : {expression}"
    )
    try:
        resultat = eval(expression, {"__builtins__": None}, {})
        return str(resultat)
    except Exception as e:
        logger.error(f"❌ Erreur de calcul sur {expression} : {e}")
        return f"Erreur lors du calcul : {e}"


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

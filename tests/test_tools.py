"""Tests de modules/tools.py — tout est isolé du réseau (fakes)."""

import json
import urllib.request

import modules.tools as outils
from modules.tools import calculatrice, meteo, recherche_web

# ---------------------------------------------------------------------------
# Calculatrice (simpleeval, injections bloquées)
# ---------------------------------------------------------------------------

def test_calculatrice_operations():
    assert calculatrice("2 + 2") == "4"
    assert calculatrice("(45 * 12) / 3") == "180.0"


def test_calculatrice_division_par_zero():
    assert calculatrice("1/0").startswith("Erreur de calcul")


def test_calculatrice_expression_invalide():
    assert calculatrice("2 +").startswith("Erreur")


def test_calculatrice_injection_bloquee():
    # simpleeval refuse les appels arbitraires : __import__ ne s'exécute pas
    # et on retombe sur le message d'erreur d'interface.
    resultat = calculatrice('__import__("os").system("echo PWNED")')
    assert resultat.startswith("Erreur")
    assert "PWNED" not in resultat.replace("Erreur", "", 1) or resultat.startswith(
        "Erreur de calcul"
    )


# ---------------------------------------------------------------------------
# Recherche web (DDGS simulé)
# ---------------------------------------------------------------------------

class _FausseDDGS:
    resultats = []

    def text(self, requete, max_results=5):
        return iter(self.resultats)


def test_recherche_web_formate_les_resultats(monkeypatch):
    _FausseDDGS.resultats = [{
        "title": "Titre test", "body": "Resume du resultat",
        "href": "https://exemple.fr",
    }]
    monkeypatch.setattr(outils, "DDGS", _FausseDDGS)
    sortie = recherche_web("actualité IA")
    assert "- Titre : Titre test" in sortie
    assert "Résumé : Resume du resultat" in sortie
    assert "Lien : https://exemple.fr" in sortie


def test_recherche_web_aucun_resultat(monkeypatch):
    _FausseDDGS.resultats = []
    monkeypatch.setattr(outils, "DDGS", _FausseDDGS)
    assert "Aucun résultat" in recherche_web("zzzz inutile")


def test_recherche_web_erreur_reseau(monkeypatch):
    class _EnPanne(_FausseDDGS):
        def text(self, requete, max_results=5):
            raise OSError("reseau coupe")

    monkeypatch.setattr(outils, "DDGS", _EnPanne)
    assert "Désolé" in recherche_web("nimporte quoi")


# ---------------------------------------------------------------------------
# Météo (urlopen simulé : géocodage puis prévisions)
# ---------------------------------------------------------------------------

def _urlopen_factice(reponses):
    """reponses : {fragment d'URL: objet JSON}"""
    class _Reponse:
        def __init__(self, donnees):
            self._donnees = donnees

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self._donnees).encode()

    def _ouvrir(url):
        for fragment, donnees in reponses.items():
            if fragment in url:
                return _Reponse(donnees)
        raise AssertionError(f"URL inattendue : {url}")

    return _ouvrir


def test_meteo_resultat_complet(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _urlopen_factice({
            "geocoding-api": {"results": [{
                "latitude": 48.85, "longitude": 2.35,
                "name": "Paris", "country": "France"}]},
            "api.open-meteo.com": {"current": {
                "temperature_2m": 21.5, "relative_humidity_2m": 60,
                "weather_code": 1}},
        }),
    )
    sortie = meteo("Paris")
    assert (
        "Météo à Paris (France) : 21.5°C, Principalement dégagé. "
        "Humidité : 60%."
    ) == sortie


def test_meteo_code_inconnu(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _urlopen_factice({
            "geocoding-api": {"results": [{
                "latitude": 1.0, "longitude": 2.0,
                "name": "VilleX", "country": "PaysY"}]},
            "api.open-meteo.com": {"current": {
                "temperature_2m": 30.0, "relative_humidity_2m": 80,
                "weather_code": 42}},
        }),
    )
    assert "Conditions variables" in meteo("VilleX")


def test_meteo_ville_inconnue(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _urlopen_factice({"geocoding-api": {"results": []}}),
    )
    assert "Je n'ai pas trouvé la ville de 'Atlantis'." == meteo("Atlantis")


def test_calculatrice_erreur_inattendue(monkeypatch):
    # Une erreur hors de la 1re tuple (p. ex. panne interne de l'évaluateur)
    # doit retomber sur le 2e garde-fou, pas propager.
    import modules.tools as outils

    def _explose(_expression):
        raise ValueError("panne interne de l evaluateur")

    monkeypatch.setattr(outils, "simple_eval", _explose)
    resultat = calculatrice("1 + 1")
    assert resultat.startswith("Erreur critique")
    assert "panne interne" in resultat


def test_meteo_erreur_reseau(monkeypatch):
    def _ko(url):
        raise OSError("pas de reseau")

    monkeypatch.setattr(urllib.request, "urlopen", _ko)
    assert "Désolé, je n'ai pas pu récupérer la météo pour Paris." == meteo("Paris")

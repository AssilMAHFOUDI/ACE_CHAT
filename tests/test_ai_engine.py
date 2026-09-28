"""Tests de modules/ai_engine.py (format, embeddings, prompt, boucle ReAct)."""

import pytest

import modules.ai_engine as engine
from modules.ai_engine import (
    format_history_for_gemini,
    generate_rag_prompt,
    get_ai_response,
    get_embedding,
    get_embeddings,
    init_ai_client,
)
from modules.config import CHAT_MODEL, MAX_ITERATIONS, OBSERVATIONS_MAX_CHARS
from tests.conftest import reponse_finale, reponse_outil


def _histoire():
    """Historique Gemini : conversation + dernier message utilisateur courant."""
    return format_history_for_gemini(
        [
            {"role": "user", "content": "contexte"},
            {"role": "assistant", "content": "reponse"},
        ]
    ) + [{"role": "user", "parts": [{"text": "ma question"}]}]


# --- Historique ------------------------------------------------------------


def test_format_history_mappe_roles_au_format_gemini():
    h = format_history_for_gemini(
        [
            {"role": "user", "content": "salut"},
            {"role": "assistant", "content": "bonjour"},
            {"role": "systeme", "content": "x"},  # rôle inconnu -> "user"
        ]
    )
    assert [m["role"] for m in h] == ["user", "model", "user"]
    assert h[0]["parts"][0]["text"] == "salut"
    assert h[1]["parts"][0]["text"] == "bonjour"


# --- Embeddings ------------------------------------------------------------


def test_get_embeddings_par_lots_et_ordre_preserve(ia):
    textes = ["a" * i for i in range(1, 46)]  # 45 textes, longueurs distinctes
    vecteurs = get_embeddings(textes, ia)
    assert [e[2] for e in ia.journal] == [20, 20, 5]  # batch_size = 20
    assert [v[0] for v in vecteurs] == [float(i) for i in range(1, 46)]


def test_get_embeddings_alignement_faux_leve_une_erreur(ia):
    ia.models.nb_retourne = 19  # l'API "oublie" un vecteur
    with pytest.raises(RuntimeError, match="alignement"):
        get_embeddings(["a" * 5] * 20, ia)


def test_get_embedding_unitaire(ia):
    assert get_embedding("abcd", ia) == [4.0, 1.0, 2.0]


def test_init_ai_client_retourne_un_client_genai():
    client = init_ai_client("cle-de-test")
    assert hasattr(client, "models")
    assert hasattr(client, "chats")


# --- Prompt RAG ------------------------------------------------------------


def test_generate_rag_prompt_etiquette_et_consigne():
    prompt = generate_rag_prompt(
        [{"file_name": "CV.pdf", "content": "ligne A"}, {"content": "ligne B"}],
        "Question ?",
    )
    assert "[Extrait de CV.pdf]\nligne A" in prompt
    assert "[Extrait]\nligne B" in prompt
    assert "Question ?" in prompt
    assert "uniquement" in prompt  # consigne d'honnêteté


# --- Boucle ReAct ----------------------------------------------------------


def test_reponse_directe_sans_appel_outil(ia):
    ia.chats.reponses = [reponse_finale("Bonjour !")]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert texte == "Bonjour !"
    assert ia.chats.creation["model"] == CHAT_MODEL
    # history = conversation SANS le message courant (retiré par la boucle).
    assert len(ia.chats.creation["history"]) == 2
    assert any("Étape 1" in s for s in statuts)
    assert any("Rédaction" in s for s in statuts)


def test_appel_outil_calculatrice_renvoie_le_resultat(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "2 + 2"}),
        reponse_finale("Le résultat est 4."),
    ]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert texte == "Le résultat est 4."
    reponse_out = ia.chats.dernier.envoyes[1]  # message des retours d'outil
    assert reponse_out[0].function_response.name == "calculatrice"
    assert reponse_out[0].function_response.response == {"result": "4"}
    assert any("Calcul en cours" in s for s in statuts)


def test_appel_outil_arguments_invalides_renvoie_erreur(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {}),  # expression manquante
        reponse_finale("Desole."),
    ]
    assert get_ai_response(ia, _histoire()) == "Desole."
    reponse_out = ia.chats.dernier.envoyes[1]
    assert "error" in reponse_out[0].function_response.response


def test_outil_inconnu_renvoie_tool_not_found(ia):
    ia.chats.reponses = [
        reponse_outil("open_file", {}),  # halluciné par le modèle
        reponse_finale("Je ne peux pas."),
    ]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert texte == "Je ne peux pas."
    reponse_out = ia.chats.dernier.envoyes[1]
    assert reponse_out[0].function_response.response == {"error": "Tool not found"}
    assert any("Utilisation de l'outil" in s for s in statuts)  # branche generique


def test_statuts_des_outils_reseau_avec_outils_factice(monkeypatch, ia):
    # On remplace l'exécution réelle (réseau) tout en gardant les statuts UI.
    monkeypatch.setitem(
        engine.AVAILABLE_TOOLS, "recherche_web", lambda requete: "web-ok"
    )
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "meteo", lambda ville: "meteo-ok")
    ia.chats.reponses = [
        reponse_outil("recherche_web", {"requete": "actualités IA"}),
        reponse_outil("meteo", {"ville": "Paris"}),
        reponse_finale("Voilà."),
    ]
    statuts = []
    assert get_ai_response(ia, _histoire(), status_callback=statuts.append) == "Voilà."
    assert any("Recherche sur le web" in s for s in statuts)
    assert any("Consultation de la météo" in s for s in statuts)


def test_boucle_bornee_forcage_de_synthese_finale(ia):
    # Le modele demande un outil a chaque tour sans jamais conclure : au dernier
    # tour autorise, la boucle doit forcer une synthese au lieu de relancer un
    # outil (avant : abandon sec apres 10 appels).
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "1+1"})
        for _ in range(MAX_ITERATIONS)
    ] + [reponse_finale("Synthese partielle : 1+1 fait 2.")]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert "Synthese partielle : 1+1 fait 2." in texte
    assert "limite de recherche" in texte
    assert any("Limite de recherche atteinte" in s for s in statuts)
    # MAX_ITERATIONS envois + 1 envoi de synthese forcee : la boucle est bornee.
    assert len(ia.chats.dernier.envoyes) == MAX_ITERATIONS + 1


def test_synthese_forcee_desactive_les_outils(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "1+1"})
        for _ in range(MAX_ITERATIONS)
    ] + [reponse_finale("Voila.")]
    get_ai_response(ia, _histoire())
    config = ia.chats.dernier.configs[-1]
    # L'appel de synthese interdit tout nouvel appel d'outil au modele.
    assert config is engine.CONFIG_SANS_OUTILS
    mode = config.tool_config.function_calling_config.mode
    assert getattr(mode, "value", mode) == "NONE"
    # Les tours precedents, eux, gardent les outils disponibles.
    assert ia.chats.dernier.configs[0] is None


def test_synthese_forcee_sans_texte_renvoie_repli(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "1+1"})
        for _ in range(MAX_ITERATIONS)
    ] + [reponse_finale("")]  # synthese vide : on retombe sur le repli
    texte = get_ai_response(ia, _histoire())
    assert "Je n'ai pas pu finaliser" in texte


def test_synthese_forcee_en_echec_renvoie_repli(ia):
    # Aucune reponse en reserve pour l'appel de synthese : il leve, et l'agent
    # doit malgre tout repondre quelque chose au lieu de propager l'erreur.
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "1+1"})
        for _ in range(MAX_ITERATIONS)
    ]
    texte = get_ai_response(ia, _histoire())
    assert "Je n'ai pas pu finaliser" in texte
    assert len(ia.chats.dernier.envoyes) == MAX_ITERATIONS + 1


def test_boucle_desactivee_renvoie_message_dedie(monkeypatch, ia):
    # Cas limite : aucune iteration autorisee. On doit renvoyer le message de
    # repli sans jamais envoyer de message au modele.
    monkeypatch.setattr(engine, "MAX_ITERATIONS", 0)
    texte = get_ai_response(ia, _histoire())
    assert "trop complexe" in texte
    assert ia.chats.dernier.envoyes == []


def test_reponse_finale_vide_relance_le_modele(ia):
    # Sortie vide (ni texte ni outil) : on relance le modele au lieu de renvoyer
    # None a l'interface.
    ia.chats.reponses = [reponse_finale(""), reponse_finale("Voici la reponse.")]
    texte = get_ai_response(ia, _histoire())
    assert texte == "Voici la reponse."
    assert len(ia.chats.dernier.envoyes) == 2


# --- Synthese forcee : rappel des observations ------------------------------


def test_formater_observations_vide():
    assert engine._formater_observations([]) == ""


def test_formater_observations_tronque_les_plus_anciennes():
    # Rappel borne a OBSERVATIONS_MAX_CHARS : les observations recentes sont
    # gardees, la troncature est signalee au modele.
    observations = ["ancienne" * 200, "moyenne" * 200, "recente" * 200]
    rappel = engine._formater_observations(observations)
    assert "recente" * 200 in rappel
    assert "tronquées" in rappel
    assert "ancienne" * 200 not in rappel
    assert "moyenne" * 200 not in rappel
    assert len(rappel) <= OBSERVATIONS_MAX_CHARS + 120  # marge = marqueur


def test_synthese_forcee_rappelle_les_observations(ia):
    # Les resultats deja obtenus sont reinjectes dans le message de synthese :
    # l'agent peut conclure meme si l'historique de session est volumineux.
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "2 + 2"})
        for _ in range(MAX_ITERATIONS)
    ] + [reponse_finale("Synthese.")]
    get_ai_response(ia, _histoire())
    message_synthese = ia.chats.dernier.envoyes[-1]
    assert "Observations" in message_synthese
    assert "calculatrice" in message_synthese

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
from modules.config import CHAT_MODEL, MAX_ITERATIONS
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


def test_boucle_bornee_sans_reponse_finale(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "1+1"})
        for _ in range(MAX_ITERATIONS)
    ]
    texte = get_ai_response(ia, _histoire())  # sans status_callback
    assert "trop complexe" in texte
    # Un envoi par itération, pas d'appel infini.
    assert len(ia.chats.dernier.envoyes) == MAX_ITERATIONS

"""Tests de services/recherche.py (embedding -> RPC -> prompt -> sources)."""

from services.recherche import (
    chercher_passages,
    construire_prompt,
    extraire_sources,
    filtre_applique_par_base,
)


def test_chercher_passages_embed_la_question_et_interroge_la_base(supabase, ia):
    supabase.resultats_rpc["match_document_chunks"] = [
        {"id": 1, "file_name": "a.pdf", "content": "A"},
        {"id": 2, "file_name": "b.pdf", "content": "B"},
    ]
    question = "Qui est Assil ?"
    res = chercher_passages(question, "s1", supabase, ia, file_name="a.pdf")
    assert len(res) == 2
    # La question a bien été vectorisée (journal de l'IA).
    assert ia.journal[0][0] == "embed"
    # La RPC reçoit exactement ce vecteur + le couple session/document.
    nom, params = supabase.appels_rpc[0]
    assert nom == "match_document_chunks"
    vecteur_attendu = [float(len(question)), 1.0, 2.0]  # vecteur du FakeIA
    assert params["query_embedding"] == vecteur_attendu
    assert params["p_session_id"] == "s1"
    assert params["p_file_name"] == "a.pdf"


def test_chercher_passages_sans_filtre(supabase, ia):
    supabase.resultats_rpc["match_document_chunks"] = []
    assert chercher_passages("q", "s1", supabase, ia) == []
    _, params = supabase.appels_rpc[0]
    assert "p_file_name" not in params


def test_extraire_sources_tri_et_supprime_doublons():
    chunks = [
        {"file_name": "b.pdf"},
        {"file_name": "a.pdf"},
        {"file_name": "a.pdf"},   # doublon -> fusionné
        {},                       # sans source -> ignoré
        {"file_name": ""},        # nom vide -> ignoré
    ]
    assert extraire_sources(chunks) == ["a.pdf", "b.pdf"]
    assert extraire_sources([]) == []


def test_construire_prompt_avec_extraits_etiquetes():
    chunks = [
        {"file_name": "CV.pdf", "content": "Assil est ingenieur."},
        {"content": "extrait sans nom"},   # sans file_name -> [Extrait]
    ]
    prompt = construire_prompt(chunks, "Qui est Assil ?")
    assert "[Extrait de CV.pdf]" in prompt
    assert "[Extrait]" in prompt
    assert "Assil est ingenieur." in prompt
    assert "Qui est Assil ?" in prompt
    assert "uniquement" in prompt          # consigne d'honnêteté du RAG


def test_construire_prompt_sans_extrait_préviens_du_repli():
    prompt = construire_prompt([], "Y a-t-il une info ?")
    assert "Y a-t-il une info ?" in prompt
    assert "aucun extrait" in prompt


def test_filtre_applique_par_base_par_defaut():
    assert filtre_applique_par_base() is True


def test_filtre_applique_par_base_apres_repli(supabase, ia):
    """Base non migrée : après repli, l'interface doit afficher l'avertissement."""

    def base_non_migree(nom, params):
        if "p_file_name" in params:
            raise RuntimeError("PGRST202 : argument p_file_name inconnu")
        return []

    supabase.gestionnaire_rpc = base_non_migree
    chercher_passages("q", "s1", supabase, ia, file_name="a.pdf")
    assert filtre_applique_par_base() is False


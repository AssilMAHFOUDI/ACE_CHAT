"""Tests de modules/database.py (historique, documents, recherche RPC)."""

from modules.database import (
    clear_chat_history,
    clear_document_chunks,
    filtre_document_disponible,
    get_chat_history,
    list_session_documents,
    save_message,
    search_relevant_chunks,
)

RETOUR_RPC = [
    {"id": 1, "file_name": "a.pdf", "content": "A1", "similarity": 0.9},
    {"id": 2, "file_name": "b.pdf", "content": "B1", "similarity": 0.8},
]


# ---------------------------------------------------------------------------
# Historique
# ---------------------------------------------------------------------------

def test_historique_trie_par_date(supabase):
    supabase.tables["chat_history"] = [
        {"session_id": "s1", "role": "user", "content": "2",
         "created_at": "2026-01-02"},
        {"session_id": "s1", "role": "assistant", "content": "1",
         "created_at": "2026-01-01"},
        {"session_id": "autre", "role": "user", "content": "X",
         "created_at": "2026-01-03"},
    ]
    lignes = get_chat_history(supabase, "s1")
    assert [l["content"] for l in lignes] == ["1", "2"]


def test_historique_table_absente_retourne_vide(supabase):
    supabase.echecs_select.add("chat_history")
    assert get_chat_history(supabase, "s1") == []


def test_enregistrement_message(supabase):
    save_message(supabase, "s1", "user", "Bonjour")
    assert supabase.tables["chat_history"] == [
        {"session_id": "s1", "role": "user", "content": "Bonjour"}
    ]


def test_effacement_historique_cible_la_session(supabase):
    supabase.tables["chat_history"] = [
        {"session_id": "s1", "role": "user", "content": "a"},
        {"session_id": "s2", "role": "user", "content": "b"},
    ]
    clear_chat_history(supabase, "s1")
    assert [l["session_id"] for l in supabase.tables["chat_history"]] == ["s2"]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def test_liste_documents_compte_et_range(supabase):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "b.pdf"},
        {"session_id": "s1", "file_name": "a.pdf"},
        {"session_id": "s1", "file_name": "a.pdf"},
        {"session_id": "s1"},  # file_name manquant -> "(sans nom)"
        {"session_id": "s2", "file_name": "z.pdf"},
    ]
    docs = list_session_documents(supabase, "s1")
    assert docs == [
        {"file_name": "(sans nom)", "chunks": 1},
        {"file_name": "a.pdf", "chunks": 2},
        {"file_name": "b.pdf", "chunks": 1},
    ]


def test_liste_documents_erreur_retourne_vide(supabase):
    supabase.echecs_select.add("document_chunks")
    assert list_session_documents(supabase, "s1") == []


def test_purge_document_ciblee(supabase):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "a.pdf"},
        {"session_id": "s1", "file_name": "b.pdf"},
        {"session_id": "s2", "file_name": "a.pdf"},
    ]
    assert clear_document_chunks(supabase, "s1", "a.pdf") is True
    restants = [(l["session_id"], l["file_name"])
                for l in supabase.tables["document_chunks"]]
    assert restants == [("s1", "b.pdf"), ("s2", "a.pdf")]


def test_purge_toute_la_session(supabase):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "a.pdf"},
        {"session_id": "s2", "file_name": "b.pdf"},
    ]
    assert clear_document_chunks(supabase, "s1") is True
    assert [l["session_id"] for l in supabase.tables["document_chunks"]] == ["s2"]


def test_purge_echec_retourne_false(supabase):
    supabase.echecs_delete.add("document_chunks")
    assert clear_document_chunks(supabase, "s1") is False


# ---------------------------------------------------------------------------
# Recherche sémantique (RPC) + repli
# ---------------------------------------------------------------------------

def test_recherche_passe_le_filtre_document(supabase):
    supabase.resultats_rpc["match_document_chunks"] = RETOUR_RPC
    res = search_relevant_chunks(supabase, [1.0], "s1", "a.pdf")
    assert [r["file_name"] for r in res] == ["a.pdf", "b.pdf"]
    _, params = supabase.appels_rpc[0]
    assert params["p_session_id"] == "s1"
    assert params["p_file_name"] == "a.pdf"
    assert params["query_embedding"] == [1.0]
    assert filtre_document_disponible() is True


def test_recherche_repli_filtre_cote_application(supabase):
    """Base non migrée : le repli est déclenché ET le filtre appliqué ici."""

    def base_non_migree(nom, params):
        if "p_file_name" in params:
            raise RuntimeError(
                "could not find function match_document_chunks(text) (PGRST202)"
            )
        return RETOUR_RPC

    supabase.gestionnaire_rpc = base_non_migree
    res = search_relevant_chunks(supabase, [1.0], "s1", "a.pdf", match_count=4)
    assert [r["file_name"] for r in res] == ["a.pdf"]   # document A uniquement
    assert len(supabase.appels_rpc) == 2                 # 1er échoué, repli ok
    _, repli = supabase.appels_rpc[1]
    assert "p_file_name" not in repli
    assert repli["match_count"] == 20                    # max(match_count * 5, 20)
    assert filtre_document_disponible() is False         # drapeau baissé


def test_recherche_repli_lignes_sans_nom_tronque(supabase):
    """Repli sur des lignes sans file_name : troncature à match_count."""

    def base_non_migree(nom, params):
        if "p_file_name" in params:
            raise RuntimeError("PGRST202 : argument p_file_name inconnu")
        return [{"id": 1}, {"id": 2}, {"id": 3}]

    supabase.gestionnaire_rpc = base_non_migree
    res = search_relevant_chunks(supabase, [1.0], "s1", "x.pdf", match_count=2)
    assert res == [{"id": 1}, {"id": 2}]                 # tronqué à match_count


def test_recherche_sans_filtre_renvoie_tout(supabase):
    """file_name=None : pas de repli, la RPC reçoit tout le résultat."""
    supabase.resultats_rpc["match_document_chunks"] = [
        {"id": 1}, {"id": 2}, {"id": 3},
    ]
    res = search_relevant_chunks(supabase, [1.0], "s1", None, match_count=2)
    assert res == [{"id": 1}, {"id": 2}, {"id": 3}]      # aucune troncature
    assert len(supabase.appels_rpc) == 1                 # un seul appel


def test_init_connection_utilise_les_secrets(monkeypatch):
    """st.secrets lu au bon moment : la connexion Supabase est bien construite."""
    from types import SimpleNamespace

    import modules.database as db

    faux_st = SimpleNamespace(
        secrets={"supabase": {"url": "http://localhost:54321", "key": "cle-test"}}
    )
    monkeypatch.setattr(db, "st", faux_st)
    client = db.init_connection()
    assert hasattr(client, "table")                   # vrai client supabase


def test_recherche_erreur_generique_retourne_vide(supabase):
    def en_panne(nom, params):
        raise RuntimeError("panne totale")

    supabase.gestionnaire_rpc = en_panne
    assert search_relevant_chunks(supabase, [1.0], "s1", None) == []
    assert search_relevant_chunks(supabase, [1.0], "s1", "a.pdf") == []

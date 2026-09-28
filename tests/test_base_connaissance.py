"""Tests de services/base_connaissance.py (liste, existence, indexation)."""

import pytest

from services.base_connaissance import (
    document_deja_indexe,
    documents_indexes,
    indexer_document,
    supprimer_document,
)
from tests.conftest import FakeIA, FakeSupabase, fichier


def test_documents_indexes_delegue_et_compte(supabase):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "b.pdf"},
        {"session_id": "s1", "file_name": "a.pdf"},
        {"session_id": "s1", "file_name": "a.pdf"},
        {"session_id": "s2", "file_name": "z.pdf"},
    ]
    assert documents_indexes(supabase, "s1") == [
        {"file_name": "a.pdf", "chunks": 2},
        {"file_name": "b.pdf", "chunks": 1},
    ]


def test_document_deja_indexe(supabase):
    docs = [{"file_name": "a.pdf", "chunks": 3}]
    assert document_deja_indexe(docs, "a.pdf") is True
    assert document_deja_indexe(docs, "b.pdf") is False
    assert document_deja_indexe([], "a.pdf") is False


def test_documents_indexes_ignore_les_lignes_corrompues(supabase):
    # Une ligne invalide en base (nom de fichier non textuel) ne doit pas faire
    # echouer toute la liste : les autres documents restent utilisables.
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "ok.pdf"},
        {"session_id": "s1", "file_name": 123},  # non conforme au schema
    ]
    assert documents_indexes(supabase, "s1") == [{"file_name": "ok.pdf", "chunks": 1}]


def test_indexation_remplace_uniquement_le_document_recharge():
    # Journal partagé : on prouve l'ordre embedding -> purge -> insertion.
    timeline = []
    supabase = FakeSupabase(journal=timeline)
    ia = FakeIA(timeline)
    supabase.tables["document_chunks"] = [
        {
            "session_id": "s1",
            "file_name": "A.txt",
            "content": "ANCIEN",
            "embedding": [0.0],
        },
        {
            "session_id": "s1",
            "file_name": "B.txt",
            "content": "B-INTACT",
            "embedding": [0.0],
        },
    ]
    contenu = "Bonjour, ceci est un document de test suffisamment long."
    n = indexer_document(
        fichier_upload=fichier("A.txt", contenu),
        session_id="s1",
        supabase_client=supabase,
        ai_client=ia,
    )
    assert n == 1
    doc_a = [
        row for row in supabase.tables["document_chunks"] if row["file_name"] == "A.txt"
    ]
    doc_b = [
        row for row in supabase.tables["document_chunks"] if row["file_name"] == "B.txt"
    ]
    assert len(doc_a) == 1 and "ANCIEN" not in doc_a[0]["content"]
    assert doc_b[0]["content"] == "B-INTACT"  # l'autre document, intact
    # Ordre critique : vectoriser PUIS purger AVANT d'écrire (jamais l'inverse).
    evenements = [e[0] for e in timeline]
    assert evenements == ["embed", "delete", "insert"]


def test_indexation_fichier_illisible_ne_touche_rien(supabase, ia):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "vide.pdf"},
    ]
    supabase.journal = []
    with pytest.raises(ValueError, match="vide"):
        indexer_document(
            fichier_upload=fichier("vide.pdf", b""),
            session_id="s1",
            supabase_client=supabase,
            ai_client=ia,
        )
    assert supabase.journal == []
    assert supabase.tables["document_chunks"]  # inchangé


def test_indexation_signale_la_progression(supabase, ia):
    progres = []
    indexer_document(
        fichier_upload=fichier("a.txt", "Un contenu assez long pour un morceau."),
        session_id="s1",
        supabase_client=supabase,
        ai_client=ia,
        progress_callback=lambda a, b: progres.append((a, b)),
    )
    assert progres == [(1, 1)]


def test_supprimer_document(supabase):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "a.pdf"},
        {"session_id": "s1", "file_name": "b.pdf"},
    ]
    assert supprimer_document(supabase, "s1", "a.pdf") is True
    assert [row["file_name"] for row in supabase.tables["document_chunks"]] == ["b.pdf"]


def test_supprimer_document_echec(supabase):
    supabase.echecs_delete.add("document_chunks")
    assert supprimer_document(supabase, "s1", "a.pdf") is False


def test_documents_indexes_rejette_un_resume_invalide(supabase, monkeypatch):
    # Garde-fou Pydantic : un resume incoherent remonte par la base est ignore
    # sans faire echouer toute la liste. On patche la reference importee par le
    # service (celle de modules.database est deja liee depuis l'import).
    import services.base_connaissance as bc

    monkeypatch.setattr(
        bc,
        "list_session_documents",
        lambda client, session_id: [
            {"file_name": "valide.pdf", "chunks": 2},
            {"file_name": "corrompu.pdf", "chunks": -5},  # viole ge=0
        ],
    )
    assert documents_indexes(supabase, "s1") == [
        {"file_name": "valide.pdf", "chunks": 2}
    ]

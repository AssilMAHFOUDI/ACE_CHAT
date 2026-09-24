"""Tests de services/session.py (identifiants, historique, réinitialisation)."""

import uuid

from services.session import (
    charger_historique,
    enregistrer_message,
    nouvel_identifiant,
    reinitialiser_session,
)


def test_nouvel_identifiant_est_un_uuid4_valide():
    ident = nouvel_identifiant()
    u = uuid.UUID(ident)
    assert u.version == 4
    assert nouvel_identifiant() != ident  # jamais réutilisé


def test_charger_historique_mappe_roles_et_contenus(supabase):
    supabase.tables["chat_history"] = [
        {
            "session_id": "s1",
            "role": "user",
            "content": "a",
            "created_at": "2026-01-01",
        },
        {
            "session_id": "s1",
            "role": "assistant",
            "content": "b",
            "created_at": "2026-01-02",
        },
        {
            "session_id": "s1",
            "role": "systeme-corrompu",
            "content": "?",
            "created_at": "2026-01-03",
        },
    ]
    historique = charger_historique(supabase, "s1")
    # Rôles hors de {user, assistant, model} : ligne ignorée.
    assert historique == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_enregistrer_message_persiste(supabase):
    enregistrer_message(supabase, "s1", "assistant", "Réponse")
    assert supabase.tables["chat_history"][0] == {
        "session_id": "s1",
        "role": "assistant",
        "content": "Réponse",
    }


def test_reinitialiser_session_efface_tout_et_change_id(supabase):
    supabase.tables["chat_history"] = [
        {"session_id": "s1", "role": "user", "content": "a"},
        {"session_id": "s2", "role": "user", "content": "b"},
    ]
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "doc.pdf"},
        {"session_id": "s2", "file_name": "autre.pdf"},
    ]
    nouveau = reinitialiser_session(supabase, "s1")
    assert nouveau != "s1"
    assert uuid.UUID(nouveau).version == 4
    # Session vidée, l'autre session intacte.
    assert [row["session_id"] for row in supabase.tables["chat_history"]] == ["s2"]
    assert [row["session_id"] for row in supabase.tables["document_chunks"]] == ["s2"]
    # Les deux purges ont bien été appelées (historique + documents).
    tables_purgees = [e[1] for e in supabase.journal if e[0] == "delete"]
    assert tables_purgees == ["document_chunks", "chat_history"]

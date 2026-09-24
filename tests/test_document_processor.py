"""Tests de modules/document_processor.py (découpage, lecture, ingestion)."""

import pytest

from modules.config import CHUNK_OVERLAP, CHUNK_SIZE
from modules.document_processor import (
    extract_text_from_file,
    process_and_store_document,
    split_text_into_chunks,
)
from tests.conftest import FakeIA, FakeSupabase, fichier

# ---------------------------------------------------------------------------
# Découpage (aucun filtre ici : le tri des morceaux vides est fait par
# process_and_store_document)
# ---------------------------------------------------------------------------


def test_decoupage_texte_vide():
    assert split_text_into_chunks("") == []


def test_decoupage_texte_court_non_filtre():
    assert split_text_into_chunks("abc") == ["abc"]


def test_decoupage_4_morceaux_pour_2500_caracteres():
    morceaux = split_text_into_chunks("x" * 2500)
    assert [len(m) for m in morceaux] == [1000, 1000, 900, 100]


def test_decoupage_chevauchement_200():
    texte = "abcdefghij" * 300  # 3000 caractères
    m1, m2, *_ = split_text_into_chunks(texte)
    assert len(m1) == CHUNK_SIZE
    assert m2[:CHUNK_OVERLAP] == m1[CHUNK_SIZE - CHUNK_OVERLAP :]


# ---------------------------------------------------------------------------
# Lecture de fichiers
# ---------------------------------------------------------------------------


def test_lecture_fichier_absent():
    assert extract_text_from_file(None) == ""


def test_lecture_txt_utf8():
    assert extract_text_from_file(fichier("a.txt", "Bonjour ÀÉÎ")) == "Bonjour ÀÉÎ"


def test_lecture_txt_invalid_utf8_ne_crash_pas():
    contenu = extract_text_from_file(fichier("a.txt", b"\xff\xfe\x00bin"))
    assert isinstance(contenu, str)


def test_lecture_fichier_vide_leve_une_erreur_lisible():
    with pytest.raises(ValueError, match="vide"):
        extract_text_from_file(fichier("vide.pdf", b""))


def test_lecture_pdf_corrompu_leve_une_erreur_lisible():
    with pytest.raises(ValueError, match="illisible"):
        extract_text_from_file(fichier("cassé.pdf", b"%PDF-1.4\ngarbage"))


def test_lecture_extension_inconnue_retourne_vide():
    assert extract_text_from_file(fichier("doc.md", "# coucou")) == ""


def test_lecture_flux_deja_consomme_est_repositionne():
    flux = fichier("deja-lu.txt", "texte utile")
    flux.read()  # un précédent rerun a consommé le flux
    assert extract_text_from_file(flux) == "texte utile"


def test_lecture_objet_sans_interface_fichier():
    with pytest.raises(ValueError, match="vide"):
        extract_text_from_file(object())


# ---------------------------------------------------------------------------
# Ingestion (purge -> lots d'embedding -> insert groupé)
# ---------------------------------------------------------------------------


def test_ingestion_46_morceaux_3_lots_1_insert():
    # Journal partagé : prouve l'ordre global embeds -> purge -> insert.
    timeline, progres = [], []
    supabase = FakeSupabase(journal=timeline)
    ia = FakeIA(timeline)
    n = process_and_store_document(
        text="x" * 36400,
        file_name="doc.pdf",
        session_id="s1",
        supabase_client=supabase,
        ai_client=ia,
        progress_callback=lambda a, b: progres.append((a, b)),
    )
    assert n == 46
    assert [e[2] for e in timeline if e[0] == "embed"] == [20, 20, 6]
    assert [e[0] for e in timeline] == ["embed", "embed", "embed", "delete", "insert"]
    assert len(timeline[4][3]) == 46  # un seul INSERT groupé
    assert progres == [(20, 46), (40, 46), (46, 46)]
    assert supabase.tables["document_chunks"][0]["file_name"] == "doc.pdf"


def test_ingestion_120_morceaux_insert_par_paliers_de_50(supabase, ia):
    supabase.journal = []
    n = process_and_store_document(
        text="x" * 95600,
        file_name="gros.pdf",
        session_id="s1",
        supabase_client=supabase,
        ai_client=ia,
    )
    assert n == 120
    assert [len(e[3]) for e in supabase.journal if e[0] == "insert"] == [50, 50, 20]
    assert [e[2] for e in ia.journal] == [20] * 6  # 6 lots d'embedding


def test_ingestion_purge_ancienne_version_avant_insert(supabase, ia):
    supabase.tables["document_chunks"] = [
        {
            "session_id": "s1",
            "file_name": "ancien.pdf",
            "content": "ANCIEN",
            "embedding": [0.0],
        },
        {
            "session_id": "autre-session",
            "file_name": "x.pdf",
            "content": "AUTRE",
            "embedding": [0.0],
        },
    ]
    supabase.journal = []
    process_and_store_document(
        text="nouveau contenu suffisamment long pour generer un morceau",
        file_name="ancien.pdf",
        session_id="s1",
        supabase_client=supabase,
        ai_client=ia,
    )
    suppressions = [e for e in supabase.journal if e[0] == "delete"]
    assert suppressions[0][2] == {"session_id": "s1", "file_name": "ancien.pdf"}
    contenus = [
        row["content"]
        for row in supabase.tables["document_chunks"]
        if row["session_id"] == "s1"
    ]
    assert "ANCIEN" not in contenus  # remplacé, pas doublonné
    assert any(
        row["session_id"] == "autre-session"
        for row in supabase.tables["document_chunks"]
    )  # intact


def test_ingestion_texte_inexploitable_ne_touche_rien(supabase, ia):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "a.pdf"},
    ]
    supabase.journal = []
    n = process_and_store_document(
        text="  \n ",
        file_name="vide.pdf",
        session_id="s1",
        supabase_client=supabase,
        ai_client=ia,
    )
    assert n == 0
    assert supabase.journal == []  # ni purge, ni insert


def test_ingestion_echec_suppression_annule_ecriture(supabase, ia):
    supabase.tables["document_chunks"] = [
        {
            "session_id": "s1",
            "file_name": "doc.pdf",
            "content": "ANCIEN",
            "embedding": [0.0],
        },
    ]
    supabase.echecs_delete.add("document_chunks")
    supabase.journal = []
    with pytest.raises(RuntimeError, match="ingestion annul"):
        process_and_store_document(
            text="x" * 50,
            file_name="doc.pdf",
            session_id="s1",
            supabase_client=supabase,
            ai_client=ia,
        )
    assert not any(e[0] == "insert" for e in supabase.journal)
    assert any(
        row["content"] == "ANCIEN" for row in supabase.tables["document_chunks"]
    )  # intact


def test_ingestion_echec_api_embedding_ne_touche_rien(supabase, ia):
    supabase.tables["document_chunks"] = [
        {"session_id": "s1", "file_name": "doc.pdf", "content": "ANCIEN"},
    ]
    ia.models.erreur = RuntimeError("API indisponible")
    supabase.journal = []
    with pytest.raises(RuntimeError, match="API indisponible"):
        process_and_store_document(
            text="x" * 50,
            file_name="doc.pdf",
            session_id="s1",
            supabase_client=supabase,
            ai_client=ia,
        )
    assert supabase.journal == []  # rien avant le succès
    assert any(row["content"] == "ANCIEN" for row in supabase.tables["document_chunks"])


def test_pdf_valide_pages_extraites():
    """Flux PDF valide (page vierge) : parcourt l'extraction sans erreur."""
    import io

    import pypdf

    tampon = io.BytesIO()
    ecrivain = pypdf.PdfWriter()
    ecrivain.add_blank_page(width=200, height=200)
    ecrivain.write(tampon)
    tampon.name = "page.pdf"
    assert extract_text_from_file(tampon) == ""  # page sans couche texte


def test_pdf_protege_rejete():
    """PDF chiffré : message clair plutôt que plantage pypdf."""
    import io

    import pypdf

    tampon = io.BytesIO()
    ecrivain = pypdf.PdfWriter()
    ecrivain.add_blank_page(width=200, height=200)
    ecrivain.encrypt("secret")
    ecrivain.write(tampon)
    tampon.name = "protege.pdf"
    with pytest.raises(ValueError, match="mot de passe"):
        extract_text_from_file(tampon)

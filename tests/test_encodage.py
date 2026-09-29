"""Garde-fou d'encodage : aucun caractère accentué perdu sous forme d'interrogation.

Les sources de ce projet sont en UTF-8 et rédigées en français. Des écritures
faites via un shell Windows (PowerShell) ont déjà remplacé les caractères
accentués par le caractère d'interrogation : la perte est irréversible à la
lecture, puisqu'il devient impossible de le distinguer d'un vrai point
d'interrogation.

Ce test détecte la signature de cette corruption (une interrogation collée à
une lettre, hors URL) et échoue pour empêcher toute régression.
"""

import pathlib
import re

RACINE = pathlib.Path(__file__).resolve().parents[1]

# Interrogation prise en sandwich dans un mot (ex. « G[n]n[re] », « ignor[s] »).
CORRUPTION = re.compile(r"[A-Za-zÀ-ÿß-Ÿ]\?|\?[a-zà-ÿ]")

# Ce fichier décrit la signature recherchée : il la contient donc littéralement.
LUI_MEME = pathlib.Path(__file__).name


def _fichiers_sources():
    """Tous les .py du projet, hors environnement virtuel et dossiers outils."""
    for chemin in sorted(RACINE.rglob("*.py")):
        dossiers = {part.lower() for part in chemin.parts}
        if {"venv", ".git", "__pycache__", "node_modules"} & dossiers:
            continue
        if chemin.name == LUI_MEME:
            continue
        yield chemin


def test_tous_les_fichiers_sont_decodables_en_utf8():
    for chemin in _fichiers_sources():
        chemin.read_text(encoding="utf-8")  # lève UnicodeDecodeError si corrompu


def test_aucun_accent_perdu_sous_forme_de_point_interrogation():
    fautifs = []
    for chemin in _fichiers_sources():
        for numero, ligne in enumerate(
            chemin.read_text(encoding="utf-8").splitlines(), 1
        ):
            if "http" in ligne:  # query strings d'URL : « ?name= », « ?latitude= »
                continue
            if CORRUPTION.search(ligne):
                fautifs.append(
                    f"{chemin.relative_to(RACINE)}:{numero}: {ligne.strip()}"
                )
    assert not fautifs, (
        "Accents perdus sous forme de '?' détectés (écriture via un shell non UTF-8 ? "
        "réécris ces lignes avec un éditeur UTF-8) :\n" + "\n".join(fautifs)
    )

"""Tests des constantes centralisées et de la politique de logging."""

import logging

import modules.config as config


def test_constantes_de_cadrage():
    assert config.CHUNK_SIZE == 1000
    assert config.CHUNK_OVERLAP == 200
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert config.CHUNK_MIN_LENGTH == 10


def test_constantes_de_lot_et_seuils():
    assert config.EMBEDDING_BATCH_SIZE == 20
    assert config.INSERT_BATCH_SIZE == 50
    assert config.MATCH_THRESHOLD == 0.3
    assert config.MATCH_COUNT == 4
    assert config.MAX_ITERATIONS == 10


def test_modeles_declarés():
    assert config.CHAT_MODEL
    assert config.EMBEDDING_MODEL
    # La dimension doit rester cohérente avec le vector(3072) de Supabase.
    assert config.EMBEDDING_DIMENSIONS == 3072


def test_constantes_de_reflexion():
    # Une question courte ne doit jamais payer un appel de planification.
    assert config.AGENT_PLAN_ACTIVEE is True
    assert config.AGENT_CRITIQUE_ACTIVEE is True
    assert config.PLAN_SEUIL_CARACTERES > 20
    assert config.PLAN_MAX_CHARS == 500
    # La raison rendue à l'agent reste plus courte que la réponse jugée.
    assert config.CRITIQUE_MAX_CHARS < config.CRITIQUE_BROUILLON_MAX_CHARS
    # Une seule répétition tolérée : la seconde est tenue pour un blocage.
    assert config.STAGNATION_MAX_APPELS_IDENTIQUES == 1


def test_seuil_de_regroupement_de_la_memoire():
    # Sans regroupement, la conversation paierait un résumé à chaque tour dès
    # que la fenêtre est dépassée. Le seuil reste en deçà de la fenêtre.
    assert config.MEMORY_MIN_OVERFLOW == 4
    assert config.MEMORY_MIN_OVERFLOW <= config.MEMORY_WINDOW_SIZE


def test_configurer_logging_appelle_basicconfig(monkeypatch):
    appels = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: appels.update(kwargs))
    config.configurer_logging()
    assert appels["level"] == logging.INFO
    assert "%(levelname)s" in appels["format"]
    assert "%(name)s" in appels["format"]
    # Format ASCII uniquement : pas d'émojis dans les journaux (mojibake).
    assert all(ord(c) < 128 for c in appels["format"])

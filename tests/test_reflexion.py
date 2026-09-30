"""Tests de modules/reflexion.py (plan initial, auto-critique, observations).

Ces briques n'appellent jamais le modèle elles-mêmes : la fonction d'appel est
injectée, ce qui permet d'exercer chaque branche (plan retenu, question jugée
simple, verdict illisible, panne réseau) sans réseau ni client Gemini.
"""

import pytest

from modules.config import (
    CRITIQUE_BROUILLON_MAX_CHARS,
    CRITIQUE_MAX_CHARS,
    OBSERVATIONS_MAX_CHARS,
    PLAN_MAX_CHARS,
    PLAN_SEUIL_CARACTERES,
)
from modules.reflexion import (
    PLAN_SIMPLE,
    PROMPT_CRITIQUE,
    PROMPT_PLAN,
    analyser_verdict,
    construire_plan,
    evaluer_brouillon,
    formater_observations,
    merite_un_plan,
)


def _appel_factice(*reponses):
    """Fonction d'appel factice : rejoue les réponses fournies, une par envoi.

    L'attribut `invites` journalise les textes reçus, pour vérifier ce qui a
    été envoyé au modèle (et surtout s'il a été appelé du tout).
    """
    file = list(reponses)
    invites = []

    def appeler(invite):
        invites.append(invite)
        return file.pop(0)

    appeler.invites = invites
    return appeler


def _appel_explosif(invite):
    """Appel de modèle en panne : la réflexion doit rester fail-open."""
    raise RuntimeError("API indisponible")


# --- Observations collectées -------------------------------------------------


def test_formater_observations_vides_renvoyer_une_chaine_vide():
    assert formater_observations([]) == ""


def test_formater_observations_conserve_l_ordre_sans_troncation():
    assert formater_observations(["a -> 1", "b -> 2"]) == "a -> 1\nb -> 2"


def test_formater_observations_tronque_les_plus_anciennes():
    # Rappel borné à OBSERVATIONS_MAX_CHARS : les observations récentes sont
    # gardées, et la troncature est signalée au modèle.
    observations = ["ancienne" * 200, "moyenne" * 200, "recente" * 200]
    rappel = formater_observations(observations)
    assert "recente" * 200 in rappel
    assert "tronquées" in rappel
    assert "ancienne" * 200 not in rappel
    assert "moyenne" * 200 not in rappel
    assert len(rappel) <= OBSERVATIONS_MAX_CHARS + 120  # marge = marqueur


# --- Choix de planifier ou non -----------------------------------------------


def test_question_courte_ne_merite_aucun_plan():
    # Le plan coûte un appel : une question simple doit rester sans plan.
    assert merite_un_plan("Quel temps fait-il à Paris") is False


def test_question_a_deux_voix_merite_un_plan():
    assert merite_un_plan("Quel temps fait-il à Paris ? Et à Madrid ?") is True


def test_question_longue_merite_un_plan():
    assert merite_un_plan("Analyse détaillée : " + "a" * PLAN_SEUIL_CARACTERES) is True


def test_question_vide_ou_absente_ne_merite_aucun_plan():
    assert merite_un_plan("") is False
    assert merite_un_plan("   ") is False
    assert merite_un_plan(None) is False


# --- Plan initial ------------------------------------------------------------


def test_question_vide_ne_declenche_aucun_appel_de_plan():
    appeler = _appel_factice("1. quoi que ce soit")
    assert construire_plan("   ", appeler) is None
    assert appeler.invites == []  # aucun appel payé pour rien


def test_le_plan_est_renvoyet_et_la_question_est_transmise():
    appeler = _appel_factice("1. Collecter\n2. Conclure")
    plan = construire_plan("Compare les ventes de 2024 et 2025", appeler)
    assert plan == "1. Collecter\n2. Conclure"
    assert PROMPT_PLAN in appeler.invites[0]
    assert "Question : Compare les ventes de 2024 et 2025" in appeler.invites[0]


def test_question_jugee_simple_ne_produit_aucun_plan():
    # Le modèle rend le marqueur attendu : la question se traite d'une seule
    # traite, le plan n'a rien à apporter et ne sera pas injecté.
    assert construire_plan("Bonjour", _appel_factice(PLAN_SIMPLE)) is None
    assert construire_plan("Bonjour", _appel_factice(" plan simple ")) is None


def test_reponse_vide_du_modele_ne_produit_aucun_plan():
    assert construire_plan("Une question", _appel_factice("   ")) is None


def test_le_plan_est_tronque_a_la_longueur_maximale():
    plan = construire_plan("Compare tout", _appel_factice("x" * (PLAN_MAX_CHARS + 200)))
    assert len(plan) == PLAN_MAX_CHARS
    assert plan.endswith("…")


def test_panne_d_appel_supprime_le_plan_sans_empecher_de_repondre():
    assert construire_plan("Une question", _appel_explosif) is None


# --- Lecture du verdict ------------------------------------------------------


@pytest.mark.parametrize(
    "verdict", ["OK", "ok", " Ok. ", "OK, rien à redire", "", None]
)
def test_verdict_de_validation(verdict):
    # Tout ce qui n'est pas un refus explicite vaut validation : une critique
    # muette ou mal formatée ne doit jamais retenir une réponse.
    assert analyser_verdict(verdict) == (True, "")


def test_verdict_insuffisant_extraie_la_raison():
    valide, raison = analyser_verdict("INSUFFISANT : la météo de Madrid manque")
    assert valide is False
    assert raison == "la météo de Madrid manque"


def test_verdict_markdown_ou_en_minuscules_est_compris():
    # Le modèle peut enrichir sa réponse de mise en forme : seule la tête du
    # verdict compte.
    valide, raison = analyser_verdict("**insuffisant : calcul faux**")
    assert valide is False
    assert "calcul faux" in raison
    assert analyser_verdict("- NON : rien sur 2025")[0] is False


def test_refus_sans_raison_produit_un_texte_generique():
    valide, raison = analyser_verdict("NON")
    assert valide is False
    assert "sans précision" in raison


def test_refus_vide_apres_les_deux_points_produit_un_texte_generique():
    assert "sans précision" in analyser_verdict("INSUFFISANT : ")[1]


def test_la_raison_est_bornee_en_longueur():
    _, raison = analyser_verdict("INSUFFISANT : " + "a" * (CRITIQUE_MAX_CHARS + 100))
    assert len(raison) == CRITIQUE_MAX_CHARS
    assert raison.endswith("…")


# --- Évaluation du brouillon -------------------------------------------------


def test_le_prompt_de_critique_assemble_question_faits_et_reponse():
    appeler = _appel_factice("OK")
    evaluer_brouillon("Quel temps à Paris", ["meteo -> 22°C"], "Il fait 22.", appeler)
    invite = appeler.invites[0]
    assert PROMPT_CRITIQUE in invite
    assert "Question : Quel temps à Paris" in invite
    assert "Observations collectées auprès des outils :" in invite
    assert "meteo -> 22°C" in invite
    assert "Réponse à évaluer :" in invite
    assert "Il fait 22." in invite


def test_le_prompt_de_critique_se_passe_des_observations_vides():
    appeler = _appel_factice("OK")
    evaluer_brouillon("Question", [], "Réponse", appeler)
    assert "Observations collectées" not in appeler.invites[0]


def test_le_brouillon_trop_long_est_tronque_dans_la_demande():
    # La réponse à juger est potentiellement longue : elle est bornée, le prompt
    # de critique reste de taille maîtrisée.
    appeler = _appel_factice("OK")
    evaluer_brouillon("Q", [], "y" * (CRITIQUE_BROUILLON_MAX_CHARS + 500), appeler)
    invite = appeler.invites[0]
    assert invite.endswith("…")  # la réponse a été coupée, pas envoyée en entier
    assert len(invite) < CRITIQUE_BROUILLON_MAX_CHARS + len(PROMPT_CRITIQUE) + 100


def test_brouillon_valide_par_la_critique():
    assert evaluer_brouillon("Q", ["outil -> 4"], "4", _appel_factice("OK")) == (
        True,
        "",
    )


def test_brouillon_rejete_renvoie_la_raison_a_corriger():
    appeler = _appel_factice("INSUFFISANT : il manque le total")
    assert evaluer_brouillon("Q", ["outil -> 4"], "je ne sais pas", appeler) == (
        False,
        "il manque le total",
    )


def test_panne_de_la_critique_vaut_validation_de_la_reponse():
    assert evaluer_brouillon("Q", ["outil -> 4"], "réponse", _appel_explosif) == (
        True,
        "",
    )

"""Tests de modules/reflexion.py (planification, auto-critique, observations).

La planification se lit dans le texte rendu par le planificateur, la critique
reçoit sa fonction d'appel injectée : chaque branche (plan, réponse directe,
marqueur indécis, verdict illisible, panne réseau) s'exerce sans réseau ni
client Gemini.
"""

import pytest

from modules.config import (
    CRITIQUE_BROUILLON_MAX_CHARS,
    CRITIQUE_MAX_CHARS,
    OBSERVATIONS_MAX_CHARS,
    PLAN_MARQUEUR_MAX_CHARS,
    PLAN_MAX_CHARS,
)
from modules.reflexion import (
    MARQUEUR_PLAN,
    MARQUEUR_REPONSE,
    PROMPT_CRITIQUE,
    PROMPT_PLAN,
    Planification,
    analyser_planification,
    analyser_verdict,
    evaluer_brouillon,
    formater_observations,
    marqueur_en_tete,
    retirer_marqueur,
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


# --- Lecture du marqueur du planificateur ------------------------------------


def test_marqueur_indecis_tant_que_la_premiere_ligne_n_est_pas_close():
    # Les morceaux arrivent un par un : tant que rien ne tranche, le flux reste
    # retenu, sinon un plan serait publié comme s'il s'agissait d'une réponse.
    assert marqueur_en_tete("") is None
    assert marqueur_en_tete("   ") is None
    assert marqueur_en_tete("REP") is None  # mot peut-être encore en cours
    # Première ligne close : la lecture peut conclure.
    assert marqueur_en_tete("REPONSE\nBonjour") == MARQUEUR_REPONSE
    assert marqueur_en_tete("PLAN\n1. Chercher") == MARQUEUR_PLAN


def test_marqueur_tranche_quand_la_premiere_ligne_devient_trop_longue():
    # Sans retour à la ligne, on n'attend pas indéfiniment : au-delà de la
    # fenêtre de lecture, l'absence de REPONSE vaut plan.
    assert marqueur_en_tete("x" * (PLAN_MARQUEUR_MAX_CHARS + 1)) == MARQUEUR_PLAN
    assert marqueur_en_tete("1. Chercher et recouper", definitif=True) == MARQUEUR_PLAN


def test_marqueur_tolere_les_decors_markdown():
    # Le modèle décore parfois sa réponse : seuls la tête et le mot comptent.
    assert marqueur_en_tete("**REPONSE** : Bonjour\nla suite") == MARQUEUR_REPONSE
    assert marqueur_en_tete("# reponse directe\nBonjour") == MARQUEUR_REPONSE


def test_marqueur_definitif_tranche_sur_le_texte_recu():
    # Fin de flux : plus rien à attendre, il faut conclure sur ce qui est reçu.
    assert marqueur_en_tete("REP", definitif=True) == MARQUEUR_PLAN


# --- Retrait du marqueur ------------------------------------------------------


def test_le_marqueur_est_retire_de_la_reponse():
    assert retirer_marqueur("REPONSE\nBonjour") == (MARQUEUR_REPONSE, "Bonjour")
    assert retirer_marqueur("REPONSE : Bonjour") == (MARQUEUR_REPONSE, "Bonjour")
    assert retirer_marqueur("PLAN\n1. Chercher") == (MARQUEUR_PLAN, "1. Chercher")


def test_texte_sans_marqueur_fait_corps():
    # Un plan écrit sans marqueur reste un plan : tout le texte fait corps.
    assert retirer_marqueur("1. Chercher\n2. Conclure") == (
        MARQUEUR_PLAN,
        "1. Chercher\n2. Conclure",
    )
    assert retirer_marqueur("") == (None, "")


# --- Verdict du planificateur -------------------------------------------------


def test_planification_vide_ne_donne_ni_plan_ni_reponse():
    # Rien à exploiter : l'agent décide seul plutôt que de recevoir un plan vide.
    assert analyser_planification("") == Planification()
    assert analyser_planification("   ") == Planification()
    assert analyser_planification("PLAN") == Planification()
    assert analyser_planification("REPONSE") == Planification()


def test_plan_rendu_par_le_planificateur():
    assert analyser_planification("PLAN\n1. Chercher\n2. Conclure") == Planification(
        plan="1. Chercher\n2. Conclure"
    )
    assert analyser_planification("1. Chercher\n2. Conclure").plan == (
        "1. Chercher\n2. Conclure"
    )


def test_reponse_directe_rendue_telle_quelle():
    # Le planificateur a la réponse : elle part telle quelle, sans son marqueur.
    assert analyser_planification("REPONSE\nBonjour !") == Planification(
        reponse="Bonjour !"
    )
    assert analyser_planification("reponse : 2 + 2 = 4").reponse == "2 + 2 = 4"


def test_plan_borne_a_la_longueur_maximale():
    plan = analyser_planification("PLAN\n" + "x" * (PLAN_MAX_CHARS + 200)).plan
    assert len(plan) == PLAN_MAX_CHARS
    assert plan.endswith("…")


def test_le_prompt_de_planification_annonce_les_deux_marqueurs():
    # Le protocole repose entièrement sur ce prompt : les deux marqueurs et la
    # consigne de repli doivent y figurer.
    assert PROMPT_PLAN.startswith("Tu es le planificateur")
    assert MARQUEUR_REPONSE in PROMPT_PLAN
    assert MARQUEUR_PLAN in PROMPT_PLAN


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

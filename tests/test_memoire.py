"""Tests de services/memoire.py (fenêtre glissante et résumé incrémental)."""

from modules.ai_engine import CONFIG_SANS_OUTILS
from modules.config import MEMORY_SUMMARY_MAX_CHARS, MEMORY_WINDOW_SIZE
from services.memoire import EN_TETE_RESUME, compresser_historique
from tests.conftest import reponse_finale


def conversation(nb_messages, roles=("user", "assistant")):
    """Historique alterné de nb_messages messages, du plus ancien au plus récent."""
    return [
        {"role": roles[i % len(roles)], "content": f"message {i}"}
        for i in range(nb_messages)
    ]


def test_historique_sous_la_fenetre_reste_inchange_sans_appel(ia):
    historique = conversation(MEMORY_WINDOW_SIZE)
    vue, resume, index = compresser_historique(historique, ia)
    assert vue == historique
    assert resume is None
    assert index == 0
    # Aucun message envoyé : la compression ne coûte rien tant que rien ne déborde.
    assert ia.chats.creation is None


def test_les_messages_hors_fenetre_sont_remplaces_par_le_resume(ia):
    historique = conversation(8)
    ia.chats.reponses = [reponse_finale("LE RESUME")]
    vue, resume, index = compresser_historique(historique, ia)

    assert resume == "LE RESUME"
    assert index == 3  # les 3 plus anciens sont désormais couverts par le résumé
    assert vue[0]["role"] == "user"
    assert vue[0]["content"].startswith(EN_TETE_RESUME)
    assert vue[0]["content"].endswith("LE RESUME")
    # Les messages récents sont conservés mot pour mot, dans l'ordre.
    assert vue[1:] == historique[3:]
    # Une fenêtre de MEMORY_WINDOW_SIZE messages récents plus le résumé.
    assert len(vue) == MEMORY_WINDOW_SIZE


def test_le_resume_est_demande_sans_outil_et_hors_historique(ia):
    ia.chats.reponses = [reponse_finale("LE RESUME")]
    compresser_historique(conversation(8), ia)
    # Chat de synthèse vierge de tout historique, outils explicitement coupés.
    assert ia.chats.creation["history"] == []
    assert ia.chats.creation["config"] is CONFIG_SANS_OUTILS


def test_seuls_les_messages_debordants_sont_envoyes_a_la_synthese(ia):
    historique = conversation(8)
    ia.chats.reponses = [reponse_finale("LE RESUME")]
    compresser_historique(historique, ia)
    invite = ia.chats.dernier.envoyes[0]
    assert "Utilisateur : message 0" in invite
    assert "Assistant : message 1" in invite
    assert "Utilisateur : message 2" in invite
    # Ce qui reste dans la fenêtre n'est pas résumé deux fois.
    assert "message 3" not in invite


def test_le_resume_deja_etabli_est_reinjecte_dans_la_demande(ia):
    # 14 messages : 4 deja resumes, 8 debordent depuis la fenetre de 6. Le
    # regroupement (MEMORY_MIN_OVERFLOW = 4) est ainsi atteint et l'appel a lieu.
    historique = conversation(14)
    ia.chats.reponses = [reponse_finale("RESUME A JOUR")]
    vue, resume, index = compresser_historique(
        historique, ia, resume="ANCIEN RESUME", deja_resume=4
    )

    invite = ia.chats.dernier.envoyes[0]
    assert "Résumé déjà établi :\nANCIEN RESUME" in invite
    # Synthèse incrémentale : uniquement les messages qui viennent de déborder.
    assert "Utilisateur : message 4" in invite
    assert "message 0" not in invite
    assert resume == "RESUME A JOUR"
    assert index == 9
    assert vue[1:] == historique[9:]


def test_aucun_appel_quand_rien_ne_deborde_depuis_le_dernier_resume(ia):
    # Le résumé couvre déjà tout ce qui sort de la fenêtre : rien à renvoyer à
    # la synthèse, et le résumé mémorisé est réutilisé tel quel.
    historique = conversation(10)
    vue, resume, index = compresser_historique(
        historique, ia, resume="ACQUIS", deja_resume=4
    )
    assert ia.chats.creation is None
    assert resume == "ACQUIS"
    assert index == 4
    assert vue[0]["content"].startswith(EN_TETE_RESUME)
    assert vue[1:] == historique[4:]


def test_la_fenetre_demarre_sur_une_reponse_du_modele(ia):
    # 10 messages : la découpe brute tomberait sur un message de l'utilisateur,
    # ce qui suivrait immédiatement le résumé (deux tours d'affilée du même
    # rôle, refusés par Gemini). Le découpage doit donc avancer d'un cran.
    historique = conversation(10)
    ia.chats.reponses = [reponse_finale("LE RESUME")]
    vue, _, index = compresser_historique(historique, ia)
    assert index == 5
    assert vue[1]["role"] == "assistant"
    assert vue[1:] == historique[5:]
    assert len(vue) == MEMORY_WINDOW_SIZE


def test_quand_le_resume_echoue_la_fenetre_recente_seule_est_envoyee(ia):
    historique = conversation(8)
    ia.chats.reponses = []  # le client simule tombe en erreur à l'appel
    vue, resume, index = compresser_historique(historique, ia)

    assert resume is None
    # L'index n'avance pas : la synthèse sera retentée au tour suivant.
    assert index == 0
    # Sans résumé injecté, l'historique doit commencer par l'utilisateur.
    assert vue == historique[4:]
    assert vue[0]["role"] == "user"


def test_reponse_vide_du_modele_est_traitee_comme_un_echec(ia):
    historique = conversation(8)
    ia.chats.reponses = [reponse_finale("   ")]
    vue, resume, index = compresser_historique(historique, ia)
    assert vue == historique[4:]
    assert resume is None
    assert index == 0


def test_quand_le_resume_echoue_le_resume_deja_etabli_est_quand_meme_injecte(ia):
    # 14 messages : le seuil de regroupement est atteint, la synthèse est tentée
    # (et échoue), l'index n'avance pas et le résumé acquis reste injecté.
    historique = conversation(14)
    ia.chats.reponses = []
    vue, resume, index = compresser_historique(
        historique, ia, resume="ANCIEN RESUME", deja_resume=4
    )
    assert resume == "ANCIEN RESUME"
    assert index == 4
    assert vue[0]["content"] == f"{EN_TETE_RESUME}\nANCIEN RESUME"
    assert vue[1:] == historique[9:]


def test_le_resume_est_tronque_a_la_longueur_maximale(ia):
    ia.chats.reponses = [reponse_finale("x" * (MEMORY_SUMMARY_MAX_CHARS + 500))]
    _, resume, _ = compresser_historique(conversation(8), ia)
    assert len(resume) == MEMORY_SUMMARY_MAX_CHARS
    assert resume.endswith("…")


def test_un_index_perime_repart_d_un_historique_entier(ia):
    # L'index mémorisé ne correspond plus rien (lignes invalides filtrées).
    historique = conversation(3)
    vue, resume, index = compresser_historique(
        historique, ia, resume="ANCIEN RESUME", deja_resume=99
    )
    assert vue == historique
    assert resume is None
    assert index == 0
    assert ia.chats.creation is None


def test_sans_reponse_du_modele_a_conserver_on_ne_compresse_pas(ia):
    # Historique anormal (aucune réponse du modèle) : renvoyer tout plutôt
    # qu'une requête réduite à son seul résumé.
    historique = conversation(10, roles=("user",))
    vue, resume, index = compresser_historique(historique, ia)
    assert vue == historique
    assert resume is None
    assert index == 0
    assert ia.chats.creation is None


def test_l_historique_d_origine_n_est_jamais_modifie(ia):
    historique = conversation(8)
    avant = [dict(message) for message in historique]
    ia.chats.reponses = [reponse_finale("LE RESUME")]
    compresser_historique(historique, ia)
    # Vue compressée : l'historique affiché et persisté reste intact.
    assert historique == avant


# --- Regroupement des synthèses (MEMORY_MIN_OVERFLOW) -----------------------


def test_les_syntheses_sont_regroupees_tant_que_le_seuil_n_est_atteint(ia):
    # Fenêtre de 6, résumé couvrant les 4 premiers, 12 messages au total : seuls
    # 2 messages débordent depuis le dernier résumé. Le seuil de 4 n'est pas
    # atteint : aucun appel, la vue garde simplement une fenêtre un peu plus
    # longue, qui sera compressée au tour suivant.
    historique = conversation(12)
    vue, resume, index = compresser_historique(
        historique, ia, resume="ACQUIS", deja_resume=4
    )
    assert ia.chats.creation is None
    assert resume == "ACQUIS"
    assert index == 4
    assert vue[0]["content"] == f"{EN_TETE_RESUME}\nACQUIS"
    assert vue[1:] == historique[4:]


def test_le_premier_debordement_est_resume_sans_attendre_le_seuil(ia):
    # Le regroupement ne vaut que pour les synthèses suivantes : dès qu'un
    # message sort de la fenêtre alors qu'aucun résumé n'existe encore, il est
    # synthétisé (ici 2 messages, sous le seuil de 4).
    historique = conversation(8)
    ia.chats.reponses = [reponse_finale("LE RESUME")]
    _, resume, index = compresser_historique(historique, ia)
    assert resume == "LE RESUME"
    assert index == 3

"""Tests de modules/ai_engine.py (format, embeddings, prompt, boucle ReAct)."""

import pytest

import modules.ai_engine as engine
from modules.ai_engine import (
    format_history_for_gemini,
    generate_rag_prompt,
    get_ai_response,
    get_embedding,
    get_embeddings,
    init_ai_client,
)
from modules.config import CHAT_MODEL, MAX_ITERATIONS
from tests.conftest import (
    reponse_finale,
    reponse_outil,
    reponse_texte_et_outil,
)


def _histoire(question="ma question"):
    """Historique Gemini : conversation + dernier message utilisateur courant."""
    return format_history_for_gemini(
        [
            {"role": "user", "content": "contexte"},
            {"role": "assistant", "content": "reponse"},
        ]
    ) + [{"role": "user", "parts": [{"text": question}]}]


def _question_longue():
    """Question assez fournie pour que la planification se déclenche."""
    return "Analyse détaillée " + "et comparaison " * 10


def _outil_compteur(monkeypatch, nom, resultat, journal):
    """Remplace un outil réseau par une fonction factice qui compte ses appels."""

    def outil(**arguments):
        journal.append(arguments)
        return resultat

    monkeypatch.setitem(engine.AVAILABLE_TOOLS, nom, outil)


# --- Historique ------------------------------------------------------------


def test_format_history_mappe_roles_au_format_gemini():
    h = format_history_for_gemini(
        [
            {"role": "user", "content": "salut"},
            {"role": "assistant", "content": "bonjour"},
            {"role": "systeme", "content": "x"},  # rôle inconnu -> "user"
        ]
    )
    assert [m["role"] for m in h] == ["user", "model", "user"]
    assert h[0]["parts"][0]["text"] == "salut"
    assert h[1]["parts"][0]["text"] == "bonjour"


# --- Embeddings ------------------------------------------------------------


def test_get_embeddings_par_lots_et_ordre_preserve(ia):
    textes = ["a" * i for i in range(1, 46)]  # 45 textes, longueurs distinctes
    vecteurs = get_embeddings(textes, ia)
    assert [e[2] for e in ia.journal] == [20, 20, 5]  # batch_size = 20
    assert [v[0] for v in vecteurs] == [float(i) for i in range(1, 46)]


def test_get_embeddings_alignement_faux_leve_une_erreur(ia):
    ia.models.nb_retourne = 19  # l'API "oublie" un vecteur
    with pytest.raises(RuntimeError, match="alignement"):
        get_embeddings(["a" * 5] * 20, ia)


def test_get_embedding_unitaire(ia):
    assert get_embedding("abcd", ia) == [4.0, 1.0, 2.0]


def test_init_ai_client_retourne_un_client_genai():
    client = init_ai_client("cle-de-test")
    assert hasattr(client, "models")
    assert hasattr(client, "chats")


# --- Prompt RAG ------------------------------------------------------------


def test_generate_rag_prompt_etiquette_et_consigne():
    prompt = generate_rag_prompt(
        [{"file_name": "CV.pdf", "content": "ligne A"}, {"content": "ligne B"}],
        "Question ?",
    )
    assert "[Extrait de CV.pdf]\nligne A" in prompt
    assert "[Extrait]\nligne B" in prompt
    assert "Question ?" in prompt
    assert "uniquement" in prompt  # consigne d'honnêteté


# --- Boucle ReAct ----------------------------------------------------------


def test_reponse_directe_sans_appel_outil(ia):
    ia.chats.reponses = [reponse_finale("Bonjour !")]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert texte == "Bonjour !"
    assert ia.chats.creation["model"] == CHAT_MODEL
    # history = conversation SANS le message courant (retiré par la boucle).
    assert len(ia.chats.creation["history"]) == 2
    assert any("Étape 1" in s for s in statuts)
    assert any("Rédaction" in s for s in statuts)


def test_appel_outil_calculatrice_renvoie_le_resultat(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "2 + 2"}),
        reponse_finale("Le résultat est 4."),
    ]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert texte == "Le résultat est 4."
    reponse_out = ia.chats.dernier.envoyes[1]  # message des retours d'outil
    assert reponse_out[0].function_response.name == "calculatrice"
    assert reponse_out[0].function_response.response == {"result": "4"}
    assert any("Calcul en cours" in s for s in statuts)


def test_appel_outil_arguments_invalides_renvoie_erreur(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {}),  # expression manquante
        reponse_finale("Desole."),
    ]
    assert get_ai_response(ia, _histoire()) == "Desole."
    reponse_out = ia.chats.dernier.envoyes[1]
    assert "error" in reponse_out[0].function_response.response


def test_outil_inconnu_renvoie_tool_not_found(ia):
    ia.chats.reponses = [
        reponse_outil("open_file", {}),  # halluciné par le modèle
        reponse_finale("Je ne peux pas."),
    ]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert texte == "Je ne peux pas."
    reponse_out = ia.chats.dernier.envoyes[1]
    assert reponse_out[0].function_response.response == {"error": "Tool not found"}
    assert any("Utilisation de l'outil" in s for s in statuts)  # branche generique


def test_statuts_des_outils_reseau_avec_outils_factice(monkeypatch, ia):
    # On remplace l'exécution réelle (réseau) tout en gardant les statuts UI.
    monkeypatch.setitem(
        engine.AVAILABLE_TOOLS, "recherche_web", lambda requete: "web-ok"
    )
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "meteo", lambda ville: "meteo-ok")
    ia.chats.reponses = [
        reponse_outil("recherche_web", {"requete": "actualités IA"}),
        reponse_outil("meteo", {"ville": "Paris"}),
        reponse_finale("Voilà."),
    ]
    statuts = []
    assert get_ai_response(ia, _histoire(), status_callback=statuts.append) == "Voilà."
    assert any("Recherche sur le web" in s for s in statuts)
    assert any("Consultation de la météo" in s for s in statuts)


def test_boucle_bornee_forcage_de_synthese_finale(ia):
    # Le modele demande un outil a chaque tour sans jamais conclure : au dernier
    # tour autorise, la boucle doit forcer une synthese au lieu de relancer un
    # outil (avant : abandon sec apres 10 appels). Les arguments different a
    # chaque tour pour ne pas tomber dans le cache d'appels (voir stagnation).
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": f"{i}+1"})
        for i in range(MAX_ITERATIONS)
    ] + [reponse_finale("Synthese partielle : 1+1 fait 2.")]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)
    assert "Synthese partielle : 1+1 fait 2." in texte
    assert "limite de recherche" in texte
    assert any("Limite de recherche atteinte" in s for s in statuts)
    # MAX_ITERATIONS envois + 1 envoi de synthese forcee : la boucle est bornee.
    assert len(ia.chats.dernier.envoyes) == MAX_ITERATIONS + 1


def test_synthese_forcee_desactive_les_outils(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": f"{i}+1"})
        for i in range(MAX_ITERATIONS)
    ] + [reponse_finale("Voila.")]
    get_ai_response(ia, _histoire())
    config = ia.chats.dernier.configs[-1]
    # L'appel de synthese interdit tout nouvel appel d'outil au modele.
    assert config is engine.CONFIG_SANS_OUTILS
    mode = config.tool_config.function_calling_config.mode
    assert getattr(mode, "value", mode) == "NONE"
    # Les tours precedents, eux, gardent les outils disponibles.
    assert ia.chats.dernier.configs[0] is None


def test_synthese_forcee_sans_texte_renvoie_repli(ia):
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": f"{i}+1"})
        for i in range(MAX_ITERATIONS)
    ] + [reponse_finale("")]  # synthese vide : on retombe sur le repli
    texte = get_ai_response(ia, _histoire())
    assert "Je n'ai pas pu finaliser" in texte


def test_synthese_forcee_en_echec_renvoie_repli(ia):
    # Aucune reponse en reserve pour l'appel de synthese : il leve, et l'agent
    # doit malgre tout repondre quelque chose au lieu de propager l'erreur.
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": f"{i}+1"})
        for i in range(MAX_ITERATIONS)
    ]
    texte = get_ai_response(ia, _histoire())
    assert "Je n'ai pas pu finaliser" in texte
    assert len(ia.chats.dernier.envoyes) == MAX_ITERATIONS + 1


def test_boucle_desactivee_renvoie_message_dedie(monkeypatch, ia):
    # Cas limite : aucune iteration autorisee. On doit renvoyer le message de
    # repli sans jamais envoyer de message au modele.
    monkeypatch.setattr(engine, "MAX_ITERATIONS", 0)
    texte = get_ai_response(ia, _histoire())
    assert "trop complexe" in texte
    assert ia.chats.dernier.envoyes == []


def test_reponse_finale_vide_relance_le_modele(ia):
    # Sortie vide (ni texte ni outil) : on relance le modele au lieu de renvoyer
    # None a l'interface.
    ia.chats.reponses = [reponse_finale(""), reponse_finale("Voici la reponse.")]
    texte = get_ai_response(ia, _histoire())
    assert texte == "Voici la reponse."
    assert len(ia.chats.dernier.envoyes) == 2


# --- Synthese forcee : rappel des observations ------------------------------


def test_synthese_forcee_rappelle_les_observations(ia):
    # Les resultats deja obtenus sont reinjectes dans le message de synthese :
    # l'agent peut conclure meme si l'historique de session est volumineux.
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": f"{i} + 2"})
        for i in range(MAX_ITERATIONS)
    ] + [reponse_finale("Synthese.")]
    get_ai_response(ia, _histoire())
    message_synthese = ia.chats.dernier.envoyes[-1]
    assert "Observations" in message_synthese
    assert "calculatrice" in message_synthese


# --- Stagnation : un appel déjà fait n'est jamais repayé ---------------------


def test_appel_duplicate_servi_sans_reexecuter_l_outil(monkeypatch, ia):
    # L'agent redemande exactement le même appel : aucune nouvelle exécution,
    # donc aucune requête réseau de plus. Faute de progrès, il est invité à
    # conclure plutôt que d'épuiser les dix itérations du budget.
    appels = []
    _outil_compteur(monkeypatch, "meteo", "22°C à Paris", appels)
    ia.chats.reponses = [
        reponse_outil("meteo", {"ville": "Paris"}),
        reponse_outil("meteo", {"ville": "Paris"}),
        reponse_finale("Synthèse de repli."),
    ]
    statuts = []
    texte = get_ai_response(ia, _histoire(), status_callback=statuts.append)

    assert appels == [{"ville": "Paris"}]  # une seule exécution réelle
    assert "Synthèse de repli." in texte
    assert any("Résultat déjà obtenu" in s for s in statuts)
    # Question, retours d'outil, demande de synthèse : trois envois au lieu de
    # onze, et une seule requête réseau au lieu de deux.
    assert len(ia.chats.dernier.envoyes) == 3
    # Le résultat déjà obtenu reste visible dans le rappel fait à la synthèse.
    assert "(déjà appelé) 22°C à Paris" in ia.chats.dernier.envoyes[-1]


def test_repetition_toleree_rend_le_resultat_mis_en_cache(monkeypatch, ia):
    # Le seuil de stagnation est relevé : la première répétition est tolérée, le
    # résultat mis en cache est rendu au modèle avec un avertissement, et la
    # boucle continue au lieu de conclure aussitôt.
    monkeypatch.setattr(engine, "STAGNATION_MAX_APPELS_IDENTIQUES", 2)
    appels = []
    _outil_compteur(monkeypatch, "meteo", "22°C", appels)
    ia.chats.reponses = [
        reponse_outil("meteo", {"ville": "Paris"}),
        reponse_outil("meteo", {"ville": "Paris"}),
        reponse_finale("Réponse finale."),
    ]
    assert get_ai_response(ia, _histoire()) == "Réponse finale."
    assert appels == [{"ville": "Paris"}]
    rendu = ia.chats.dernier.envoyes[-1][0].function_response.response
    assert rendu["result"] == "22°C"
    assert "déjà été effectué" in rendu["avertissement"]


def test_stagnation_quand_la_synthese_forcee_echoue_renvoie_le_repli(monkeypatch, ia):
    # Même bloqué, l'agent doit répondre quelque chose : la synthèse en panne
    # ne propage pas l'erreur jusqu'à l'interface.
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "meteo", lambda ville: "22°C")
    ia.chats.reponses = [
        reponse_outil("meteo", {"ville": "Paris"}),
        reponse_outil("meteo", {"ville": "Paris"}),
    ]  # rien en réserve pour l'appel de synthèse : il lève


# --- Auto-critique de la réponse ---------------------------------------------


def test_auto_critique_validee_renvoie_la_reponse(monkeypatch, ia):
    monkeypatch.setattr(engine, "AGENT_CRITIQUE_ACTIVEE", True)
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "calculatrice", lambda expression: "4")
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "2 + 2"}),
        reponse_finale("Le résultat est 4."),
        reponse_finale("OK"),
    ]
    statuts = []
    assert get_ai_response(ia, _histoire(), status_callback=statuts.append) == (
        "Le résultat est 4."
    )
    assert any("Vérification de la réponse" in s for s in statuts)
    # Question, retours d'outil, critique : trois envois, et la critique est
    # demandée dans la session de l'agent, outils explicitement coupés.
    assert len(ia.chats.dernier.envoyes) == 3
    assert ia.chats.dernier.configs[-1] is engine.CONFIG_SANS_OUTILS
    assert "Le résultat est 4." in ia.chats.dernier.envoyes[-1]


def test_auto_critique_insuffisante_fait_reprendre_la_reponse(monkeypatch, ia):
    monkeypatch.setattr(engine, "AGENT_CRITIQUE_ACTIVEE", True)
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "calculatrice", lambda expression: "4")
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "2 + 2"}),
        reponse_finale("Je ne sais pas."),
        reponse_finale("INSUFFISANT : le calcul donne 4, la réponse doit le dire"),
        reponse_finale("2 + 2 = 4."),
    ]
    statuts = []
    assert get_ai_response(ia, _histoire(), status_callback=statuts.append) == (
        "2 + 2 = 4."
    )
    assert any("insuffisante" in s for s in statuts)
    relance = ia.chats.dernier.envoyes[-1]
    assert "Auto-critique" in relance
    assert "le calcul donne 4" in relance


def test_auto_critique_ne_s_applique_qu_une_seule_fois(monkeypatch, ia):
    # La reprise n'est pas relue : une question ne paie jamais deux critiques,
    # et une seconde réponse jugée insuffisante serait malgré tout envoyée.
    monkeypatch.setattr(engine, "AGENT_CRITIQUE_ACTIVEE", True)
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "calculatrice", lambda expression: "4")
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "2 + 2"}),
        reponse_finale("premiere reponse"),
        reponse_finale("INSUFFISANT : sans detail"),
        reponse_finale("seconde reponse"),
        reponse_finale("INSUFFISANT : encore insuffisant"),
    ]
    assert get_ai_response(ia, _histoire()) == "seconde reponse"
    # Le cinquieme verdict n'est jamais demande : la boucle s'arrete la.
    assert len(ia.chats.dernier.envoyes) == 4


def test_question_sans_outil_n_est_jamais_critiquee(monkeypatch, ia):
    # Le « bonjour » du matin ne doit pas coûter le double : sans observation,
    # rien à confronter, la réponse part directement.
    monkeypatch.setattr(engine, "AGENT_CRITIQUE_ACTIVEE", True)
    ia.chats.reponses = [reponse_finale("Bonjour.")]
    assert get_ai_response(ia, _histoire()) == "Bonjour."
    assert len(ia.chats.dernier.envoyes) == 1


def test_auto_critique_en_panne_n_empeche_pas_de_repondre(monkeypatch, ia):
    monkeypatch.setattr(engine, "AGENT_CRITIQUE_ACTIVEE", True)
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "calculatrice", lambda expression: "4")
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": "2 + 2"}),
        reponse_finale("Le résultat est 4."),
    ]  # aucune réponse en réserve pour la critique : elle lève
    assert get_ai_response(ia, _histoire()) == "Le résultat est 4."


# --- Planification systématique ----------------------------------------------


def test_planification_systematique_meme_pour_une_question_courte(monkeypatch, ia):
    # La planification précède toujours la boucle : même « bonjour » passe par le
    # planificateur, qui peut alors répondre directement (voir plus bas).
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", True)
    ia.chats.reponses = [
        reponse_finale("PLAN\n1. Saluer\n2. Répondre"),
        reponse_finale("Bonjour !"),
    ]
    statuts = []
    assert get_ai_response(ia, _histoire(), status_callback=statuts.append) == (
        "Bonjour !"
    )
    chat_plan, chat_agent = ia.chats.chats_crees
    # Outils coupés dès la création de la session de planification.
    assert ia.chats.appels[0]["config"] is engine.CONFIG_SANS_OUTILS
    assert chat_plan.envoyes[0].startswith("Tu es le planificateur")
    assert "1. Saluer" in chat_agent.envoyes[0]
    assert any("Création du plan de réponse" in s for s in statuts)


def test_plan_non_active_ne_coute_aucun_appel(monkeypatch, ia):
    # Drapeau retiré (interrupteur utilisé par les tests) : la boucle agit seule.
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", False)
    ia.chats.reponses = [reponse_finale("Réponse directe.")]
    assert get_ai_response(ia, _histoire(_question_longue())) == "Réponse directe."
    assert len(ia.chats.chats_crees) == 1


def test_le_planificateur_partage_le_contexte_sans_le_polluer(monkeypatch, ia):
    # Le planificateur voit la conversation (indispensable pour juger un suivi),
    # mais dans sa propre copie : son aller-retour ne s'ajoute pas à la session de
    # l'agent, qui garde l'historique d'origine.
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", True)
    ia.chats.reponses = [
        reponse_finale("PLAN\n1. Collecter\n2. Conclure"),
        reponse_finale("Réponse finale."),
    ]
    statuts = []
    assert get_ai_response(
        ia, _histoire(_question_longue()), status_callback=statuts.append
    ) == ("Réponse finale.")
    chat_plan, chat_agent = ia.chats.chats_crees
    plan_histoire = ia.chats.appels[0]["history"]
    agent_histoire = ia.chats.appels[1]["history"]
    assert plan_histoire == agent_histoire  # même contexte...
    assert plan_histoire is not agent_histoire  # ...mais listes distinctes
    assert ia.chats.appels[0]["config"] is engine.CONFIG_SANS_OUTILS
    # Outils coupés dès la création : aucune config repassée aux envois du plan.
    assert chat_plan.configs == [None]
    assert chat_plan.envoyes[0].startswith("Tu es le planificateur")
    assert any("Plan retenu" in s for s in statuts)
    # Le plan est rappelé dans la première question de l'agent, qui garde la sienne.
    premiere_question = chat_agent.envoyes[0]
    assert "[Plan à suivre]" in premiere_question
    assert "1. Collecter" in premiere_question
    assert _question_longue() in premiere_question


def test_le_planificateur_repond_directement_sans_ouvrir_la_boucle(monkeypatch, ia):
    # Le planificateur a la réponse : elle est rendue telle quelle, sans seconde
    # session ni boucle ReAct. La question ne coûte qu'un appel.
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", True)
    ia.chats.reponses = [reponse_finale("REPONSE\nBonjour !")]
    statuts = []
    assert get_ai_response(ia, _histoire(), status_callback=statuts.append) == (
        "Bonjour !"
    )
    assert len(ia.chats.chats_crees) == 1
    assert any("Réponse directe" in s for s in statuts)


def test_reponse_directe_publiee_en_flux_sans_son_marqueur(monkeypatch, ia):
    # La réponse directe s'écrit au fil de l'eau : le marqueur du protocole ne
    # doit jamais apparaître à l'écran.
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", True)
    ia.chats.reponses = [reponse_finale("REPONSE : 2 + 2 = 4")]
    publies = []
    assert get_ai_response(ia, _histoire(), texte_callback=publies.append) == (
        "2 + 2 = 4"
    )
    assert publies[-1] == "2 + 2 = 4"
    assert not any("REPONSE" in texte for texte in publies)


def test_planificateur_sans_verdict_l_agent_agit_seul(monkeypatch, ia):
    # Sortie vide : ni plan ni réponse. Le statut le dit, et la question part
    # inchangée vers la boucle.
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", True)
    ia.chats.reponses = [reponse_finale(""), reponse_finale("Réponse.")]
    statuts = []
    assert get_ai_response(ia, _histoire(), status_callback=statuts.append) == (
        "Réponse."
    )
    assert any("Aucun plan retenu" in s for s in statuts)
    assert ia.chats.chats_crees[1].envoyes[0] == "ma question"


def test_panne_du_planificateur_n_empeche_pas_de_repondre(monkeypatch, ia):
    # La session de planification tombe en panne : l'échec est absorbé et l'agent
    # agit quand même, sans plan et sans réponse directe.
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", True)
    vrai_create = ia.chats.create

    def echouer(*arguments, **mots_cles):
        raise RuntimeError("service indisponible")

    def create_avec_panne(model, history, config):
        chat = vrai_create(model, history, config)
        if len(ia.chats.chats_crees) == 1:  # première session : le planificateur
            chat.send_message_stream = echouer
        return chat

    monkeypatch.setattr(ia.chats, "create", create_avec_panne)
    ia.chats.reponses = [reponse_finale("Réponse sans plan.")]
    publies = []
    assert (
        get_ai_response(
            ia, _histoire(_question_longue()), texte_callback=publies.append
        )
        == "Réponse sans plan."
    )
    # La panne efface la zone de réponse, au cas où un plan s'y était glissé.
    assert "" in publies
    # La question part telle quelle : aucun plan n'a été ajouté au message.
    assert ia.chats.chats_crees[1].envoyes[0] == _question_longue()


def test_cle_appel_ignore_l_ordre_des_arguments():
    # Deux mappings équivalents du point de vue du modèle donnent la même clé :
    # la stagnation est détectée même si l'ordre des arguments change.
    assert engine._cle_appel("meteo", {"ville": "Paris", "unite": "C"}) == (
        engine._cle_appel("meteo", {"unite": "C", "ville": "Paris"})
    )
    assert engine._cle_appel("meteo", {"ville": "Paris"}) != (
        engine._cle_appel("meteo", {"ville": "Lyon"})
    )
    # Arguments absents : clé stable malgré tout.
    assert engine._cle_appel("outil", None) == engine._cle_appel("outil", {})


def test_cle_appel_replie_les_arguments_non_serialisables():
    # Un argument non JSON (ici un ensemble) ne doit pas faire échouer l'appel :
    # la clé retombe sur sa représentation texte.
    cle = engine._cle_appel("outil", {"essai": {1, 2}})
    assert cle.startswith("outil ")


def test_synthese_forcee_rappelle_le_plan(monkeypatch, ia):
    # Le plan survit jusqu'à la synthèse forcée : la conclusion doit couvrir les
    # étapes prévues, même dix itérations plus tard.
    monkeypatch.setattr(engine, "AGENT_PLAN_ACTIVEE", True)
    ia.chats.reponses = (
        [reponse_finale("1. Chercher\n2. Croiser\n3. Conclure")]
        + [
            reponse_outil("calculatrice", {"expression": f"{i} + 2"})
            for i in range(MAX_ITERATIONS)
        ]
        + [reponse_finale("Synthèse.")]
    )
    assert "Synthèse." in get_ai_response(ia, _histoire(_question_longue()))
    message_synthese = ia.chats.dernier.envoyes[-1]
    assert "Plan initial à couvrir" in message_synthese
    assert "2. Croiser" in message_synthese
    # Le plan vit dans sa propre session : la boucle de l'agent n'a pas grossi.
    # MAX_ITERATIONS envois d'agent + 1 envoi de synthèse forcée.
    assert len(ia.chats.dernier.envoyes) == MAX_ITERATIONS + 1


# --- Réponse publiée au fil de l'eau -----------------------------------------


def test_envoyer_sans_callback_livre_la_reponse_d_un_bloc(ia):
    # Sans callback, l'envoi reste un send_message : c'est le chemin de tous les
    # tests qui ne s'intéressent pas au flux.
    ia.chats.reponses = [reponse_finale("Bonjour !")]
    chat = ia.chats.create(model=CHAT_MODEL, history=[], config=None)
    reponse = engine._envoyer(chat, "question")
    assert reponse.text == "Bonjour !"
    assert chat.envoyes == ["question"]
    assert chat.configs == [None]


def test_envoyer_agrege_les_morceaux_du_flux(ia):
    # La réponse reconstruite porte la même information qu'une réponse d'un bloc :
    # c'est ce qui permet à la boucle d'ignorer le mode d'envoi.
    ia.chats.reponses = [reponse_finale("Bonjour tout le monde")]
    chat = ia.chats.create(model=CHAT_MODEL, history=[], config=None)
    publies = []
    reponse = engine._envoyer(chat, "question", texte_callback=publies.append)
    assert reponse.text == "Bonjour tout le monde"
    assert reponse.function_calls == []
    # Plusieurs morceaux, et chaque état publié prolonge le précédent.
    assert len(publies) > 1
    assert all(
        avant == apres[: len(avant)]
        for avant, apres in zip(publies, publies[1:], strict=False)
    )
    assert publies[-1] == "Bonjour tout le monde"


def test_envoyer_agrege_un_appel_d_outil(ia):
    ia.chats.reponses = [reponse_outil("meteo", {"ville": "Paris"})]
    chat = ia.chats.create(model=CHAT_MODEL, history=[], config=None)
    publies = []
    reponse = engine._envoyer(chat, "question", texte_callback=publies.append)
    assert reponse.text == ""
    assert reponse.function_calls[0].name == "meteo"
    assert reponse.function_calls[0].args == {"ville": "Paris"}
    # Aucun texte à montrer : la zone de réponse est effacée.
    assert publies == [""]


def test_le_texte_est_publie_au_fil_de_l_eau(ia):
    ia.chats.reponses = [reponse_finale("Le résultat est quatre.")]
    publies = []
    statuts = []
    assert (
        get_ai_response(
            ia,
            _histoire(),
            status_callback=statuts.append,
            texte_callback=publies.append,
        )
        == "Le résultat est quatre."
    )
    assert publies[-1] == "Le résultat est quatre."
    # En flux, la rédaction n'est plus annoncée : elle se voit à l'écran.
    assert not any("Rédaction" in s for s in statuts)


def test_le_texte_d_un_tour_d_outil_est_efface(monkeypatch, ia):
    # Le modèle annonce une action puis appelle l'outil : ce texte n'est pas la
    # réponse, la zone doit être vidée avant que la vraie réponse n'arrive.
    monkeypatch.setitem(engine.AVAILABLE_TOOLS, "meteo", lambda ville: "22°C")
    ia.chats.reponses = [
        reponse_texte_et_outil("Je vérifie la météo.", "meteo", {"ville": "Paris"}),
        reponse_finale("Il fait 22°C à Paris."),
    ]
    publies = []
    assert get_ai_response(ia, _histoire(), texte_callback=publies.append) == (
        "Il fait 22°C à Paris."
    )
    assert "Je vérifie la météo." in publies  # publié...
    assert "" in publies  # ...puis effacé (le tour était un tour d'outil)
    assert publies[-1] == "Il fait 22°C à Paris."


def test_flux_coupe_par_le_drapeau(monkeypatch, ia):
    # Valeur de sécurité : flux coupé, la réponse arrive d'un bloc et la rédaction
    # est de nouveau annoncée dans le statut.
    monkeypatch.setattr(engine, "AGENT_FLUX_ACTIVE", False)
    ia.chats.reponses = [reponse_finale("Bonjour !")]
    publies = []
    statuts = []
    assert (
        get_ai_response(
            ia,
            _histoire(),
            status_callback=statuts.append,
            texte_callback=publies.append,
        )
        == "Bonjour !"
    )
    assert publies == []
    assert any("Rédaction" in s for s in statuts)


def test_la_synthese_forcee_est_publiee_en_flux(ia):
    # La réponse de secours s'écrit elle aussi au fil de l'eau : la zone ne doit
    # pas se figer puis se remplir d'un coup.
    ia.chats.reponses = [
        reponse_outil("calculatrice", {"expression": f"{i} + 1"})
        for i in range(MAX_ITERATIONS)
    ] + [reponse_finale("Synthèse finale.")]
    publies = []
    texte = get_ai_response(ia, _histoire(), texte_callback=publies.append)
    assert "Synthèse finale." in texte
    assert any("Synthèse finale." in etat for etat in publies)

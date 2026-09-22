"""Fixtures, doubles (fakes) et habillage du rapport de couverture.

Aucun test ne touche au réseau, ni à Supabase, ni à Gemini : les doubles
rejouent exactement les appels réels (table()/rpc()/models.embed_content/
chats.create), ce qui permet de vérifier les contrats (ordre des écritures,
batching, replis) sans rien consommer.

Le thème « ACE CHAT » du rapport tests/htmlcov/ est appliqué automatiquement à
chaque exécution (hook atexit) : `python -m pytest` suffit.
"""

import atexit
import io
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest

# Le projet doit rester importable quel que soit le mode de lancement.
RACINE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))


# ---------------------------------------------------------------------------
# Supabase simulé (même API que celle utilisée par modules/database.py)
# ---------------------------------------------------------------------------

class _Reponse:
    """Même forme que supabase-py : objet avec .data."""

    def __init__(self, data):
        self.data = data


class _Requete:
    """Requête chaînable : select/insert/delete -> .eq* -> .order -> .execute()."""

    def __init__(self, supabase, table, action, payload=None):
        self._supabase = supabase
        self._table = table
        self._action = action
        self._payload = payload
        self._filtres = {}
        self._ordre = None

    def eq(self, colonne, valeur):
        self._filtres[colonne] = valeur
        return self

    def order(self, colonne):
        self._ordre = colonne
        return self

    def _selection(self):
        return [
            ligne
            for ligne in self._supabase.tables.get(self._table, [])
            if all(ligne.get(c) == v for c, v in self._filtres.items())
        ]

    def execute(self):
        if self._action == "select":
            if self._table in self._supabase.echecs_select:
                raise RuntimeError(f"echec simule : select {self._table}")
            lignes = self._selection()
            if self._ordre:
                lignes = sorted(lignes, key=lambda l: l.get(self._ordre) or "")
            return _Reponse([dict(l) for l in lignes])

        if self._action == "insert":
            if self._table in self._supabase.echecs_insert:
                raise RuntimeError(f"echec simule : insert {self._table}")
            self._supabase.tables.setdefault(self._table, []).extend(
                dict(l) for l in self._payload
            )
            self._supabase.journal.append(
                ("insert", self._table, dict(self._filtres), list(self._payload))
            )
            return _Reponse(self._payload)

        # delete
        if self._table in self._supabase.echecs_delete:
            raise RuntimeError(f"echec simule : delete {self._table}")
        restantes, supprimees = [], []
        for ligne in self._supabase.tables.get(self._table, []):
            cible = all(ligne.get(c) == v for c, v in self._filtres.items())
            (supprimees if cible else restantes).append(ligne)
        self._supabase.tables[self._table] = restantes
        self._supabase.journal.append(
            ("delete", self._table, dict(self._filtres), len(supprimees))
        )
        return _Reponse(supprimees)


class _Rpc:
    def __init__(self, supabase, nom, parametres):
        self._supabase = supabase
        self._nom = nom
        self._parametres = parametres

    def execute(self):
        self._supabase.appels_rpc.append((self._nom, dict(self._parametres)))
        if self._supabase.gestionnaire_rpc:
            # Peut lever une exception (base non migrée, panne simulée…).
            data = self._supabase.gestionnaire_rpc(self._nom, self._parametres)
            return _Reponse(data)
        return _Reponse(self._supabase.resultats_rpc.get(self._nom, []))


class _TableFacade:
    def __init__(self, supabase, nom):
        self._supabase = supabase
        self._nom = nom

    def select(self, colonnes="*"):
        return _Requete(self._supabase, self._nom, "select")

    def insert(self, payload):
        lignes = payload if isinstance(payload, list) else [payload]
        return _Requete(self._supabase, self._nom, "insert", payload=lignes)

    def delete(self):
        return _Requete(self._supabase, self._nom, "delete")


class FakeSupabase:
    """Client Supabase en mémoire.

    - `tables` : {nom_table: [lignes]} (état réellement consulté)
    - `journal` : écritures dans l'ordre exact ; éléments
      ("insert", table, filtres, payload) / ("delete", table, filtres, n)
    - `appels_rpc` : [(nom, paramètres)]
    - `resultats_rpc` : {nom: lignes} renvoyées par la fonction SQL simulée
    - `gestionnaire_rpc` : callable(nom, params) -> lignes, peut lever
    - `echecs_*` : noms de tables dont select/insert/delete lèvent une erreur
    """

    def __init__(self, tables=None, journal=None):
        self.tables = {
            n: [dict(l) for l in lignes] for n, lignes in (tables or {}).items()
        }
        self.journal = journal if journal is not None else []
        self.appels_rpc = []
        self.resultats_rpc = {}
        self.gestionnaire_rpc = None
        self.echecs_select = set()
        self.echecs_insert = set()
        self.echecs_delete = set()

    def table(self, nom):
        return _TableFacade(self, nom)

    def rpc(self, nom, parametres):
        return _Rpc(self, nom, parametres)


# ---------------------------------------------------------------------------
# Client Gemini simulé (embeddings + chats ReAct)
# ---------------------------------------------------------------------------

class _ModeleEmbed:
    def __init__(self, journal):
        self._journal = journal
        self.nb_retourne = None  # force un mauvais alignement si fixé
        self.erreur = None       # exception à lever (panne API simulée)

    def embed_content(self, model, contents):
        if self.erreur is not None:
            raise self.erreur
        textes = [contenu.parts[0].text for contenu in contents]
        self._journal.append(("embed", model, len(textes)))
        nb = len(textes) if self.nb_retourne is None else self.nb_retourne
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(len(t)), 1.0, 2.0])
                for t in textes[:nb]
            ]
        )


class _Chat:
    def __init__(self, reponses):
        self._reponses = list(reponses)
        self.envoyes = []  # chaque message passé à send_message

    def send_message(self, message):
        self.envoyes.append(message)
        return self._reponses.pop(0)


class _Chats:
    def __init__(self):
        self.reponses = []    # file des réponses du prochain chat
        self.creation = None  # dernier appel create(model=, history=, config=)

        self.dernier = None     # dernier chat créé (pour inspecter les envois)

    def create(self, model, history, config):
        self.creation = {"model": model, "history": history, "config": config}
        self.dernier = _Chat(self.reponses)
        return self.dernier


class FakeIA:
    """Client google-genai simulé : `.models.embed_content` et `.chats.create`."""

    def __init__(self, journal=None):
        self.journal = journal if journal is not None else []
        self.models = _ModeleEmbed(self.journal)
        self.chats = _Chats()


# ---------------------------------------------------------------------------
# Briques de test partagées
# ---------------------------------------------------------------------------

def reponse_finale(texte):
    """Réponse de modèle sans appel d'outil (fin de la boucle ReAct)."""
    return SimpleNamespace(function_calls=[], text=texte)


def reponse_outil(nom, args):
    """Réponse de modèle demandant un appel d'outil."""
    return SimpleNamespace(
        function_calls=[SimpleNamespace(name=nom, args=args)], text=""
    )


def fichier(nom, contenu):
    """Fichier uploadé simulé : BytesIO avec .name, comme côté Streamlit."""
    octets = contenu if isinstance(contenu, bytes) else contenu.encode("utf-8")
    flux = io.BytesIO(octets)
    flux.name = nom
    return flux


@pytest.fixture
def supabase():
    return FakeSupabase()


@pytest.fixture
def ia():
    return FakeIA()


@pytest.fixture(autouse=True)
def filtre_document_par_defaut():
    """Chaque test part du drapeau « la base sait filtrer » = True."""
    from modules.database import _filtre_document_par_sql
    _filtre_document_par_sql["disponible"] = True
    yield
    _filtre_document_par_sql["disponible"] = True


# ---------------------------------------------------------------------------
# Habillage automatique du rapport tests/htmlcov/ (thème sombre + marque)
# ---------------------------------------------------------------------------

def themer_htmlcov():
    """Applique le thème ACE CHAT à tests/htmlcov/ — idempotent (marqueurs)."""
    dossier = RACINE / "tests" / "htmlcov"
    theme = pathlib.Path(__file__).with_name("_theme_htmlcov.css")
    # coverage peut nommer la feuille style.css ou style_<hash>.css.
    styles = sorted(dossier.glob("style*.css")) if dossier.exists() else []
    if styles and theme.exists():
        for style in styles:
            css = style.read_text(encoding="utf-8")
            if "ACE-CHAT-THEME" not in css:
                style.write_text(
                    css + "\n" + theme.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

    script = (
        '<script>/* ACE-CHAT-BRAND */\n'
        "(function(){var b=document.querySelector('body');"
        "if(b&&!document.querySelector('.ace-brand')){"
        "var d=document.createElement('div');d.className='ace-brand';"
        "d.textContent='\\u26A1 ACE CHAT \\u00B7 Rapport de couverture';"
        "b.insertBefore(d,b.firstChild);}"
        "document.querySelectorAll('td,span').forEach(function(el){"
        "var t=el.textContent.trim(),x=t.match(/^(\\d+(?:\\.\\d+)?)%$/);"
        "if(!x)return;var v=parseFloat(x[1]);"
        "var c=v>=90?'#34d399':v>=70?'#fbbf24':'#f87171';"
        "el.style.setProperty('color',c,'important');"
        "el.style.setProperty('font-weight','700','important');});})();</script>\n</body>"
    )
    pages = sorted(dossier.glob("*.html")) if dossier.exists() else []
    for page in pages:
        texte = page.read_text(encoding="utf-8")
        modifie = False
        if page.name == "index.html" and "ACE CHAT" not in texte:
            texte = re.sub(
                r"<title>.*?</title>",
                "<title>ACE CHAT · Rapport de couverture</title>",
                texte,
                flags=re.S,
            )
            modifie = True
        if "ACE-CHAT-BRAND" not in texte:
            texte = texte.replace("</body>", script)
            modifie = True
        if modifie:
            page.write_text(texte, encoding="utf-8")


# atexit tourne après l'écriture du rapport par pytest-cov : déterministe.
atexit.register(themer_htmlcov)


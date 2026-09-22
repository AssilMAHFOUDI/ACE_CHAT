import logging

import pypdf
from pypdf.errors import PdfReadError

from modules.ai_engine import get_embeddings
from modules.config import (
    CHUNK_MIN_LENGTH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    INSERT_BATCH_SIZE,
)
from modules.database import clear_document_chunks

logger = logging.getLogger(__name__)


def extract_text_from_file(fichier_upload):
    """
    Extrait le texte d'un fichier uploadé via Streamlit (TXT ou PDF).

    Lève une ValueError si le fichier est vide ou illisible : l'interface
    affiche alors un message compréhensible au lieu d'une trace pypdf.
    """
    if fichier_upload is None:
        return ""

    # On se replace au début du flux : pypdf lit à partir de la position
    # courante, et Streamlit réutilise le même objet d'un rerun à l'autre.
    # Sans ce seek(0), une seconde lecture voit un fichier de 0 octet.
    try:
        fichier_upload.seek(0)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        contenu = fichier_upload.getvalue()
    except AttributeError:
        contenu = b""

    if len(contenu) == 0:
        raise ValueError("le fichier est vide (0 octet).")

    nom_fichier = (getattr(fichier_upload, "name", "") or "").lower()

    if nom_fichier.endswith(".txt"):
        return contenu.decode("utf-8", errors="replace")

    if nom_fichier.endswith(".pdf"):
        try:
            pdf_reader = pypdf.PdfReader(fichier_upload)
            if pdf_reader.is_encrypted and not pdf_reader.decrypt(""):
                raise ValueError("le PDF est protégé par un mot de passe.")
            pages = [page.extract_text() for page in pdf_reader.pages]
        except PdfReadError as erreur:
            raise ValueError(f"le PDF est illisible ({erreur}).") from erreur
        return "\n".join(texte for texte in pages if texte)

    return ""


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Découpe un texte long en plusieurs morceaux (chunks)
    avec un chevauchement pour ne pas perdre le sens.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # On définit la fin de notre morceau
        end = start + chunk_size

        # On extrait le texte
        chunk = text[start:end]
        chunks.append(chunk)

        # On avance le curseur pour le prochain morceau,
        # en reculant de la valeur de l'overlap
        start += (chunk_size - overlap)

    return chunks


def process_and_store_document(text, file_name, session_id, supabase_client, ai_client, progress_callback=None):
    """
    Orchestre le découpage, la vectorisation et la sauvegarde dans Supabase.

    La base de connaissance d'une session peut contenir PLUSIEURS documents :
    seuls les morceaux portant le même nom de fichier sont remplacés (mise à
    jour d'un document), les autres documents de la session sont conservés.

    L'ordre des étapes est volontaire :

    1. on calcule d'abord TOUS les vecteurs (aucune écriture en base) ;
    2. on supprime ensuite les chunks déjà présents pour CE document ;
    3. on insère enfin les nouveaux chunks.

    Ainsi un document B chargé après un document A ne peut plus mélanger les
    deux dans la recherche sémantique, et un échec de l'API d'embedding laisse
    la version précédente intacte en base.

    `progress_callback(done, total)` est appelé après chaque paquet
    d'embeddings pour alimenter la barre de progression de l'interface.

    Retourne le nombre de morceaux stockés (0 si le document est illisible).
    """
    # 1. On découpe le texte et on ignore les morceaux trop vides
    chunks = [
        chunk
        for chunk in split_text_into_chunks(text)
        if len(chunk.strip()) >= CHUNK_MIN_LENGTH
    ]
    total = len(chunks)
    logger.info("Découpage terminé : %s morceaux exploitables pour %s.", total, file_name)

    if total == 0:
        logger.warning(
           "⚠️ Aucun morceau exploitable dans %s : rien n'a été stocké.", file_name
        )
        return 0

    # 2. On calcule les vecteurs par paquets (1 appel API pour N morceaux)
    vectors = []
    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        vectors.extend(get_embeddings(batch, ai_client))
        if progress_callback:
            progress_callback(len(vectors), total)

    # 3. On remplace la version précédente de CE document (les autres documents
    #    de la session sont conservés)
    if not clear_document_chunks(supabase_client, session_id, file_name):
        raise RuntimeError(
            "suppression de l'ancienne version de ce document impossible : "
            "ingestion annulée pour ne pas créer de doublons."
        )

    # 4. On insère les nouveaux chunks par paquets
    rows = [
        {
            "session_id": session_id,
            "file_name": file_name,
            "content": chunk,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        supabase_client.table("document_chunks").insert(
            rows[start:start + INSERT_BATCH_SIZE]
        ).execute()

    logger.info(
       "✅ %s morceaux vectorisés et sauvegardés dans Supabase pour %s.",
        len(rows),
        file_name,
    )
    return len(rows)

import logging

import pypdf

from modules.ai_engine import get_embeddings
from modules.database import clear_document_chunks

logger = logging.getLogger(__name__)

# Nombre de morceaux envoyés en un seul appel à l'API d'embedding
EMBEDDING_BATCH_SIZE = 20
# Nombre de lignes envoyées en un seul INSERT Supabase
INSERT_BATCH_SIZE = 50


def extract_text_from_file(fichier_upload):
    """
    Extrait le texte d'un fichier uploadé via Streamlit (TXT ou PDF).
    """
    if fichier_upload is None:
        return ""

    if fichier_upload.name.endswith(".txt"):
        return fichier_upload.getvalue().decode("utf-8")

    elif fichier_upload.name.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(fichier_upload)
        texte_document = "\n".join(
            [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        )
        return texte_document

    return ""


def split_text_into_chunks(text, chunk_size=1000, overlap=200):
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

    L'ordre des étapes est volontaire :

    1. on calcule d'abord TOUS les vecteurs (aucune écriture en base) ;
    2. on supprime ensuite les chunks déjà présents pour cette session ;
    3. on insère enfin les nouveaux chunks.

    Ainsi un document B chargé après un document A ne peut plus mélanger les
    deux dans la recherche sémantique, et un échec de l'API d'embedding laisse
    l'ancien document intact en base.

    `progress_callback(done, total)` est appelé après chaque paquet
    d'embeddings pour alimenter la barre de progression de l'interface.

    Retourne le nombre de morceaux stockés (0 si le document est illisible).
    """
    # 1. On découpe le texte et on ignore les morceaux trop vides
    chunks = [
        chunk
        for chunk in split_text_into_chunks(text, chunk_size=1000, overlap=200)
        if len(chunk.strip()) >= 10
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

    # 3. On remplace l'ancien document de cette session
    if not clear_document_chunks(supabase_client, session_id):
        raise RuntimeError(
            "suppression des anciens chunks impossible : ingestion annulée pour "
            "ne pas mélanger deux documents dans la même session."
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

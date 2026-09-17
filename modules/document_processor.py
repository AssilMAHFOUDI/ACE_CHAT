import pypdf

from modules.ai_engine import get_embedding 

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

def process_and_store_document(text, file_name, session_id, supabase_client, ai_client):
    """
    Orchestre le découpage, la vectorisation et la sauvegarde dans Supabase.
    """
    # On importe la fonction d'embedding depuis ton autre fichier
    
    
    # 1. On découpe le texte
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)
    
    print(f"Découpage terminé : {len(chunks)} morceaux trouvés pour {file_name}.")
    
    # 2. On traite chaque morceau
    for chunk in chunks:
        # On ignore les morceaux trop vides
        if len(chunk.strip()) < 10:
            continue
            
        # A. On demande le vecteur (l'embedding) à Gemini
        vector = get_embedding(chunk, ai_client)
        
        # B. On prépare la "boîte" de données pour Supabase
        data = {
            "session_id": session_id,
            "file_name": file_name,
            "content": chunk,
            "embedding": vector
        }
        
        # C. On insère dans la base de données
        supabase_client.table("document_chunks").insert(data).execute()
        
    print("✅ Document entièrement vectorisé et sauvegardé dans Supabase !")




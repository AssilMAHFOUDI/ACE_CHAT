import pypdf


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

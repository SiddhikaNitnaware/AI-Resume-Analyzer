import pymupdf

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = pymupdf.open(pdf_path)

    for page in doc:
        text += page.get_text()

    return text.lower()
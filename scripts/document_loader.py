import os
from langchain.document_loaders import UnstructuredPDFLoader

def load_pdfs(pdf_folder):
    """Loads and processes PDFs from the specified directory."""
    pdf_files = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.endswith('.pdf')
    ]
    
    documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = UnstructuredPDFLoader(file_path=pdf_file, mode='single')
        documents.extend(loader.load())  # Load and append in one step
    return documents
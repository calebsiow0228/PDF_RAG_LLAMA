from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

def initialize_embeddings(model_name):
    """Initializes the embeddings model."""
    if model_name.startswith('nomic-embed-text'):
        return OllamaEmbeddings(model=model_name)
    else:
        return HuggingFaceEmbeddings(model_name)
import os
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def create_vectorstore(docs, persist_directory):
    """Creates or loads the ChromaDB vector store."""
    os.makedirs(persist_directory, exist_ok=True)  # Simplified folder creation

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model='nomic-embed-text', show_progress=True),
        collection_name="local-rag",
        persist_directory=persist_directory
    )
    return vectorstore
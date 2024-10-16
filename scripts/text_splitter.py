from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=7500, chunk_overlap=100):
    """Splits documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)  # Direct return of the result
# %%
import warnings
warnings.filterwarnings("ignore")
import os

from scripts.document_loader import load_pdfs
from scripts.text_splitter import split_documents
from scripts.embeddings_initializer import initialize_embeddings
from scripts.vectorstore_creator import create_vectorstore
from scripts.llm_initializer import initialize_llm
from scripts.qa_chain_setup import initialize_retriever, setup_qa_chain
from langchain.vectorstores import Chroma

def main():
    # Configuration parameters
    pdf_folder = 'pdfs'
    embedding_model_name = 'nomic-embed-text'
    llm_model_name = 'llama3.2'  # Use a smaller model for testing
    persist_directory = 'data/chroma_db'

    # Step 1: Initialize embeddings
    embeddings = initialize_embeddings(embedding_model_name)

    # Step 2: Check if vector database already exists
    vector_db_exists = os.path.exists(os.path.join(persist_directory, "index"))  # Checking if Chroma's index exists

    if vector_db_exists:
        print("Vector database already exists, loading the existing vectorstore...")
        # Load the existing vectorstore (e.g., Chroma)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Vector database does not exist, processing documents and creating vectorstore...")

        # Load and process documents
        documents = load_pdfs(pdf_folder)

        # Split documents into chunks
        chunks = split_documents(documents)

        # Create or load ChromaDB vector store
        vectorstore = create_vectorstore(chunks, persist_directory)

    # Step 5: Initialize LLM
    llm = initialize_llm(llm_model_name)

    # Step 6: Initialize retriever
    retriever = initialize_retriever(llm, vectorstore)

    # Step 7: Set up the full QA chain
    qa_chain = setup_qa_chain(llm, retriever)

    # Step 8: Run the interactive QA loop
    def interactive_qa_loop(qa_chain):
        print("\nInteractive QA session started. Type 'exit' to quit.")
        while True:
            # Get user input
            query = input("\nEnter your question: ")
            if query.lower() == 'exit':
                print("Exiting the QA loop.")
                break
            
            # Run the QA chain with the user query
            response = qa_chain.invoke({"query": query})
            
            # Display the response
            print(f"\nAnswer:\n{response}\n")

    # Start the interactive QA loop
    interactive_qa_loop(qa_chain)
# %%
# Run the main function if this script is executed
if __name__ == "__main__":
    main()
# %%

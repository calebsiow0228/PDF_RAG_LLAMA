# Interactive QA System

This project sets up an interactive QA (Question Answering) system that processes a set of PDFs, creates embeddings using the `nomic-embed-text` model, and stores the data in a ChromaDB vector store. Users can then ask questions based on the content of the PDFs, and the system will respond using the documents as context.

## Features
- Loads and processes PDF documents into a vector store.
- Uses `nomic-embed-text` for creating embeddings or a fallback HuggingFace model.
- Sets up an interactive QA loop where users can ask questions based on the PDFs.
- Efficient document retrieval using a multi-query retriever.

## Setup

### Prerequisites

Before running the application, ensure you have the following installed and set up:

1. **Conda** (for managing the environment)  
   - Install Conda by following the [official Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Python 3.10+**  
   - Ensure that Python 3.10 or higher is installed. You can verify your Python version by running the following command in your terminal:
     ```bash
     python --version
     ```

3. **Ollama** (Required for embeddings and LLM)  
   - Install [Ollama](https://ollama.com/) on your PC or Mac. Follow the instructions on the official website for your operating system.

4. **Required Models** (for Ollama)  
   - After installing Ollama, pull the necessary models by running the following commands in your terminal:
     ```bash
     ollama pull nomic-embed-text
     ollama pull llama3.2
     ```

### Installation
s
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/interactive-qa-system.git
   cd interactive-qa-system

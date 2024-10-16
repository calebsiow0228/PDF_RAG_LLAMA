from langchain_community.chat_models import ChatOllama

def initialize_llm(model_name: str = 'llama3.2'):
    """Initializes the LLM with the given model name."""
    return ChatOllama(
        model=model_name,  # Use the provided model_name argument
        temperature=0.3
    )
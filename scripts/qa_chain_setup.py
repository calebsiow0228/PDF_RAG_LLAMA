from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# Function to create the retriever
def initialize_retriever(llm, vector_db):
    """Initialize the retriever using the given LLM and vector database."""
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an AI assistant specialized in providing answers based solely on the documents and data stored in a vector database.

        Your task is to answer the user's question using ONLY the context provided from the retrieved documents. Do not use your own knowledge or make assumptions beyond the given information.

        If the answer cannot be found in the provided documents, say: "The information you are asking for is not available in the provided documents."

        Ensure your answer is concise, accurate, and directly addresses the user's question. Additionally, provide a summary of the most relevant retrieved documents if applicable.

        Original question: {question}
        """
    )
    return MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm=llm,
        prompt=query_prompt,
    )


# Function to create the ChatPromptTemplate
def initialize_prompt_template():
    """Create a chat prompt template for the QA chain."""
    template = """
    Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    return ChatPromptTemplate.from_template(template)


# Function to set up the full QA chain
def setup_qa_chain(llm, retriever):
    """Sets up the complete QA chain."""
    prompt = initialize_prompt_template()

    # Define the chain's flow
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# Wrapper function to run the chain with user input
def run_qa_chain(chain, question):
    """Invoke the chain with a specific question."""
    return chain.invoke(question)
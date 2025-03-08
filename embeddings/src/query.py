from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_postgres.vectorstores import PGVector
from langchain.schema import Document
from langgraph.graph import StateGraph,START
from spellchecker import SpellChecker
from langchain_core.prompts import PromptTemplate
import psycopg, requests
from typing import TypedDict, List
import re
PG_HOST = 'psql-da-chatbot.postgres.database.azure.com'
PG_DBNAME = 'postgres'
PG_USERNAME = 'postgresadmin'
PG_PASSWORD = '6KSXZFMMhZVnAKh83tmcaArJAp4'
PG_PORT = 5432
# PostgreSQL connection settings
PG_CONNECTION_STRING = f"postgresql+psycopg://{PG_USERNAME}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"

PG_CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@127.0.0.1:6024/langchain"
 

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="mistral")  
# Initialize PGVector store
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="text_embeddings",
    connection=PG_CONNECTION_STRING,
    use_jsonb=True
)

# query = "where is the market?"
# query_embedding = embeddings.embed_query(query)

# # Perform similarity search
# similar_docs = vector_store.similarity_search(query, k=3)

# for doc in similar_docs:
#     print(doc.page_content)


# === Define State for LangGraph ===
class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    web_docs: List[str]
    response: str

def web_search(state):

    search_url = f"https://www.google.com/search?q={state['query']}"

    response = requests.get(search_url)

    state['web_docs'] = response.text[:500]

    return state


def preprocess_step(state):
    spell_check = SpellChecker()

    pattern = r'([a-zA-Z]+)([^a-zA-z]*)'

    def correct_word(match):
        word = match.group(1)
        special_chars = match.group(2)
        corrected_word = spell_check.correction(word)
        if corrected_word is None:
            corrected_word = word
        # Return the corrected word with its special characters attached
        return corrected_word + special_chars
    query = state["query"]
    words = query.split()
    corrected_words = [re.sub(pattern, correct_word, word) for word in words]
    corrected_query = " ".join(corrected_words)
    print(corrected_query)
    return {"query": corrected_query}


def retrieve_step(state):
    """Retrieve relevant documents from PGVector."""
    print(state)
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(query, k =3)
    

    state["retrieved_docs"] = retrieved_docs  
    return state

def generate_step(state):
    """Use retrieved docs to generate a response with the LLM."""
    if "retrieved_docs" not in state:
        context = state["web_docs"]
    else:
        retrieved_docs = state["retrieved_docs"]
        print(retrieved_docs)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    tools = [web_search, rag]

    prompt_template = PromptTemplate(
        template=(
            "You are a helpful assistant. Given the query: '{query}', first search in the database for relevant results. "
            "If the results are not sufficient, search the web for additional information.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n"
            "Answer:"
        ),
        input_variables=["context", "query"]
    )

    # Generate response using LLM
    response = llm.invoke(prompt_template.format(context=context, query=state["query"]))
    return {"response": response}

def decision_node(state):
    """Decide whether to retrieve from the web or from PGVector."""
    state = retrieve_step(state)
    if "retrieved_docs" not in state or not state["retrieved_docs"]:
        state = web_search(state)
    else:
        # Optionally, check for very low similarity scores in the documents
        retrieved_docs = state["retrieved_docs"]
        if len(retrieved_docs) < 3:  # If too few documents were retrieved, consider it as a failure to find good results
            state = web_search(state)
    return state

# === Define LangGraph Workflow ===
workflow = StateGraph(RAGState)
workflow.add_node("preprocess", preprocess_step)
workflow.add_node("retrieve", retrieve_step)
workflow.add_node("retrieve_web", web_search)
workflow.add_node("de", decision_node)
workflow.add_node("generate", generate_step)
workflow.set_entry_point("preprocess")
workflow.add_edge(START, "preprocess")
workflow.add_edge("preprocess", "de")
workflow.add_edge("de", "generate")

# === Compile & Run ===
app = workflow.compile()

query = "when was putin born?"
result = app.invoke({"query": query})
print("RAG Response:", result["response"])
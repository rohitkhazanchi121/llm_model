from langchain_ollama import OllamaEmbeddings, OllamaLLM, ChatOllama
from langchain_postgres.vectorstores import PGVector
from langchain.schema import Document
from langgraph.graph import StateGraph,START
from langchain_community.tools import TavilySearchResults
from spellchecker import SpellChecker
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import psycopg, requests
from typing import TypedDict, List
import re, json
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
json_mode_llm = ChatOllama(model="mistral", format="json", temperature=0)
# Initialize PGVector store
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="text_embeddings",
    connection=PG_CONNECTION_STRING,
    use_jsonb=True
)

# === Define State for LangGraph ===
class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    retrieval_grade: str
    web_docs: List[Document]
    response: str

def web_search(state):

    TAVILY_API_KEY = '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'
    tools = TavilySearchResults(max_results=3, search_depth="advanced", include_answer=True, include_raw_content=True, include_images=False)
    result = tools.invoke(state['query'])

    state['retrieved_docs'] = [Document(page_content=d['content'], metadata={'url':d['url']}) for d in result]
    
    return state

def router_prompt(state):
    QUESTION_ROUTER_SYSTEM_PROMPT = """
    You are a question router. Given a question, determine if it is related to 'LLMs' or 'agents'. 
    If it is, respond with {{'route': 'vector_store'}}. Otherwise, respond with {{'route': 'web_search'}}.
    """

    question=state['query']

    question_router_prompt = ChatPromptTemplate.from_messages(
        [{"role": "system", "content": QUESTION_ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": "{question}"}]
    )

    prompt = question_router_prompt.format(question=question)
    response = json_mode_llm.invoke(prompt)
    return response.get('route')

def retrieval_grader_prompt(state):

    RETRIEVAL_GRADER_SYSTEM_PROMPT = """
    You are a grader assessing relevance of a retrieved document to a user question.
    Here is the retrieved document:

    <document>
    {documents}
    </document>

    Here is the user question:
    <question>
    {question}
    </question>

    If the document contains keywords related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.`;

    """

    question = state['query']
    documents = "\n".join([doc.page_content for doc in state["retrieved_docs"]])

    retrieval_grader_prompt = ChatPromptTemplate.from_messages(
        [{"role": "system", "content": RETRIEVAL_GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": "Question: {question}\nDocuments: {documents}"}]
    )

    prompt = retrieval_grader_prompt.format_prompt(
        question=question, documents=documents
    )
    response = json_mode_llm.invoke(prompt)
    state['retrieval_grade'] = json.loads(response.content)['score']
    return state

# def adapt_rag_system(state):
#     route = router_prompt(state)
#     if route == 'vector_store':
#         scores = retrieval_grader_prompt(state)
#         if max(scores) < 3:
#             route = 'web_search'
#         else:




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

def decide_retrieval(state):
    if state['retrieval_grade'] == 'no':
        return "web_search"

def generate_step(state):
    """Use retrieved docs to generate a response with the LLM."""
    # if state['retrieval_grade'] == 'no':
    #     #web_search(state)
    #     retrieved_docs = state["web_docs"]
    # else:
    retrieved_docs = state["retrieved_docs"]
    print(retrieved_docs)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    

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


# === Define LangGraph Workflow ===
workflow = StateGraph(RAGState)
workflow.add_node("preprocess", preprocess_step)
workflow.add_node("retrieve", retrieve_step)
workflow.add_node("generate", generate_step)
workflow.add_node("web_search", web_search)
workflow.add_node("decide_retrieval", decide_retrieval)
workflow.add_node("retrieval_grader",retrieval_grader_prompt)
workflow.set_entry_point("preprocess")
workflow.add_edge(START, "preprocess")
workflow.add_edge("preprocess", "retrieve")
workflow.add_edge("retrieve", "retrieval_grader")
workflow.add_conditional_edges("retrieval_grader", decide_retrieval)
workflow.add_edge("decide_retrieval", "generate")

# === Compile & Run ===
app = workflow.compile()

query = "when is trump birthday?"
result = app.invoke({"query": query})
print("RAG Response:", result["response"])
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.schema import Document
#from langchain.embeddings import OpenAIEmbeddings
from extract_from_zip import extract_from_zip
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg

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

# Initialize PGVector store
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="text_embeddings",
    connection=PG_CONNECTION_STRING,
    use_jsonb=True
)

zip_path = '/Users/rohit.khazanchi@teck.com/repo/da-hackathon-chatbot/embeddings/data/Confluence-export-teckresources.atlassian.net-DASA-csv.zip'
extract_path = '/Users/rohit.khazanchi@teck.com/repo/da-hackathon-chatbot/embeddings/data/extracted/DASA-csv'


#extracted_text = extract_from_zip(zip_path,extract_path )

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) 

# Function to create documents with metadata
# def create_documents_with_metadata(texts, metadata_column):
#     documents = []
#     for text, metadata in zip(texts, metadata_column):
#         # Split text into chunks
#         chunks = text_splitter.split_text(text)
#         for chunk in chunks:
#             # Create document with metadata
#             document = Document(page_content=chunk, metadata={'source': metadata})
#             documents.append(document)
#     return documents

# # Create documents with metadata from 'body_text' and 'metadata_column'
# for x in range(0, 7000, 1000):
#     print(x)
#     documents = create_documents_with_metadata(extracted_text['body'].tolist()[x:x+999], extracted_text['url'].tolist()[x:x+999])

# # Display the first 3 documents with metadata
#     for doc in documents[:3]:
#         print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}")

#     vector_store.add_documents(documents= documents)

# print("Embedding stored successfully!")



docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]

vector_store.add_documents(docs, ids=[doc.metadata['id'] for doc in docs])

print("Embedding stored successfully!")
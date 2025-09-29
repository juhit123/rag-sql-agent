# rag.py
import chromadb
from chromadb.utils import embedding_functions

# Initialize Chroma client (persistent storage)
client = chromadb.PersistentClient(path="./chroma_db")

# SentenceTransformer for embeddings
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def add_document(doc_id: str, text: str, metadata: dict):
    """Add a document to Chroma collection"""
    collection = client.get_or_create_collection(
        name="default", embedding_function=embedding_func
    )
    collection.add(documents=[text], metadatas=[metadata], ids=[doc_id])


def rag_query(question: str) -> str:
    """Query ChromaDB for relevant context with debug logging"""
    collection = client.get_or_create_collection(
        name="default", embedding_function=embedding_func
    )
    
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    # Debug: print everything we got back
    print("\n--- DEBUG: RAG Query ---")
    print("Question:", question)
    print("Raw Results:", results)
    print("------------------------\n")

    if results and results.get("documents"):
        docs = results["documents"][0]
        if docs:
            return " ".join(docs)
    return "No relevant documents found."


def list_documents():
    results = collection.get()
    docs = []
    for i in range(len(results["ids"])):
        docs.append({
            "id": results["ids"][i],
            "text": results["documents"][i],
            "metadata": results["metadatas"][i]
        })
    return docs

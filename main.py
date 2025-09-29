from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import uuid
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

chroma_client = chromadb.PersistentClient(path="chromadb_store")
openai_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=openai_ef
)

class AddTableRequest(BaseModel):
    table_name: str
    rows: list[dict]

class AddDocRequest(BaseModel):
    content: str

class RagRequest(BaseModel):
    question: str

class SQLRequest(BaseModel):
    question: str

class AskRequest(BaseModel):
    question: str

def ask_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

@app.post("/add_table")
def add_table(req: AddTableRequest):
    try:
        docs = [str(row) for row in req.rows]
        ids = [f"{req.table_name}_{uuid.uuid4()}" for _ in req.rows]
        collection.add(documents=docs, ids=ids, metadatas=[{"table": req.table_name}] * len(req.rows))
        return {"status": "success", "rows_added": len(req.rows)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/add_doc")
def add_doc(req: AddDocRequest):
    try:
        doc_id = str(uuid.uuid4())
        collection.add(documents=[req.content], ids=[doc_id], metadatas=[{"type": "doc"}])
        return {"status": "success", "id": doc_id}
    except Exception as e:
        return {"error": str(e)}

@app.get("/list_docs")
def list_docs():
    try:
        docs = collection.get()
        return {"documents": docs}
    except Exception as e:
        return {"error": str(e)}

@app.post("/rag")
def rag(req: RagRequest):
    try:
        results = collection.query(query_texts=[req.question], n_results=5)
        documents = results.get("documents", [[]])[0]

        if not documents:
            return {"answer": "No relevant documents found."}

        context = "\n".join(documents)
        prompt = f"""
        Use the following context to answer the question concisely.

        Context:
        {context}

        Question: {req.question}
        Answer:
        """
        answer = ask_gemini(prompt)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.post("/text_to_sql")
def text_to_sql(req: SQLRequest):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Convert this natural language question into a PostgreSQL SQL query.
        Table: employees(id, name, role, department)
        Question: {req.question}
        Only provide the SQL query, no explanations.
        """
        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        if sql_query.startswith("```"):
            sql_query = "\n".join(sql_query.split("\n")[1:-1]).strip()
        return {"sql_query": sql_query}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"AI error: {str(e)}")

@app.post("/ask")
def ask(req: AskRequest):
    try:
        answer = ask_gemini(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"AI error: {str(e)}")

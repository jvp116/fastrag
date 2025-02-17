import json

import chromadb
import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

PDF_PATH = "manual-da-igreja-2022.pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# modelo local
OLLAMA_MODEL = "llama2"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Inicializa o modelo de embeddings e ChromaDB
model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("manual_iasd")


def extract_text_from_pdf(pdf_path, chunk_size=500):
    """
    Extrai o texto do PDF e divide-o em blocos de aproximadamente 'chunk_size' caracteres
    para manter um bom contexto.
    """
    doc = fitz.open(pdf_path)
    sections = []
    buffer = ""
    
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            buffer += " " + text
            if len(buffer) >= chunk_size:
                sections.append(buffer.strip())
                buffer = ""
    
    if buffer:
        sections.append(buffer.strip())
        
    return sections


def index_pdf():
    """Indexa o conteúdo do PDF no banco vetorial (ChromaDB)"""
    sections = extract_text_from_pdf(PDF_PATH)
    for i, text in enumerate(sections):
        embedding = model.encode(text).tolist()
        collection.add(ids=[str(i)], embeddings=[embedding], metadatas=[{"content": text}])


# Se o banco estiver vazio, indexa o PDF
if collection.count() == 0:
    print("Indexando o manual...")
    index_pdf()
    print("Indexação concluída!")


app = FastAPI()

class QueryRequest(BaseModel):
    query: str


def generate_response(query, retrieved_text, model_name=OLLAMA_MODEL):
    """
    Gera uma resposta refinada utilizando um modelo local do Ollama.
    É necessário que o Ollama esteja rodando e o modelo especificado esteja disponível.
    """
    prompt = f"""You are a Seventh-day Adventist virtual assistant who specializes in providing clear, simple and objective answers based on the manual of the Seventh-day Adventist Church. Your role is to help users understand Adventist principles, practices and doctrines in a faithful manner and in line with the official teachings of the denomination. When answering, always use language that is accessible, direct and inspired by Adventist values, such as the importance of the Sabbath, holistic health, the second coming of Christ and the mission of the church. If the question is not directly related to the Adventist context, answer in a gentle way, bringing insights that reflect the Seventh-day Adventist vision. Avoid speculation or personal interpretations, always keeping in line with the official content of the church.
Answer the question, in portuguese (Brazil), based on the information below:

--- Relevant information taken from the Seventh-day Adventist Church Manual ---
{retrieved_text}

--- User question ---
{query}

    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.0,
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Erro ao chamar o modelo local do Ollama: {response.text}")

@app.post("/query")
def ask_question(request: QueryRequest):
    """
    Endpoint que recebe a pergunta do usuário, busca os trechos mais relevantes do manual
    e gera uma resposta refinada utilizando o modelo local do Ollama.
    """
    query_embedding = model.encode(request.query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if not results["metadatas"][0]:
        raise HTTPException(status_code=404, detail="Nenhuma informação relevante encontrada.")

    retrieved_text = "\n\n".join([res["content"] for res in results["metadatas"][0]])
    try:
        response_text = generate_response(request.query, retrieved_text)
        
        final_response = ""

        for line in response_text.split("\n"):
            if line.strip():  # Verifica se a linha não está vazia
                part = json.loads(line)
                final_response += part["response"]
                
                if part["done"]:
                    break
        
        print("Resposta final:", final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "resposta": final_response,
        "trechos_relevantes": retrieved_text
    }


# Para rodar a API:
# uvicorn fastrag:app --reload
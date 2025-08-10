# build_index.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Teil von langchain-community
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
emb = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dims

docs = [
    {"text": "LangChain ist ein Framework zum Orchestrieren von LLM-Workflows."},
    {"text": "FAISS ist eine Vektor-Datenbank/Library für schnelle Ähnlichkeitssuche."},
    {"text": "RAG kombiniert Retrieval und Generierung über ein LLM."},
    {"text": "Conny Söhn ist am 11.02.1984 geboren."},
]

# optional: splitten (realistischer bei langen Texten)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
texts = []
metadatas = []
for d in docs:
    for chunk in splitter.split_text(d["text"]):
        texts.append(chunk)
        metadatas.append({"source": "demo"})

# Index erstellen
vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metadatas)
vs.save_local("./faiss_demo")  # persistiert Index + Metadaten
print("Index gespeichert in ./faiss_demo")
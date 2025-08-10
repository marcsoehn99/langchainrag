# build_index_azure_ocr.py
import os
from pathlib import Path
from io import BytesIO

from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
if not endpoint or not key:
    raise SystemExit("Bitte AZURE_FORM_RECOGNIZER_ENDPOINT und AZURE_FORM_RECOGNIZER_KEY in .env setzen.")

client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

docs_dir = Path("./docs")
index_dir = Path("./waermepumpe_index")
index_dir.mkdir(exist_ok=True)

emb = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

all_chunks = []
for pdf_path in docs_dir.glob("*.pdf"):
    print(f"→ OCR: {pdf_path.name}")
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-read", document=f)
    result = poller.result()

    # Text pro Seite zusammenbauen
    pages_text = []
    for p in result.pages:
        lines = []
        for line in p.lines:
            lines.append(line.content)
        page_text = "\n".join(lines).strip()
        pages_text.append(page_text)

    has_text = any(t.strip() for t in pages_text)
    if not has_text:
        print(f"⚠️ Keine OCR-Texte extrahiert: {pdf_path.name}")
        continue

    # In LangChain Documents wandeln (mit Metadaten)
    page_docs = []
    for i, t in enumerate(pages_text):
        if not t.strip():
            continue
        page_docs.append(Document(page_content=t, metadata={"source_file": pdf_path.name, "page": i}))

    # Chunking
    chunks = splitter.split_documents(page_docs)
    all_chunks.extend(chunks)
    print(f"✓ {pdf_path.name}: {len(page_docs)} Seiten → {len(chunks)} Chunks")

if not all_chunks:
    raise SystemExit("Kein Text extrahiert – prüfe Endpoint/Key oder PDF-Qualität.")

vs = FAISS.from_documents(all_chunks, embedding=emb)
vs.save_local(str(index_dir))
print(f"✅ Index gespeichert in {index_dir.resolve()} | Chunks: {len(all_chunks)}")
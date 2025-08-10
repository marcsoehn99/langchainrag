# build_index.py
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
docs_dir = Path("./docs")
index_dir = Path("./waermepumpe_index")
index_dir.mkdir(exist_ok=True)

emb = OpenAIEmbeddings(model="text-embedding-3-small")
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

all_docs = []
for pdf_path in docs_dir.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()  # list[Document] mit page_content
    # Seiten ohne Text filtern (falls OCR fehlt)
    pages = [p for p in pages if (p.page_content or "").strip()]
    if not pages:
        print(f"⚠️ Kein Text extrahiert: {pdf_path} (evtl. ohne OCR?)")
        continue
    for d in pages:
        d.metadata = {**d.metadata, "source_file": pdf_path.name}
    chunks = splitter.split_documents(pages)
    all_docs.extend(chunks)
    print(f"✓ {pdf_path.name}: {len(pages)} Seiten → {len(chunks)} Chunks")

if not all_docs:
    raise SystemExit("Keine Texte gefunden. Prüfe, ob Lens wirklich OCR aktiviert hat.")

vs = FAISS.from_documents(all_docs, embedding=emb)
vs.save_local(str(index_dir))
print(f"✅ Index gespeichert nach {index_dir.resolve()} | Chunks: {len(all_docs)}")
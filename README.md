# Wärmepumpen-Assistent mit RAG (Retrieval-Augmented Generation)

Dieses Projekt implementiert einen **intelligenten Assistenten für Wärmepumpen** auf Basis von **RAG** und **GPT-5**.  
Anleitungen, Kurzanleitungen und weitere relevante Dokumente (z. B. Wartungsverträge, Inbetriebnahme-Anweisungen) werden in einem **FAISS-Vektorindex** gespeichert und dienen als kontextbezogene Wissensquelle.

---

## ✨ Features

- **GPT-5** als LLM für hochwertige, kontextreiche Antworten
- **Retrieval-Augmented Generation**:  
  Das LLM greift auf die im Index gespeicherten Dokumente zu, um Antworten zu formulieren
- **Azure Form Recognizer (prebuilt-read)** für OCR von gescannten PDFs (Bedienungsanleitung, Kurzanleitung etc.)
- **FAISS** als lokaler Vektorindex für schnelle Ähnlichkeitssuche
- **Tool-Calling-Agent**:  
  Das LLM kann gezielt das RAG-Tool aufrufen, um relevante Textpassagen zu finden und daraus eine Antwort zu generieren
- **Kontext-Injection**:  
  Gefundene Textpassagen werden als Kontext ins LLM-Prompt eingefügt, um Halluzinationen zu vermeiden

---

## 📂 Projektstruktur
.
├── docs/                          # Eingabedokumente (PDFs)
├── waermepumpe_index/              # Persistenter FAISS-Index
├── build_index_azure_ocr.py        # OCR + Indexaufbau mit Azure Form Recognizer
├── agent_wp.py                     # Wärmepumpen-Agent mit GPT-5 + RAG-Tool
├── .env                            # API-Keys (nicht ins Repo!)
└── requirements.txt                # Python-Abhängigkeiten
---

## 🚀 Installation & Setup

### 1. Repository klonen
```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

(.env)
OPENAI_API_KEY=sk-...
AZURE_FORM_RECOGNIZER_ENDPOINT=https://<dein-endpoint>.cognitiveservices.azure.com/
AZURE_FORM_RECOGNIZER_KEY=<dein-key>
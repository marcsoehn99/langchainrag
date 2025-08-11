# agent_wp.py
import os, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY fehlt in .env")

# --- Profil laden & formatieren ----------------------------------------------
def load_profile(path: str = "./profile.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"⚠️ Profil konnte nicht geladen werden: {e}")
        return {}

def format_profile(p: dict) -> str:
    if not p:
        return "kein Profil vorhanden"
    parts = []
    dev = p.get("device", {})
    inst = p.get("installation", {})
    ctrl = p.get("control", {})
    if dev.get("model"): parts.append(f"Modell: {dev['model']}")
    hcs = inst.get("heating_circuits")
    if isinstance(hcs, list) and hcs:
        hc_names = ", ".join([hc.get("name", f"HK{hc.get('id','?')}") for hc in hcs])
        parts.append(f"Heizkreise: {hc_names}")
    if ctrl.get("outdoor_temperature_sensor") is True:
        parts.append("Außentemperaturfühler: vorhanden")
    return " | ".join(parts)

PROFILE = load_profile()
PROFILE_TXT = format_profile(PROFILE)

def single_heating_circuit_name(p: dict):
    try:
        hcs = p.get("installation", {}).get("heating_circuits", [])
        if isinstance(hcs, list) and len(hcs) == 1:
            return hcs[0].get("name") or f"HK{hcs[0].get('id','1')}"
    except Exception:
        pass
    return None

SINGLE_HC_NAME = single_heating_circuit_name(PROFILE)
if PROFILE:
    print(f"ℹ️ Profil aktiv: {PROFILE_TXT}")

# --- Index laden --------------------------------------------------------------
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.load_local("./waermepumpe_index", embeddings=emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 4})

# --- EIN einziges LLM (deine Einstellung) ------------------------------------
# Hinweis: falls "gpt-5" in deiner Umgebung nicht verfügbar ist, auf "gpt-4o-mini" wechseln.
llm = ChatOpenAI(model="gpt-5", temperature=1)

# --- RAG-Prompt: Kontext + Profil + strikte Regeln ---------------------------
rag_prompt = ChatPromptTemplate.from_template(
    "Antworte NUR anhand des folgenden Kontexts UND des Installationsprofils. "
    "Stelle KEINE Rückfragen. Wenn die Antwort nicht sicher im Kontext steht, schreibe: "
    "'Im Handbuch konnte ich dazu keine Information finden.'\n\n"
    "Regeln:\n"
    "- Das Installationsprofil hat Vorrang vor widersprüchlichen Angaben im Kontext.\n"
    "- Wenn im Kontext mehrere Heizkreise (HK1/HK2) erwähnt sind, das Profil aber genau EINEN Heizkreis enthält, "
    "  antworte AUSSCHLIESSLICH für diesen Heizkreis und vermeide Formulierungen, die weitere Heizkreise implizieren.\n"
    "- Füge nach relevanten Aussagen Quellen in Klammern an: (Quelle: <Dateiname> S.<Seite>).\n\n"
    "Installationsprofil:\n{profile}\n\n"
    "Frage: {question}\n\n"
    "Kontext:\n{context}\n\n"
    "Antwort auf Deutsch, präzise und schrittweise:"
)
rag_chain = rag_prompt | llm | StrOutputParser()

def _format_ctx(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "unbekannt"))
        page = d.metadata.get("page", None)
        tag = f"{src}" + (f", S.{page+1}" if page is not None else "")
        content = (d.page_content or "").strip()
        if content:
            lines.append(f"- ({tag}) {content}")
    return "\n\n".join(lines)

def _rerank_by_profile(docs):
    """Penalisiere Chunks mit HK2/Heizkreis 2, wenn nur ein Heizkreis existiert."""
    if not SINGLE_HC_NAME:
        return docs
    def penalty(doc):
        txt = (doc.page_content or "").lower()
        return 1 if ("heizkreis 2" in txt or "hk2" in txt) else 0
    return sorted(docs, key=penalty)

@tool(return_direct=True)  # <-- Antwort geht direkt an den User
def ask_manual(question: str) -> str:
    """Beantworte eine Frage zur Wärmepumpe basierend auf den gescannten Anleitungen (RAG)."""
    docs = retriever.invoke(question)
    docs = _rerank_by_profile(docs)

    if not docs:
        return "Im Handbuch konnte ich dazu keine Information finden."

    context = _format_ctx(docs)
    if not context.strip():
        return "Im Handbuch konnte ich dazu keine Information finden."

    return rag_chain.invoke({
        "question": question,
        "context": context,
        "profile": PROFILE_TXT
    })

TOOLS = [ask_manual]

# --- Tool-calling Agent -------------------------------------------------------
agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein Assistent für Wärmepumpenfragen. Nutze Tools immer dann, wenn Fragen zur Wärmepumpe gestellt werden. "
     "Beachte das Installationsprofil; stelle KEINE Rückfragen. "
     "Wenn Informationen fehlen, antworte trotzdem anhand der Manuals und des Profils."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, TOOLS, agent_prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

if __name__ == "__main__":
    q = "lohnt es sich für mich smarte thermostate für meine heizungen zu kaufen mit meiner wärmepumpe?"
    out = executor.invoke({"input": q, "chat_history": []})
    print("\n--- Antwort ---\n", out["output"])
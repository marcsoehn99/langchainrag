# agent_wp.py
import os
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

# --- Index laden ---
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.load_local("./waermepumpe_index", embeddings=emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 4})

# --- LLMs ---
# kleines, günstiges Modell ist hier ok
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt für den RAG-Schritt (Antwort nur aus Kontext)
rag_prompt = ChatPromptTemplate.from_template(
    "Antworte NUR anhand des folgenden Kontexts. "
    "Wenn die Antwort nicht sicher im Kontext steht, sage das ehrlich.\n\n"
    "Frage: {question}\n\n"
    "Kontext:\n{context}\n\n"
    "Antwort auf Deutsch, präzise und schrittweise, mit Verweisen (Quelle: Dateiname/Seite, falls vorhanden):"
)
rag_chain = rag_prompt | llm | StrOutputParser()

def _format_ctx(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "unbekannt"))
        page = d.metadata.get("page", None)
        tag = f"{src}" + (f", S.{page+1}" if page is not None else "")
        lines.append(f"- ({tag}) {d.page_content.strip()}")
    return "\n\n".join(lines)

@tool
def ask_manual(question: str) -> str:
    """Beantworte eine Frage zur Wärmepumpe basierend auf den gescannten Anleitungen (RAG)."""
    docs = retriever.invoke(question)
    context = _format_ctx(docs)
    return rag_chain.invoke({"question": question, "context": context})

TOOLS = [ask_manual]

# --- Tool-calling Agent ---
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Assistent für Wärmepumpenfragen. Nutze Tools, wenn sie helfen."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, TOOLS, agent_prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

if __name__ == "__main__":
    q = "Wie aktiviere ich den Eco-Warmwasser-Modus?"
    out = executor.invoke({"input": q, "chat_history": []})
    print("\n--- Antwort ---\n", out["output"])
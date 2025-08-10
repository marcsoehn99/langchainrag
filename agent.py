from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
import os
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY fehlt in .env")


emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.load_local("./waermepumpe_index", embeddings=emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 3})



@tool
def multiply(a: float, b: float) -> float:
    """Multipliziert zwei Zahlen."""
    return a * b
@tool
def search_docs(query: str) -> str:
    """Durchsucht den FAISS-Index und liefert die Top-Treffer als Textliste."""
    docs = retriever.invoke(query)
    return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])

TOOLS = [multiply, search_docs]
llm = ChatOpenAI(model="gpt-5", temperature=1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein hilfreicher Assistent. Nutze Tools, wenn sie helfen."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # von create_tool_calling_agent genutzt, kein String-Problem
])

agent = create_tool_calling_agent(llm, TOOLS, prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

if __name__ == "__main__":
    out = executor.invoke({"input": "Wie setze ich Warmwasser auf ECO, nutze das suchtool", "chat_history": []})
    print(out["output"])
# app.py
import json
import asyncio
import textwrap
from typing import List, Literal, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.callbacks.base import BaseCallbackHandler

# dein Agent liegt im selben Ordner:
import agent_wp as agent

app = FastAPI(title="Wärmepumpen-Assistent API")

# CORS (locker eingestellt – bei Bedarf einschränken)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- REST API (klassisch) ----------
class ChatRequest(BaseModel):
    input: str
    chat_history: List[Tuple[Literal["human","ai"], str]] = []

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Synchroner Call (kein Live-Status)."""
    out = agent.executor.invoke(
        {"input": req.input, "chat_history": req.chat_history}
    )
    return {"output": out["output"]}

# ---------- thread-sicherer Callback-Handler ----------
class UiCallbackHandler(BaseCallbackHandler):
    """Thread-safe: pusht Events in eine asyncio.Queue über call_soon_threadsafe."""
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue

    def _emit(self, kind: str, payload: dict):
        # aus JEDEM Thread sicher in die asyncio-Queue pushen
        self.loop.call_soon_threadsafe(self.queue.put_nowait, {"type": kind, **payload})

    # Chains / Agent
    def on_chain_start(self, *_, **__):
        self._emit("status", {"message": "Entering AgentExecutor …"})

    def on_chain_end(self, *_, **__):
        self._emit("status", {"message": "Finished chain."})

    # Tools
    def on_tool_start(self, serialized, input_str, **__):
        name = (serialized or {}).get("name", "tool")
        self._emit("tool_start", {"name": name, "input": input_str})

    def on_tool_end(self, output, **__):
        snippet = textwrap.shorten(str(output).replace("\n", " "), width=220)
        self._emit("tool_end", {"output": snippet})

    # LLM
    def on_llm_start(self, *_, **__):
        self._emit("status", {"message": "Generating answer …"})

    def on_llm_end(self, *_, **__):
        self._emit("status", {"message": "LLM done."})

# ---------- SSE Streaming (Live-Status) ----------
@app.get("/api/chat_stream")
async def chat_stream(input: str, chat_history: str = "[]"):
    """
    Server‑Sent Events:
    - streamt Status und Tool-Events
    - sendet am Ende die finale Antwort (type='final')
    """
    try:
        history = json.loads(chat_history)
    except Exception:
        history = []

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = UiCallbackHandler(loop, queue)

    async def runner():
        # LangChain-Executor im Thread ausführen und Callbacks anreichen
        out = await asyncio.to_thread(
            lambda: agent.executor.invoke(
                {"input": input, "chat_history": history},
                config={"callbacks": [handler]},
            )
        )
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "final", "output": out["output"]})

    async def event_generator():
        task = asyncio.create_task(runner())
        try:
            while True:
                msg = await queue.get()
                yield {"event": "message", "data": json.dumps(msg)}
                if msg.get("type") == "final":
                    break
        finally:
            task.cancel()

    return EventSourceResponse(event_generator())

# ---------- Profil-Endpunkte (für UI) ----------
@app.get("/api/profile/summary")
def profile_summary():
    return {"summary": getattr(agent, "PROFILE_TXT", "kein Profil vorhanden")}

@app.get("/api/profile/full")
def profile_full():
    return {"profile": getattr(agent, "PROFILE", {})}

# ---------- Web (HTML) ----------
@app.get("/")
def index():
    return FileResponse("web/index.html")

# statische Assets (falls du später CSS/JS/Icons brauchst)
app.mount("/assets", StaticFiles(directory="web"), name="assets")
"""FastAPI backend for the Magic Item Shop RAG chatbot."""

import logging
import uuid

import chromadb
import litellm
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL = "vertex_ai/gemini-2.0-flash-lite"
CHROMA_PATH = "chroma_data"
COLLECTION_NAME = "magic_items"
TOP_K = 5

# --- ChromaDB setup (read-only at runtime) ---
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)
print(f"Loaded ChromaDB collection with {collection.count()} items")

# --- Session storage ---
sessions: dict[str, list[dict]] = {}

# --- FastAPI app ---
app = FastAPI()

SYSTEM_PROMPT_TEMPLATE = """\
<role>
You are the Keeper of Curious Things, proprietor of a magic item shop that exists
in the spaces between planes. Your shop has no fixed door, it simply appears when
someone needs it badly enough. You speak with quiet authority, as though you have
seen the rise and fall of empires and found them all mildly interesting.

You know every item in your collection intimately, where it was forged, what it
has done, and what it might yet do. You are helpful, but your helpfulness has an
unsettling quality, as though you know more about the customer's fate than you
let on. You occasionally reference things the customer hasn't told you yet.
</role>

<context>
{retrieved_items}
</context>

<instructions>
- Answer using ONLY the magic items provided in <context>.
- If no relevant items appear in context, say you'll have to check the back room,
  and suggest the customer describe what they need differently.
- Include item type, rarity, and attunement requirements when describing items.
- Keep responses concise but informative.
- Occasionally hint that you know why the customer really needs the item.
- Refer to items as though they are old acquaintances: "Ah, this one has been
  waiting for someone like you."
</instructions>
"""


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


def retrieve_items(query: str, top_k: int = TOP_K) -> str:
    """Query ChromaDB and return formatted item texts."""
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    logger.info("RAG query: %r", query)
    formatted = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        name = meta.get("name", "unknown")
        relevance = 1 - dist
        logger.info("  retrieved: %r (relevance=%.2f)", name, relevance)
        formatted.append(f"[relevance: {relevance:.2f}]\n{doc}")

    return "\n---\n".join(formatted)


def build_system_prompt(retrieved_items: str) -> dict:
    """Format the system prompt with retrieved context."""
    return {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(retrieved_items=retrieved_items),
    }


def generate_response(messages: list[dict]) -> str:
    """Call LiteLLM with the assembled messages."""
    response = litellm.completion(model=MODEL, messages=messages)
    return response.choices[0].message.content


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    # RAG retrieval based on the latest user message
    retrieved_items = retrieve_items(request.message)
    system_prompt = build_system_prompt(retrieved_items)

    # Assemble full message list
    user_message = {"role": "user", "content": request.message}
    messages = [system_prompt] + sessions[session_id] + [user_message]

    # Generate response
    assistant_text = generate_response(messages)

    # Store conversation turns (not the system prompt)
    sessions[session_id].append(user_message)
    sessions[session_id].append({"role": "assistant", "content": assistant_text})

    return ChatResponse(response=assistant_text, session_id=session_id)


@app.post("/clear")
async def clear(request: ChatRequest) -> dict:
    session_id = request.session_id
    if session_id and session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

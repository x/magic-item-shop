"""FastAPI backend for the Magic Item Shop RAG chatbot.

Architecture: tool-call-driven RAG
  Instead of always retrieving items before every response, we give the LLM
  a `search_magic_items` tool and let it decide when to use it. The /chat
  endpoint runs a simple tool-calling loop:

    1. Send user message → LLM (with tool definition)
    2. If LLM calls the tool → run ChromaDB retrieval → send result back → get final response
    3. If LLM responds directly → use that response as-is

  This means casual conversation ("How are you?") never touches ChromaDB,
  while inventory questions trigger focused, query-specific retrieval.
"""

import json
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
MODEL = "vertex_ai/gemini-2.5-flash-lite"
CHROMA_PATH = "chroma_data"
COLLECTION_NAME = "magic_items"
RETRIEVE_N = 10   # how many candidates to pull from ChromaDB
RERANK_TOP_K = 3  # how many to keep after re-ranking
MAX_TOOL_ITERATIONS = 3  # safety cap on the tool-calling loop

# --- ChromaDB setup (read-only at runtime) ---
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)
logger.info("Loaded ChromaDB collection with %d items", collection.count())

# --- Session storage ---
sessions: dict[str, list[dict]] = {}

# --- FastAPI app ---
app = FastAPI()

# --- Static system prompt (no template injection needed anymore) ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": """\
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

<instructions>
- When a customer asks about magic items, use the search_magic_items tool to look
  up relevant items in your inventory before responding.
- After searching, describe the retrieved items including type, rarity, and
  attunement requirements.
- Occasionally hint that you know why the customer really needs the item.
- Refer to items as though they are old acquaintances: "Ah, this one has been
  waiting for someone like you."
- For casual conversation or questions unrelated to inventory, respond directly
  without searching — no need to consult the shelves for small talk.
</instructions>
""",
}

# --- Tool definition (plain dict — no framework magic) ---
# This is the schema LiteLLM sends to the model so it knows what tools exist.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_magic_items",
            "description": (
                "Search the shop's inventory for magic items matching a description. "
                "Use this whenever a customer asks about items, their properties, or "
                "what you have available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural-language description of the items to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


def rerank(query: str, documents: list[str], top_k: int = RERANK_TOP_K) -> list[int]:
    """Re-rank candidate documents against a query using the LLM.

    Sends all candidates in a single LLM call and asks for a ranked list of
    indices.  Returns up to top_k indices in descending relevance order.
    """
    numbered = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(documents))
    prompt = (
        f"You are a relevance ranking assistant.\n\n"
        f"Query: {query}\n\n"
        f"Candidates:\n{numbered}\n\n"
        f"Return ONLY a JSON array of the candidate indices ordered from most to "
        f"least relevant to the query. Example: [2, 0, 1]. Include all indices."
    )
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    ranked_indices = json.loads(raw.strip())
    logger.info("Re-rank order: %s → keeping top %d", ranked_indices, top_k)
    return ranked_indices[:top_k]


def search_magic_items(query: str) -> str:
    """Execute a ChromaDB similarity search, re-rank, and return top results.

    1. Retrieve RETRIEVE_N candidates via vector similarity.
    2. Re-rank with the LLM to find the most relevant RERANK_TOP_K items.
    3. Return those items formatted for the model.
    """
    results = collection.query(query_texts=[query], n_results=RETRIEVE_N)
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Log initial retrieval
    for meta, dist in zip(metadatas, distances):
        logger.info("  retrieved: %r (relevance=%.2f)", meta.get("name", "?"), 1 - dist)

    # Re-rank and select top K
    top_indices = rerank(query, documents)

    formatted = []
    for idx in top_indices:
        doc = documents[idx]
        meta = metadatas[idx]
        dist = distances[idx]
        name = meta.get("name", "unknown")
        relevance = 1 - dist
        logger.info("  reranked top: %r (relevance=%.2f)", name, relevance)
        formatted.append(f"[relevance: {relevance:.2f}]\n{doc}")

    return "\n---\n".join(formatted)


# Maps tool names to their implementations.
# Adding a new tool = add the function + an entry here + a schema in TOOLS.
TOOL_FN_MAP = {
    "search_magic_items": search_magic_items,
}


def run_tool_calling_loop(messages: list[dict]) -> str:
    """Send messages to the LLM and handle tool calls until we get a final response.

    The loop:
      - Call the LLM with the current message list and tool definitions
      - If the model returns a tool_call: execute it, append the result, repeat
      - If the model returns a plain text response: return it

    A MAX_TOOL_ITERATIONS cap prevents runaway loops.
    """
    for iteration in range(MAX_TOOL_ITERATIONS):
        response = litellm.completion(model=MODEL, messages=messages, tools=TOOLS)
        choice = response.choices[0]

        # No tool call → plain response, we're done
        if choice.finish_reason != "tool_calls":
            logger.info("KEEPER: %s", choice.message.content)
            return choice.message.content

        # The model wants to call a tool — extract the call(s)
        tool_calls = choice.message.tool_calls

        # Append the assistant's tool-call message to the conversation
        # (the model's "I want to call this tool" turn must stay in history)
        messages.append(choice.message)

        # Execute each requested tool call and append results
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            kwargs_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in fn_args.items())
            logger.info("TOOL_CALL: %s(%s)", fn_name, kwargs_str)

            fn = TOOL_FN_MAP.get(fn_name)
            result = fn(**fn_args) if fn else f"Unknown tool: {fn_name}"

            logger.info("TOOL_RESPONSE: %s", result)

            # Tool result message — role "tool", tied to the call by tool_call_id
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    # Fell through the loop — ask for a final answer without tools as a fallback
    logger.warning("Hit MAX_TOOL_ITERATIONS (%d), forcing final response", MAX_TOOL_ITERATIONS)
    response = litellm.completion(model=MODEL, messages=messages)
    final_text = response.choices[0].message.content
    logger.info("KEEPER: %s", final_text)
    return final_text


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    user_message = {"role": "user", "content": request.message}
    logger.info("USER: %s", request.message)

    # Assemble: static system prompt + conversation history + new user message
    messages = [SYSTEM_PROMPT] + sessions[session_id] + [user_message]

    # Run the tool-calling loop — may invoke search_magic_items zero or more times
    assistant_text = run_tool_calling_loop(messages)

    # Persist only the clean user/assistant turns in session history
    # (tool call/result messages are ephemeral — they don't need to carry forward)
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

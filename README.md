# Magic Item Shop

Teaching example for **IEOR 4576**.

This app introduces **Retrieval Augmented Generation (RAG)** using a D&D magic item shop as the scenario.

A shopkeeper character (powered by Gemini via LiteLLM) decides when to search its inventory, 1,031 magic items embedded in ChromaDB, and responds only with what it finds.

The key ideas demonstrated:
- **Offline indexing**: embed a corpus once, persist it to disk (`index_items.py`)
- **Tool-call-driven retrieval**: the LLM invokes `search_magic_items()` when it decides to, not on every message
- **Manual tool-calling loop**: explicit `TOOL_FN_MAP` dispatch with no agent framework

What we skip:
- Chunking
- Anything fancy

## Setup

```bash
uv sync                          # install dependencies
uv run python index_items.py     # build chroma_data/ (one-time)
uv run python app.py             # start server → http://localhost:8000
```

## Docker

Indexing runs at build time. The image is fully self-contained:

```bash
docker build -t magic-item-shop .
docker run -p 8000:8000 magic-item-shop
```



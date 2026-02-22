"""Index magic item markdown files into ChromaDB for RAG retrieval."""

# What this script does:
#   1. Reads all .md files from the magic_items/ directory
#   2. Parses each file for the item name and metadata (type, rarity, attunement)
#   3. Embeds the full markdown text using ChromaDB's built-in embedding model
#   4. Stores the embeddings + metadata in a persistent ChromaDB collection
#
# This is a one-time offline step — run it once before starting the server.
# The output (chroma_data/) is a build artifact: gitignored locally, baked
# into the Docker image at build time.
#
# At query time, app.py opens the same chroma_data/ directory (read-only)
# and calls collection.query() to find items similar to the user's message.

import os
import re
import chromadb

ITEMS_DIR = "magic_items"
CHROMA_PATH = "chroma_data"
COLLECTION_NAME = "magic_items"

# Patterns for parsing markdown headers and metadata
H1_PATTERN = re.compile(r"^#\s+\[?([^\]\n]+)\]?")
TAG_PATTERN = re.compile(r"#(\w+)")


def parse_item(filepath: str) -> dict | None:
    """Parse a magic item markdown file into id, document, and metadata."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Skip empty files (a few exist in the dataset)
    if not text.strip():
        return None

    # Use the filename stem as a stable unique ID (e.g. "Flame Tongue")
    filename_stem = os.path.splitext(os.path.basename(filepath))[0]
    lines = text.strip().split("\n")

    # --- Parse item name from the H1 header ---
    # Most files open with:  # [Flame Tongue](https://...)
    # Some files have no H1 (e.g. Amethyst Lodestone) — fall back to filename.
    name = filename_stem
    for line in lines:
        m = H1_PATTERN.match(line.strip())
        if m:
            # The regex captures everything after "# ", including the link text.
            # If the name is a markdown link like "[Flame Tongue](url)", split
            # on "](" to extract just the display text.
            raw = m.group(1)
            name = raw.split("](")[0].strip()
            break

    # --- Parse metadata from the tag line ---
    # The second or third line of each file contains hashtag-style metadata, e.g.:
    #   #Weapon *item (sword),* #Rare *(requires attunement)*
    # We scan the first 5 lines to find this tag line, then extract:
    #   - item_type: the equipment category (#Weapon, #Armor, #Wondrous, etc.)
    #   - rarity:    the power tier (#Common, #Uncommon, #Rare, #VeryRare, etc.)
    #   - attunement: whether the item requires attunement (True/False)
    item_type = ""
    rarity = ""
    attunement = False

    for line in lines[:5]:
        tags = TAG_PATTERN.findall(line)
        if tags:
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower in ("weapon", "armor", "wondrous", "potion", "ring",
                                 "rod", "scroll", "staff", "wand"):
                    item_type = tag
                elif tag_lower in ("common", "uncommon", "rare", "veryrare",
                                   "legendary", "artifact"):
                    rarity = tag
            if "attunement" in line.lower():
                attunement = True
            break  # stop after the first tag line

    return {
        "id": filename_stem,
        "document": text,       # full markdown text — this is what gets embedded
        "metadata": {           # structured fields — filterable at query time
            "name": name,
            "item_type": item_type,
            "rarity": rarity,
            "attunement": attunement,
        },
    }


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection uses ChromaDB's default embedding function:
    # all-MiniLM-L6-v2 (via sentence-transformers), which runs locally with
    # no API key or network call required. It's fast and good enough for demos.
    #
    # In production you'd likely use your LLM provider's embedding model instead
    # (e.g. Vertex AI text-embedding-005) for better quality and consistency.
    #
    # IMPORTANT: whatever embedding model you use at index time MUST be the same
    # model used at query time — otherwise the similarity scores are meaningless.
    collection = client.get_or_create_collection(COLLECTION_NAME)

    md_files = sorted(
        os.path.join(ITEMS_DIR, f)
        for f in os.listdir(ITEMS_DIR)
        if f.endswith(".md")
    )

    ids = []
    documents = []
    metadatas = []
    skipped = 0

    for filepath in md_files:
        item = parse_item(filepath)
        if item is None:
            skipped += 1
            continue
        ids.append(item["id"])
        documents.append(item["document"])
        metadatas.append(item["metadata"])

    # Upsert in batches of 200.
    #
    # Why upsert instead of add? upsert() is idempotent — re-running this script
    # won't fail if the items are already in the collection. add() would raise a
    # duplicate ID error on the second run.
    #
    # Why batch? Some vector databases have a maximum request size, so batching
    # is good practice even when the DB handles it internally (as ChromaDB does).
    BATCH_SIZE = 200
    for i in range(0, len(ids), BATCH_SIZE):
        collection.upsert(
            ids=ids[i : i + BATCH_SIZE],
            documents=documents[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    print(f"Indexed {len(ids)} items (skipped {skipped} empty files)")


if __name__ == "__main__":
    main()

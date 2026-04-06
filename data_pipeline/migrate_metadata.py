"""
migrate_metadata.py
-------------------
Fixes records in Milvus that were ingested with an empty metadata field because
the Excel column was named "Metadata" (capital M) instead of "metadata".

Strategy (no re-embedding):
  1. Read the source Excel file to build a text_content → metadata mapping.
  2. Fetch all entities from Milvus (including their stored vectors).
  3. For each entity, look up the correct metadata from the Excel mapping.
  4. Delete all existing entities.
  5. Re-insert with the same vectors + corrected metadata.

Usage:
    python data_pipeline/migrate_metadata.py <path_to_excel_file>
"""

import sys
import os
import json
import pandas as pd
import structlog
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection, utility
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
log = structlog.get_logger()

BATCH_SIZE = 1000  # Milvus query pagination batch size


def connect_milvus():
    if settings.MILVUS_URI.startswith("https"):
        log.info("Connecting to Zilliz Cloud...", uri=settings.MILVUS_URI)
        connections.connect(alias="default", uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
    else:
        log.info("Connecting to local Milvus...", uri=settings.MILVUS_URI)
        connections.connect(alias="default", uri=settings.MILVUS_URI)


def build_metadata_map(excel_path: str) -> dict:
    """Returns a dict mapping text_content -> metadata dict from the Excel file."""
    df = pd.read_excel(excel_path)

    # Case-insensitive lookup for metadata column
    meta_col = next((c for c in df.columns if c.lower() == "metadata"), None)
    if not meta_col:
        log.warning("No metadata column found in Excel. Migration will set all metadata to {}.")

    text_col = next((c for c in df.columns if c.lower() == "text_content"), None)
    if not text_col:
        log.error("No text_content column found in Excel.", columns=list(df.columns))
        sys.exit(1)

    log.info("Excel columns detected", text_col=text_col, meta_col=meta_col)

    mapping = {}
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        meta = {}
        if meta_col and pd.notna(row[meta_col]):
            try:
                val = row[meta_col]
                if isinstance(val, str):
                    meta = json.loads(val)
                elif isinstance(val, dict):
                    meta = val
            except Exception:
                meta = {"raw": str(row[meta_col])}
        mapping[text] = meta

    log.info("Metadata map built", total_rows=len(mapping))
    return mapping


def fetch_all_entities(collection: Collection) -> list[dict]:
    """Fetches all entities from Milvus in batches using query pagination."""
    collection.load()
    all_entities = []
    offset = 0

    while True:
        results = collection.query(
            expr="pk >= 0",
            output_fields=["pk", "text_content", "source_type", "metadata", "vector"],
            limit=BATCH_SIZE,
            offset=offset,
        )
        if not results:
            break
        all_entities.extend(results)
        log.info("Fetched batch", offset=offset, batch_size=len(results), total_so_far=len(all_entities))
        if len(results) < BATCH_SIZE:
            break
        offset += BATCH_SIZE

    log.info("Total entities fetched from Milvus", count=len(all_entities))
    return all_entities


def migrate(excel_path: str):
    if not os.path.exists(excel_path):
        log.error("Excel file not found", path=excel_path)
        sys.exit(1)

    metadata_map = build_metadata_map(excel_path)

    connect_milvus()

    collection_name = settings.MILVUS_COLLECTION_NAME
    if not utility.has_collection(collection_name):
        log.error("Collection not found", collection=collection_name)
        sys.exit(1)

    collection = Collection(collection_name)

    # Step 1: Fetch all existing entities (with their vectors)
    log.info("Fetching all entities from Milvus...")
    entities = fetch_all_entities(collection)

    if not entities:
        log.warning("No entities found in collection. Nothing to migrate.")
        return

    # Step 2: Build corrected data
    matched = 0
    unmatched = 0
    corrected = []

    for entity in entities:
        text = (entity.get("text_content") or "").strip()
        correct_meta = metadata_map.get(text)

        if correct_meta is None:
            unmatched += 1
            correct_meta = entity.get("metadata") or {}  # keep existing (empty) metadata
        else:
            matched += 1

        corrected.append({
            "vector": entity["vector"],
            "source_type": entity.get("source_type", ""),
            "text_content": text,
            "metadata": correct_meta,
        })

    log.info("Match summary", matched=matched, unmatched=unmatched)

    if unmatched > 0:
        log.warning(
            "Some Milvus records had no matching row in Excel. "
            "Their metadata will remain unchanged.",
            unmatched=unmatched,
        )

    # Step 3: Delete all existing entities
    log.info("Deleting all existing entities...")
    ids_to_delete = [e["pk"] for e in entities]
    # Milvus delete requires an expression; delete by PKs in chunks
    chunk_size = 500
    for i in range(0, len(ids_to_delete), chunk_size):
        chunk = ids_to_delete[i : i + chunk_size]
        expr = f"pk in [{', '.join(str(x) for x in chunk)}]"
        collection.delete(expr)
    collection.flush()
    log.info("All entities deleted.")

    # Step 4: Re-insert with corrected metadata
    log.info("Re-inserting entities with corrected metadata...", count=len(corrected))
    vectors     = [r["vector"]      for r in corrected]
    sources     = [r["source_type"] for r in corrected]
    texts       = [r["text_content"]for r in corrected]
    metas       = [r["metadata"]    for r in corrected]

    entities_to_insert = [vectors, sources, texts, metas]
    collection.insert(entities_to_insert)
    collection.flush()

    log.info("Migration complete.", total_reinserted=len(corrected))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline/migrate_metadata.py <path_to_excel_file_or_directory>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        excel_files = []
        for root, _, files in os.walk(target):
            for f in files:
                if f.endswith(".xlsx") or f.endswith(".xls"):
                    excel_files.append(os.path.join(root, f))
        if not excel_files:
            print(f"No Excel files found in {target}")
            sys.exit(1)
        print(f"Found {len(excel_files)} Excel files. Starting migration...")
        for i, path in enumerate(excel_files, 1):
            print(f"\n[{i}/{len(excel_files)}] Migrating: {path}")
            migrate(path)
        print("\nAll files migrated.")
    else:
        migrate(target)

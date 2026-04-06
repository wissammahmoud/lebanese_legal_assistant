"""
transfer_to_zilliz.py
----------------------
Transfers all entities from a local Milvus collection to Zilliz Cloud,
reusing existing vectors (no re-embedding cost).

Usage:
    python data_pipeline/transfer_to_zilliz.py \
        --zilliz-uri  https://your-endpoint.zillizcloud.com \
        --zilliz-token your_api_key

The local Milvus URI and collection name are read from app.core.config (settings).
"""

import sys
import os
import argparse
import structlog
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection, utility
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
log = structlog.get_logger()

BATCH_SIZE = 500   # entities per query/insert batch
PK_FIELD   = "pk"


def fetch_all_from_local(collection: Collection) -> list[dict]:
    collection.load()
    all_entities = []
    offset = 0

    log.info("Fetching all entities from local Milvus...")
    while True:
        batch = collection.query(
            expr=f"{PK_FIELD} >= 0",
            output_fields=[PK_FIELD, "text_content", "source_type", "metadata", "vector"],
            limit=BATCH_SIZE,
            offset=offset,
        )
        if not batch:
            break
        all_entities.extend(batch)
        log.info("Fetched batch", offset=offset, batch_size=len(batch), total=len(all_entities))
        if len(batch) < BATCH_SIZE:
            break
        offset += BATCH_SIZE

    log.info("Fetch complete", total_entities=len(all_entities))
    return all_entities


def insert_to_zilliz(collection: Collection, entities: list[dict]):
    total = len(entities)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = entities[i : i + BATCH_SIZE]

        vectors = [e["vector"]       for e in batch]
        sources = [e["source_type"]  for e in batch]
        texts   = [e["text_content"] for e in batch]
        metas   = [e["metadata"] or {} for e in batch]

        collection.insert([vectors, sources, texts, metas])
        inserted += len(batch)
        log.info("Inserted batch", inserted=inserted, total=total)

    collection.flush()
    log.info("All entities inserted and flushed to Zilliz.", total=inserted)


def main():
    parser = argparse.ArgumentParser(description="Transfer local Milvus → Zilliz Cloud")
    parser.add_argument("--zilliz-uri",   required=True, help="Zilliz Cloud cluster endpoint")
    parser.add_argument("--zilliz-token", required=True, help="Zilliz Cloud API key")
    parser.add_argument("--collection",   default=settings.MILVUS_COLLECTION_NAME,
                        help=f"Collection name (default: {settings.MILVUS_COLLECTION_NAME})")
    args = parser.parse_args()

    collection_name = args.collection

    # ── Connect to local Milvus ──────────────────────────────────────────────
    local_uri = settings.MILVUS_URI
    if local_uri.startswith("https"):
        log.error(
            "MILVUS_URI in settings points to a cloud endpoint, not local Milvus. "
            "Set MILVUS_URI=http://localhost:19530 in your .env before running this script."
        )
        sys.exit(1)

    log.info("Connecting to local Milvus...", uri=local_uri)
    connections.connect(alias="local", uri=local_uri)

    if not utility.has_collection(collection_name, using="local"):
        log.error("Collection not found in local Milvus", collection=collection_name)
        sys.exit(1)

    local_col = Collection(collection_name, using="local")
    entities  = fetch_all_from_local(local_col)

    if not entities:
        log.warning("No entities found in local collection. Nothing to transfer.")
        return

    # ── Connect to Zilliz Cloud ──────────────────────────────────────────────
    log.info("Connecting to Zilliz Cloud...", uri=args.zilliz_uri)
    connections.connect(alias="zilliz", uri=args.zilliz_uri, token=args.zilliz_token)

    if not utility.has_collection(collection_name, using="zilliz"):
        log.error(
            "Collection not found on Zilliz Cloud. "
            "Run milvus_setup.py with Zilliz credentials first.",
            collection=collection_name,
        )
        sys.exit(1)

    zilliz_col = Collection(collection_name, using="zilliz")

    # ── Transfer ─────────────────────────────────────────────────────────────
    insert_to_zilliz(zilliz_col, entities)
    log.info("Transfer complete.")


if __name__ == "__main__":
    main()

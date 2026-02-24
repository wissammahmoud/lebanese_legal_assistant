import sys
import os
import pandas as pd
from openai import OpenAI
import json
import structlog
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection, utility
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
log = structlog.get_logger()

def get_embedding(client, text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def ingest_data(excel_path: str):
    if not os.path.exists(excel_path):
        log.error("File not found", path=excel_path)
        return

    log.info("Reading Excel file...", path=excel_path)
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        log.error("Failed to read Excel file", error=str(e))
        return
    
    # Check required columns
    required_cols = ["source_type", "text_content"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.error("Missing required columns", missing=missing)
        return

    # Init OpenAI
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
    except Exception as e:
        log.error("Failed to initialize OpenAI client", error=str(e))
        return

    # Connect to Milvus
    log.info("Connecting to Milvus...", uri=settings.MILVUS_URI)
    try:
        connections.connect(uri=settings.MILVUS_URI)
    except Exception as e:
        log.error("Failed to connect to Milvus", error=str(e))
        return

    collection_name = settings.MILVUS_COLLECTION_NAME
    if not utility.has_collection(collection_name):
        log.error(f"Collection {collection_name} does not exist. Please run milvus_setup.py first.")
        return

    collection = Collection(collection_name)

    data_rows = []
    
    log.info(f"Processing {len(df)} rows...")
    
    for index, row in df.iterrows():
        try:
            text = str(row["text_content"])
            source = str(row["source_type"])
            
            # Simple Metadata handling
            meta = {}
            if "metadata" in df.columns and pd.notna(row["metadata"]):
                try:
                    val = row["metadata"]
                    if isinstance(val, str):
                        meta = json.loads(val)
                    elif isinstance(val, dict):
                        meta = val
                except:
                    meta = {"raw": str(row["metadata"])}
            
            vector = get_embedding(client, text)
            
            data_rows.append([
                vector,
                source,
                text,
                meta
            ])
            
            if (index + 1) % 10 == 0:
                log.info(f"Processed {index + 1} rows")

        except Exception as e:
            log.warning(f"Error processing row {index}", error=str(e))

    if data_rows:
        log.info(f"Inserting {len(data_rows)} vectors into Milvus...")
        # PyMilvus insert expects list of columns, not list of rows for some versions, 
        # but list of dicts or list of lists depending on API usage.
        # Collection.insert usually takes: [[vector1, vector2], [source1, source2], ...] 
        # Or a list of dicts if using DataFrame? 
        # The safest way is column-based list of lists.
        
        vectors = [r[0] for r in data_rows]
        sources = [r[1] for r in data_rows]
        texts = [r[2] for r in data_rows]
        metas = [r[3] for r in data_rows]
        
        entities = [
            vectors,
            sources,
            texts,
            metas
        ]
        
        try:
            collection.insert(entities)
            collection.flush()
            log.info("Ingestion complete successfully.")
        except Exception as e:
            log.error("Failed to insert data", error=str(e))

    else:
        log.info("No valid data to insert.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_data.py <path_to_excel>")
    else:
        ingest_data(sys.argv[1])

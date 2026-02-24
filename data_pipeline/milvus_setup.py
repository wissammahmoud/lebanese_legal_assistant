import sys
import os

# Add project root to sys.path to access app.core.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from app.core.config import settings
import structlog
import logging

# Configure basic logging if structlog isn't fully setup via app
logging.basicConfig(level=logging.INFO)
log = structlog.get_logger()

def create_collection():
    log.info("Connecting to Milvus...", uri=settings.MILVUS_URI)
    try:
        connections.connect(uri=settings.MILVUS_URI)
    except Exception as e:
        log.error("Failed to connect to Milvus", error=str(e))
        return

    collection_name = settings.MILVUS_COLLECTION_NAME
    dim = settings.MILVUS_DIMENSION

    if utility.has_collection(collection_name):
        log.info(f"Collection {collection_name} already exists. Skipping creation.")
        return

    log.info(f"Creating collection {collection_name} with dim={dim}")

    # Schema Definition based on User Request
    fields = [
        # pk: Int64, Primary Key, Auto ID
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True, description="Unique identifier"),
        
        # vector: FloatVector, HNSW
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim, description="Embedding vector"),
        
        # source_type: VarChar, Partition Key
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True, description="Source type (e.g., ruling, article)"),
        
        # text_content: VarChar
        FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65535, description="The actual text content"),
        
        # metadata: JSON
        FieldSchema(name="metadata", dtype=DataType.JSON, description="Additional metadata")
    ]

    schema = CollectionSchema(fields, description="Lebanese Legal Assistant Knowledge Base")

    collection = Collection(name=collection_name, schema=schema)
    log.info("Collection created.")

    # Index Creation
    log.info("Creating index...")
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    
    collection.create_index(field_name="vector", index_params=index_params)
    log.info("Index created successfully.")
    
    # Load collection to memory
    collection.load()
    log.info("Collection loaded.")

if __name__ == "__main__":
    create_collection()

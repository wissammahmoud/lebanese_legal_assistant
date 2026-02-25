import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, utility, Collection
from app.core.config import settings

def check_zilliz():
    print(f"Connecting to: {settings.MILVUS_URI}")
    try:
        if settings.MILVUS_URI.startswith("https"):
            connections.connect(
                alias="default",
                uri=settings.MILVUS_URI,
                token=settings.MILVUS_TOKEN
            )
        else:
            connections.connect(alias="default", uri=settings.MILVUS_URI)
        
        print("✅ Connection successful!")
        
        collection_name = settings.MILVUS_COLLECTION_NAME
        if utility.has_collection(collection_name):
            print(f"✅ Collection '{collection_name}' exists.")
            coll = Collection(collection_name)
            coll.load()
            print(f"   Entities count: {coll.num_entities}")
        else:
            print(f"❌ Collection '{collection_name}' NOT found.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_zilliz()

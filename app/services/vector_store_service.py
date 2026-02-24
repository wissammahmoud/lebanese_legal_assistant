import asyncio
import structlog
import pybreaker
from pymilvus import connections, Collection, utility
from app.core.config import settings
from langsmith import traceable

log = structlog.get_logger()

# Circuit Breaker: Trip after 3 failures, reset after 60s
# This protects the system from cascading failures if Milvus is down.
db_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60)

class VectorStoreService:
    def __init__(self):
        self._collection = None

    def _connect(self):
        """Ensures a connection to Milvus exists."""
        if not connections.has_connection("default"):
            connections.connect(uri=settings.MILVUS_URI)
        
        if not self._collection:
            if utility.has_collection(settings.MILVUS_COLLECTION_NAME):
                self._collection = Collection(settings.MILVUS_COLLECTION_NAME)
                self._collection.load()
            else:
                log.error(f"Collection {settings.MILVUS_COLLECTION_NAME} not found.")
                raise Exception("Collection not found")

    @db_breaker
    @traceable(run_type="retriever", name="Milvus Vector Search")
    async def search(self, vector: list[float], limit: int = 3) -> list[dict]:
        """
        Performs a vector search. Wrapped in a circuit breaker.
        Runs the synchronous Milvus call in a separate thread.
        """
        try:
            return await asyncio.to_thread(self._search_sync, vector, limit)
        except pybreaker.CircuitBreakerError:
            log.error("Circuit breaker passed: Milvus is unavailable.")
            raise
        except Exception as e:
            log.error("Vector search failed", error=str(e))
            raise

    def _search_sync(self, vector: list[float], limit: int) -> list[dict]:
        self._connect()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 10},
        }
        
        results = self._collection.search(
            data=[vector],
            anns_field="vector",
            param=search_params,
            limit=limit,
            output_fields=["text_content", "source_type", "metadata"]
        )
        
        hits_data = []
        for hits in results:
            for hit in hits:
                hits_data.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text_content"),
                    "source": hit.entity.get("source_type"),
                    "metadata": hit.entity.get("metadata")
                })
        return hits_data

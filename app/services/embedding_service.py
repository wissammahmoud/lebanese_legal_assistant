import hashlib
import json
import structlog
import redis.asyncio as redis
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from langsmith import traceable

log = structlog.get_logger()

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @traceable(run_type="embedding", name="OpenAI Embedding")
    async def get_embedding(self, text: str) -> list[float]:
        text_hash = self._get_hash(text)
        cache_key = f"embedding:{text_hash}"
        
        # Check Cache
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                log.info("Embedding cache hit", text_hash=text_hash[:8])
                return json.loads(cached)
        except Exception as e:
            log.warning("Redis cache read error", error=str(e))

        # Call OpenAI with retry logic
        try:
            embedding = await self._call_openai(text)
            
            # Save to Cache (TTL 24h)
            try:
                await self.redis.setex(cache_key, 86400, json.dumps(embedding))
            except Exception as e:
                log.warning("Redis cache write error", error=str(e))
                
            return embedding
        except Exception as e:
            log.error("Embedding generation failed after retries", error=str(e))
            raise e

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _call_openai(self, text: str) -> list[float]:
        log.info("Calling OpenAI for embedding")
        text = text.replace("\n", " ")
        response = await self.client.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding

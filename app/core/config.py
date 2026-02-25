from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Lebanese Legal Assistant Microservice"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    
    SERVICE_API_KEY: str

    OPENAI_API_KEY: str
    
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_TOKEN: Optional[str] = None
    MILVUS_COLLECTION_NAME: str = "lebanese_laws"
    MILVUS_DIMENSION: int = 1536
    
    REDIS_URL: str = "redis://localhost:6379/0"
    
    OTEL_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "lebanese-legal-assistant"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    
    # LangSmith Tracing
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "lebanese-legal-assistant"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()

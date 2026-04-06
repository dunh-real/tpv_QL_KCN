from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    LLM_BASE_URL:str
    LLM_MODEL_LLAMA:str
    LLM_MODEL_QWEN35:str
    LLM_MODEL_QWEN3:str
    LLM_MODEL_QWEN25:str
    LLM_MODEL_GEMMA:str
    LLM_API_KEY:str
    LLM_MAX_TOKENS: int = 1204
    OLLAMA_BASE_URL: str


    HYPERSPACE_HOST:str
    HYPERSPACE_COLLECTION_NAME:str  
    HYPERSPACE_API_KEY:str
    SQL_COLLECTION_NAME:str

    VECTOR_DIMENSION:int


    QDRANT_HOST:str
    QDRANT_PORT:int
    QDRANT_COLLECTION_NAME:str
    QDRANT_API_KEY:str

    EMBEDDING_MODEL: str
    OCR_MODEL: str

    REDIS_HOST: str 
    REDIS_PORT: int 
    REDIS_DB: int

    MONGO_URL:str
    MONGODB_NAME:str
    MONGODB_COLLECTION_NAME:str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    LOCAL_STORAGE_DIR: str = "local_storage"
    
    model_config=SettingsConfigDict(env_file=".env",env_file_encoding="utf-8",extra="ignore")
settings = Settings()

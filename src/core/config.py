from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    LLM_BASE_URL:str
    LLM_MODEL_LLAMA:str
    LLM_MODEL_QWEN35:str
    LLM_MODEL_QWEN3:str
    LLM_MODEL_QWEN25:str
    LLM_MODEL_GEMMA:str
    LLM_API_KEY:str


    HYPERSPACE_HOST:str
    HYPERSPACE_COLLECTION_NAME:str  
    HYPERSPACE_API_KEY:str
    HYPERSPACE_VECTOR_DIMENSION:int


    EMBEDDING_MODEL: str
    REDIS_HOST: str 
    REDIS_PORT: int 
    REDIS_DB: int
    
    model_config=SettingsConfigDict(env_file=".env",env_file_encoding="utf-8",extra="ignore")
settings = Settings()

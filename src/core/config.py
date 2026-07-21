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

    SQL_COLLECTION_NAME:str

    VECTOR_DIMENSION:int

    QDRANT_HOST:str
    QDRANT_PORT:int
    QDRANT_COLLECTION_NAME:str
    QDRANT_API_KEY:str

    EMBEDDING_MODEL: str
    OCR_MODEL: str
    RERANK_MODEL:str

    REDIS_HOST: str 
    REDIS_PORT: int 
    REDIS_DB: int

    MONGO_URL:str
    MONGODB_NAME:str
    MONGODB_COLLECTION_NAME:str

    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_DB: str

    MSSQL_USER: str = "sa"
    MSSQL_PASSWORD: str = "password"
    MSSQL_HOST: str = "localhost"
    MSSQL_PORT: int = 1433
    MSSQL_DB: str = "db"

    MINIO_ENDPOINT:str
    MINIO_ACCESS_KEY:str
    MINIO_SECRET_KEY:str
    MINIO_BUCKET:str
    MINIO_SECURE:bool

    RABBITMQ_URL:str

    # Security configs
    API_KEYS: str = ""
    RATE_LIMIT_PER_MINUTE: int = 20

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def MYSQL_DATABASE_URL(self) -> str:
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"

    @property
    def MSSQL_DATABASE_URL(self) -> str:
        return f"mssql+pymssql://{self.MSSQL_USER}:{self.MSSQL_PASSWORD}@{self.MSSQL_HOST}:{self.MSSQL_PORT}/{self.MSSQL_DB}"
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    LOCAL_STORAGE_DIR: str = "local_storage"
    
    SCHEMA_CACHE_TTL_SECONDS: int = 300
    CACHE_SCORE_THRESHOLD: float = 0.9
    CACHE_TOP_K: int = 1

    model_config=SettingsConfigDict(env_file=".env",env_file_encoding="utf-8",extra="ignore")
settings = Settings()

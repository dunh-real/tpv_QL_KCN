from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LLM_BASE_URL:str
    LLM_MODEL_LLAMA:str
    LLM_MODEL_QWEN35:str
    LLM_MODEL_QWEN3:str
    LLM_MODEL_QWEN25:str
    LLM_MODEL_GEMMA:str
    LLM_API_KEY:str
    model_config=SettingsConfigDict(env_file=".env",env_file_encoding="utf-8",extra="ignore")

settings = Settings()

    
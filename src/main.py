from fastapi import FastAPI
from src.api.endpoints import ingest_document
from src.core.config import settings
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="RAG Service API",
    description="API xử lý tài liệu và truy vấn vector",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ingest_document.router, prefix="/ingest_document", tags=["ingest_document"])

@app.get("/")
async def root():
    return {"message": "RAG Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)

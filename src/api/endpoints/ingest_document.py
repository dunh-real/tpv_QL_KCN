import os
import shutil
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.core.config import settings
from src.core.logger import get_logger
from src.db.mongodb import docs_collection
from src.worker.process_document import process_document_task
from bson import ObjectId

logger = get_logger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Định dạng {ext} không hỗ trợ. Vui lòng dùng: {allowed_extensions}"
        )

    # Khởi tạo bản ghi PENDING trong MongoDB để lấy doc_id
    new_doc = {
        "filename": file.filename,
        "status": "PENDING",
        "created_at": datetime.now(),
        "total_pages": 0,
        "total_chunks": 0
    }
    insert_result = docs_collection.insert_one(new_doc)
    doc_id = str(insert_result.inserted_id)
    storage_dir = Path(settings.LOCAL_STORAGE_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / f"{doc_id}{ext}"
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        docs_collection.delete_one({"_id": ObjectId(doc_id)})
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {str(e)}")

    # Gửi sang Celery Worker
    process_document_task.delay(str(file_path), doc_id, file.filename)
    return {
        "message": "Tài liệu đã được tiếp nhận và đang xử lý.",
        "doc_id": doc_id,
        "filename": file.filename
    }

@router.get("/status/{doc_id}")
async def get_document_status(doc_id: str):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(status_code=400, detail="Mã doc_id không hợp lệ.")

    doc = docs_collection.find_one({"_id": ObjectId(doc_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")

    if doc.get("status") != "COMPLETED":
        doc.pop("context", None)

    result = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result
import os
from langchain_core.documents import Document
from src.services.chunking_service import ChunkingBatch
from src.services.ingestion_service import get_ingestion_service
from src.core.logger import get_logger

logger = get_logger(__name__)

def ingest_file(file_path: str) -> dict:
    """
    Đọc nội dung từ file, xử lý chunking và ingest vào VectorDB (Qdrant) & MongoDB.
    
    Args:
        file_path (str): Đường dẫn tài liệu cần ingest.
        
    Returns:
        dict: Thống kê kết quả ingest từ IngestionService.
    """
    if not os.path.exists(file_path):
        logger.error(f"File không tồn tại: {file_path}")
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
    logger.info(f"Bắt đầu xử lý file: {file_path}")

    # Bước 1: Đọc nội dung file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {e}")
        raise e
        
    doc = Document(
        page_content=content,
        metadata={
            "source": file_path,
            "filename": os.path.basename(file_path)
        }
    )
    
    # Bước 2: Chunking tài liệu thành parent & children
    logger.info("Đang thực hiện phân tách tài liệu (chunking)...")
    chunking_service = ChunkingBatch()
    parent_chunks, children_chunks = chunking_service.process_and_split(doc)
    
    if not parent_chunks and not children_chunks:
        logger.warning("Không có chunk nào được tạo ra từ tài liệu này.")
        return {"parent_chunks": 0, "children_chunks": 0}
        
    # Bước 3: Đưa qua IngestionService
    logger.info("Tiến hành ingest vào database (MongoDB & Qdrant)...")
    ingestion_service = get_ingestion_service()
    
    result = ingestion_service.ingest(parent_chunks, children_chunks)
    
    logger.info(f"Hoàn tất ingest file '{os.path.basename(file_path)}': {result}")
    return result
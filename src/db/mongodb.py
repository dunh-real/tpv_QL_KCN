from src.core.config import settings
from src.core.logger import get_logger
from pymongo import MongoClient

logger = get_logger(__name__)

_client = None
_db = None

def get_db():
    global _client, _db
    if _client is None:
        _client = MongoClient(settings.MONGO_URL, serverSelectionTimeoutMS=5000)
        _db = _client[settings.MONGODB_NAME]
        logger.info("Khởi tạo kết nối MongoDB thành công")
    return _db

def get_docs_collection():
    return get_db()[settings.MONGODB_COLLECTION_NAME]

class _LazyCollection:
    def __getattr__(self, name):
        return getattr(get_docs_collection(), name)

docs_collection = _LazyCollection()
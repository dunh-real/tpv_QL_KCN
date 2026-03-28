from src.core.config import settings
from src.db.hyperspace import EnterpriseDocumentStore

def get_vector_db():
    return EnterpriseDocumentStore(settings.HYPERSPACE_HOST, settings.HYPERSPACE_API_KEY, settings.HYPERSPACE_COLLECTION_NAME
    )

import re
import networkx as nx
from networkx.algorithms.approximation.steinertree import steiner_tree
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class MSSQLManager:
    """
    Class quản lý toàn bộ tương tác với MSSQL: Kết nối, Đồ thị Schema, và Thực thi Query.
    Đảm bảo nguyên tắc Singleton (chỉ khởi tạo 1 lần để giữ Connection Pool và Schema Graph in-memory).
    """
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = None
        self.db = None
        self.schema_graph = nx.Graph()
        
        self._connect()
        self._build_graph_from_db()

    def _connect(self):
        """Khởi tạo Connection Pool."""
        try:
            self.engine = create_engine(
                self.db_url,
                pool_size=10,          
                max_overflow=20,       
                pool_pre_ping=True,    
                pool_recycle=1800      
            ) 
            self.db = SQLDatabase(engine=self.engine)
            logger.info("Kết nối MSSQL thành công.")
        except Exception as e:
            logger.error(f"Lỗi kết nối MSSQL: {e}")
            self.db = None

    def _build_graph_from_db(self):
        """Tự động quét cấu trúc Khóa ngoại để vẽ Đồ thị."""
        if not self.engine:
            return
            
        logger.info("Đang tự động quét schema và xây dựng Schema Graph...")
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            for table in tables:
                self.schema_graph.add_node(table)
                fks = inspector.get_foreign_keys(table)
                for fk in fks:
                    referred_table = fk['referred_table']
                    self.schema_graph.add_edge(table, referred_table)
            logger.info(f"Hoàn tất! Graph có {self.schema_graph.number_of_nodes()} bảng.")
        except Exception as e:
            logger.error(f"Lỗi khi xây dựng Graph từ MSSQL: {e}")

    def get_related_tables(self, target_tables: list[str]) -> list[str]:
        """
        Phiên bản Tối ưu hóa cực hạn (Ultra-Optimized) cho GraphRAG.
        Sử dụng Set Operations O(1) và Memory-efficient Subgraph Views.
        """
        target_set = set(target_tables)
        valid_nodes = target_set.intersection(self.schema_graph.nodes)
        orphaned_nodes = target_set.difference(self.schema_graph.nodes)
        if len(valid_nodes) < 2:
            return list(valid_nodes.union(orphaned_nodes))

        final_tables = set(orphaned_nodes)
        for component in nx.connected_components(self.schema_graph):
            nodes_in_comp = valid_nodes.intersection(component)
            if not nodes_in_comp:
                continue
            n_count = len(nodes_in_comp)
            if n_count == 1:
                final_tables.update(nodes_in_comp)
            elif n_count == 2:
                u, v = nodes_in_comp
                try:
                    path = nx.shortest_path(self.schema_graph, u, v)
                    final_tables.update(path)
                except nx.NetworkXNoPath:
                    final_tables.update(nodes_in_comp)
                    
            else:
                try:
                    subgraph_view = self.schema_graph.subgraph(component)
                    st_tree = steiner_tree(subgraph_view, list(nodes_in_comp))
                    final_tables.update(st_tree.nodes)
                except Exception as e:
                    logger.warning(f"Lỗi Steiner Tree an toàn: {e}")
                    final_tables.update(nodes_in_comp)

        return list(final_tables)

    def get_schema_ddl(self, table_names: list[str]) -> str:
        """
        Lấy cấu trúc CREATE TABLE từ CSDL thật và dọn dẹp các cột rác (Audit).
        """
        if not self.db or not table_names:
            return ""
            
        try:
            raw_schema = self.db.get_table_info(table_names=table_names)
            
            # Xóa các cột Audit không cần thiết cho LLM
            audit_columns = ['TenantId', 'CreationTime', 'CreatorUserId', 
                             'LastModificationTime', 'LastModifierUserId', 
                             'IsDeleted', 'DeleterUserId', 'DeletionTime']
                             
            for col in audit_columns:
                raw_schema = re.sub(rf"^\s*{col}\s+.*?,?\n", "", raw_schema, flags=re.MULTILINE | re.IGNORECASE)
                
            return raw_schema
        except Exception as e:
            logger.error(f"Lỗi khi lấy DDL: {e}")
            return ""

    def run_query(self, sql_query: str) -> str:
        """
        Thực thi câu lệnh SQL thô với lớp bảo vệ Gatekeeper.
        """
        if not self.db:
            raise ConnectionError("Database chưa được kết nối.")
            
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE", "MERGE"]
        sql_upper = sql_query.upper()
        
        clean_sql = re.sub(r'--.*$', '', sql_upper, flags=re.MULTILINE)
        clean_sql = re.sub(r'\s+', ' ', clean_sql)
        
        for word in forbidden:
            if f" {word} " in f" {clean_sql} ":
                logger.warning(f"[Phát hiện câu lệnh không an toàn ở tầng DB]: {sql_query}")
                raise ValueError(f"Lỗi bảo mật DB-layer: AI Agent không được dùng lệnh {word}.")
        
        try:
            return self.db.run(sql_query)
        except Exception as e:
            logger.error(f"Lỗi khi chạy SQL '{sql_query}': {e}")
            raise e


_mssql_db_manager: MSSQLManager | None = None


def get_mssql_manager() -> MSSQLManager:
    """Lazy singleton — tạo MSSQLManager 1 lần duy nhất."""
    global _mssql_db_manager
    if _mssql_db_manager is None:
        _mssql_db_manager = MSSQLManager(settings.MSSQL_DATABASE_URL)
    return _mssql_db_manager

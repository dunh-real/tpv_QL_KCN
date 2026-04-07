import uuid
import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from src.core.logger import get_logger

logger = get_logger(__name__)


class ChunkingBatch:
    def __init__(self, tokenizer_model_name: str = "BAAI/bge-m3", max_tokens: int = 500, overlap: int = 100):
        """
        Khởi tạo ChunkingBatch với các tham số:
        Args:
            tokenizer_model_name (str): Tên model tokenizer để sử dụng cho việc chunking.
            max_tokens (int): Số lượng token tối đa cho mỗi chunk.
            overlap (int): Số lượng token overlap giữa các chunk.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
            strip_headers=False
        )
        self.child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=max_tokens,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    def protect_tables(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Tìm tất cả bảng Markdown và thay thế bằng placeholder để tránh bị cắt khi chunking.
        """
        protected_blocks = {}
        table_pattern = r'((?:^\|.*\|$\n?)+)'

        def replace_table(match):
            block_id = f"__PROTECTED_TABLE_{uuid.uuid4().hex[:8]}__"
            protected_blocks[block_id] = match.group(1).strip()
            return f"\n\n{block_id}\n\n"

        text = re.sub(table_pattern, replace_table, text, flags=re.MULTILINE)
        return text, protected_blocks

    def get_table_preview(self, table_str: str, max_rows: int = 4) -> str:
        """
        Lấy vài dòng đầu của bảng để lưu vào vector store.
        """
        lines = table_str.strip().split('\n')
        if len(lines) <= max_rows:
            return table_str
        preview_lines = lines[:max_rows]
        return "\n".join(preview_lines) + "\n| ... | ... (Bảng còn tiếp) |"

    def process_and_split(self, input_doc: Document) -> Tuple[List[Document], List[Document]]:
        """
        Xử lý và chia nhỏ tài liệu thành parent_chunks và children_chunks.

        - parent_chunks: Nội dung đầy đủ theo từng header section (bảng được khôi phục).
                         Dùng để lưu vào MongoDB.
        - children_chunks: Các đoạn nhỏ (text chunk + table preview) kèm parent_id.
                           Dùng để upsert vào Qdrant.

        Args:
            input_doc (Document): Tài liệu gốc cần xử lý.
        Returns:
            Tuple[List[Document], List[Document]]: (parent_chunks, children_chunks)
        """
        original_metadata = input_doc.metadata.copy()
        safe_text, tables_dict = self.protect_tables(input_doc.page_content)
        parent_docs = self.header_splitter.split_text(safe_text)

        parent_chunks: List[Document] = []
        children_chunks: List[Document] = []

        for parent in parent_docs:
            parent_id = str(uuid.uuid4())
            merged_metadata = {**original_metadata, **parent.metadata, "parent_id": parent_id}
            headers = [parent.metadata[h] for h in ["H1", "H2", "H3", "H4"] if h in parent.metadata]
            headers_context = " > ".join(headers)

            # Khôi phục bảng cho parent (nội dung đầy đủ)
            full_parent_content = parent.page_content
            # Phần text dành cho children (không chứa placeholder bảng)
            children_text = parent.page_content

            for block_id, table_content in tables_dict.items():
                if block_id in full_parent_content:
                    full_parent_content = full_parent_content.replace(block_id, table_content)
                    children_text = children_text.replace(block_id, "")

                    # Children chunk cho bảng
                    table_preview = self.get_table_preview(table_content)
                    contextualized_table = f"[Bối cảnh: {headers_context} - Bảng thống kê]\n{table_preview}"
                    table_meta = {**merged_metadata, "chunk_type": "table"}
                    children_chunks.append(
                        Document(page_content=contextualized_table, metadata=table_meta)
                    )

            # Lưu parent chunk 
            parent_chunks.append(
                Document(page_content=full_parent_content, metadata=merged_metadata)
            )

            # Children chunks cho phần văn bản
            children_text = children_text.strip()
            if children_text:
                child_splits = self.child_splitter.split_text(children_text)
                for child_text in child_splits:
                    contextualized_text = (
                        f"[Bối cảnh: {headers_context}]\n{child_text}" if headers_context else child_text
                    )
                    text_meta = {**merged_metadata, "chunk_type": "text"}
                    children_chunks.append(
                        Document(page_content=contextualized_text, metadata=text_meta)
                    )

        return parent_chunks, children_chunks

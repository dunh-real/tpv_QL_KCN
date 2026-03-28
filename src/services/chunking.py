import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

logger = logging.getLogger("ChunkingService")
logger.setLevel(logging.INFO)

class ChunkingService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False 
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

    def _reconstruct_header_string(self, headers: Dict[str, str]) -> str:
        """ Dựng lại chuỗi Markdown Header từ metadata cho trang tiếp theo"""
        header_text = ""
        for key, value in headers.items():
            if key == "Header 1": header_text += f"# {value}\n"
            elif key == "Header 2": header_text += f"## {value}\n"
            elif key == "Header 3": header_text += f"### {value}\n"
            elif key == "Header 4": header_text += f"#### {value}\n"
        return header_text

    def chunk_documents(self, pages: List[Document], base_metadata: Dict[str, Any]) -> List[Document]:
        """
        Xử lý từng trang một nhưng vẫn giữ được ngữ cảnh chéo trang.
        """
        final_chunks: List[Document] = []
        
        active_headers: Dict[str, str] = {}

        for page_doc in pages:
            page_num = page_doc.metadata.get("page", 1)
            page_text = page_doc.page_content

            if not page_text.strip():
                continue

            context_prefix = self._reconstruct_header_string(active_headers)
            text_to_split = context_prefix + "\n" + page_text

            # Cắt theo ngữ nghĩa Markdown
            md_docs = self.md_splitter.split_text(text_to_split)
            
            if not md_docs:
                continue

            last_chunk_metadata = md_docs[-1].metadata
            active_headers = {k: v for k, v in last_chunk_metadata.items() if k.startswith("Header")}

            # Cắt theo kích thước & Ép Metadata
            for md_chunk in md_docs:
                merged_metadata = {
                    **base_metadata,
                    "page": page_num, 
                    **md_chunk.metadata
                }

                # Cắt nhỏ thêm nếu một phân đoạn Header quá dài
                split_texts = self.text_splitter.split_text(md_chunk.page_content)

                for text_segment in split_texts:
                    clean_text = text_segment.strip()
                    
                    if clean_text and clean_text != context_prefix.strip():
                        final_chunks.append(
                            Document(
                                page_content=clean_text,
                                metadata=merged_metadata.copy()
                            )
                        )

        # 3. Đánh số Index cho toàn bộ Chunks
        total_chunks = len(final_chunks)
        for i, chunk in enumerate(final_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total_chunks

        logger.info(f"Đã tạo {total_chunks} chunks .")
        return final_chunks
import io
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Union

import fitz
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


_DEFAULT_PROMPT = (
    "Bạn là một chuyên gia số hóa tài liệu. "
    "Hãy trích xuất chính xác toàn bộ văn bản, bảng biểu, danh sách trong ảnh này. "
    "Trình bày kết quả dưới định dạng Markdown thuần túy. Không thêm lời mở đầu hay kết luận."
)

_MAX_OCR_WORKERS = 2


class OCRService:
    """
    Extract text from PDF/image documents using a Vision-Language Model
    and produce Markdown output.
    """

    def __init__(
        self,
        model_name: str = settings.OCR_MODEL,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=settings.OLLAMA_BASE_URL,
            )
            logger.info(f"Initialized OCR LLM with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR LLM: {e}")
            raise



    @staticmethod
    def _pixmap_to_base64(pix: fitz.Pixmap, max_size: int = 1024) -> str:
        """Convert a PyMuPDF Pixmap to a Base64-encoded JPEG string."""
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        if img.mode != "RGB":
            img = img.convert("RGB")

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to: {img.size}")

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        buffered.close()
        return b64_str

    def extract_from_image(
        self, b64_image: str, custom_prompt: Optional[str] = None
    ) -> str:
        """Send a base64 image to the VLM and return extracted text."""
        prompt_text = custom_prompt or _DEFAULT_PROMPT
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ]
        )
        response = self.llm.invoke([message])
        return response.content


    def _process_single_page(
        self,
        doc_bytes: bytes,
        page_num: int,
        total_pages: int,
        prompt: Optional[str],
    ) -> tuple[int, str]:
        """Process a single page and return (page_num, markdown_text).

        Opens a fresh fitz document from bytes so this method is safe
        to call from multiple threads (PyMuPDF Document is NOT thread-safe,
        but opening a new one per thread is).
        """
        try:
            with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150, colorspace=fitz.csRGB)
                b64_image = self._pixmap_to_base64(pix)
                del pix  # Free pixmap memory immediately

            page_text = self.extract_from_image(b64_image, prompt)
            del b64_image  # Free base64 string memory

            logger.info(f"Processed page {page_num + 1}/{total_pages}")
            return page_num, f"\n\n\n# Page {page_num + 1}\n{page_text}"
        except Exception as page_err:
            logger.error(f"Error processing page {page_num + 1}: {page_err}")
            return page_num, f"\n\n\n# Page {page_num + 1}\n[ERROR PROCESSING THIS PAGE]"



    def process_file(
        self,
        pdf_input: Union[bytes, str, Path],
        prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Trích xuất nội dung từ file PDF -> Định dạng Markdown.
        """
        if isinstance(pdf_input, bytes):
            pdf_bytes = pdf_input
        elif isinstance(pdf_input, (str, Path)):
            path_obj = Path(pdf_input)
            if not path_obj.exists():
                raise FileNotFoundError(f"File not found: {pdf_input}")
            pdf_bytes = path_obj.read_bytes()
        else:
            raise ValueError("pdf_input must be bytes, str, or Path")

        # --- Determine page count ---
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
        logger.info(f"Document has {total_pages} pages. Starting extraction...")

        # --- Process pages concurrently ---
        results: dict[int, str] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=_MAX_OCR_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._process_single_page, pdf_bytes, pn, total_pages, prompt
                ): pn
                for pn in range(total_pages)
            }

            for future in as_completed(futures):
                page_num, page_md = future.result()
                results[page_num] = page_md
                completed += 1

                if progress_callback:
                    progress_callback(completed, total_pages)

        # --- Reassemble in page order ---
        markdown_pages = [results[pn] for pn in range(total_pages)]
        final_markdown = "".join(markdown_pages)
        logger.info("Markdown assembly complete.")
        return final_markdown




_ocr_service_instance: Optional[OCRService] = None
_ocr_service_lock = threading.Lock()


def get_ocr_service() -> OCRService:
    """Thread-safe lazy singleton — creates OCRService once."""
    global _ocr_service_instance
    if _ocr_service_instance is None:
        with _ocr_service_lock:
            if _ocr_service_instance is None:
                _ocr_service_instance = OCRService()
    return _ocr_service_instance

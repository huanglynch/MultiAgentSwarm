"""
PDF 阅读工具（终极强化版 v2.0）
- 内存加载（BytesIO）彻底解决 "document closed" 和 Windows 文件锁定
- 3次自动重试 + 智能等待
- 更强的路径解析 + 详细日志
- 支持扫描件/图片 PDF（text + fallback）
"""
from pathlib import Path
import fitz  # PyMuPDF
from io import BytesIO
import time
import logging

def tool_function(file_path: str, max_pages: int = 50):
    for attempt in range(3):  # 3次重试机制
        try:
            # 路径规范化（兼容 Windows \ 和动态 UUID）
            file_path = str(file_path).replace('\\', '/').strip()
            if not file_path.startswith('uploads/'):
                file_path = 'uploads/' + file_path.split('uploads/')[-1].strip()

            logging.info(f"📎 pdf_reader 第{attempt+1}次尝试: {file_path}")

            PROJECT_ROOT = Path.cwd().resolve()
            target_path = PROJECT_ROOT / file_path
            target_path = target_path.resolve()

            if not target_path.exists() or target_path.suffix.lower() != ".pdf":
                return {
                    "success": False,
                    "error": f"文件不存在或不是 PDF: {target_path}",
                    "attempted_path": str(target_path)
                }

            # === 关键改进：先读入内存（彻底杜绝文件锁定）===
            with open(target_path, "rb") as f:
                pdf_bytes = f.read()

            # 使用内存流打开（最稳方式）
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                text = []
                for page_num in range(min(max_pages, len(doc))):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text")
                    if not page_text.strip():  # 扫描件 fallback
                        page_text = page.get_text("html")[:2000]  # 提取结构化内容
                    text.append(f"--- Page {page_num+1} ---\n{page_text}")

                full_text = "\n".join(text)

                return {
                    "success": True,
                    "file": file_path,
                    "pages": len(doc),
                    "extracted_pages": min(max_pages, len(doc)),
                    "content": full_text[:15000] + "\n...（如需更多内容请分批读取）",
                    "length": len(full_text)
                }

        except Exception as e:
            error_str = str(e).lower()
            if "document closed" in error_str or "invalid" in error_str:
                logging.warning(f"📎 pdf_reader 第{attempt+1}次失败（document closed），等待重试...")
                time.sleep(0.5 * (attempt + 1))  # 渐进式等待
                continue
            else:
                logging.error(f"pdf_reader 异常: {e}")
                return {"success": False, "error": str(e)}

    # 所有重试都失败
    return {"success": False, "error": "多次重试后仍无法读取 PDF（可能文件损坏或被占用）"}

tool_schema = {
    "type": "function",
    "function": {
        "name": "pdf_reader",
        "description": "读取 PDF 文件并提取纯文本（强化版：内存加载 + 重试 + 扫描件支持）。",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "PDF 文件路径（相对于项目根目录）"},
                "max_pages": {"type": "integer", "description": "最大读取页数", "default": 50}
            },
            "required": ["file_path"]
        }
    }
}
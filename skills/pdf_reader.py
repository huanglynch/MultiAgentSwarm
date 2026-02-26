"""
PDF 阅读工具
功能：提取 PDF 文本（论文、报告、技术文档）
"""
from pathlib import Path
import fitz  # PyMuPDF

def tool_function(file_path: str, max_pages: int = 50):
    try:
        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = SCRIPT_ROOT / target_path
        target_path = target_path.resolve()

        # ===== 安全检查（关键修复）=====
        try:
            relative_path = target_path.relative_to(SCRIPT_ROOT)
        except ValueError:
            return {
                "success": False,
                "error": "安全错误：不允许读取脚本目录外的文件",
                "attempted_path": str(target_path)
            }

        if not target_path.exists() or not target_path.suffix.lower() == ".pdf":
            return {"success": False, "error": "文件不存在或不是 PDF"}

        doc = fitz.open(target_path)
        text = []
        for page_num in range(min(max_pages, len(doc))):
            page = doc.load_page(page_num)
            text.append(f"--- Page {page_num+1} ---\n{page.get_text('text')}")

        doc.close()
        full_text = "\n".join(text)

        return {
            "success": True,
            "file": str(target_path.relative_to(SCRIPT_ROOT)),
            "pages": len(doc),
            "extracted_pages": min(max_pages, len(doc)),
            "content": full_text[:15000] + "\n...（如需更多内容请分批读取）",
            "length": len(full_text)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


tool_schema = {
    "type": "function",
    "function": {
        "name": "pdf_reader",
        "description": "读取 PDF 文件并提取纯文本（支持论文、报告、技术文档）。",
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
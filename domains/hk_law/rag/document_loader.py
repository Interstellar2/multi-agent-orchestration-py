"""
文档加载器

从文件系统加载法律文档，支持 .txt、.md、.pdf 格式。
"""
from typing import List

from langchain_core.documents import Document
from core.utils.logger import get_logger
from domains.hk_law.rag.config import RAGConfig

logger = get_logger(__name__)

_default_config = RAGConfig()


def load_documents(domain: str, config: RAGConfig = None) -> List[Document]:
    """
    从 documents/<domain>/ 目录加载法律文档。
    支持 .txt, .md, .pdf（需安装 PyPDF2）
    """
    cfg = config or _default_config
    domain_dir = cfg.documents_dir / domain
    if not domain_dir.exists():
        return []

    docs: List[Document] = []

    for ext, loader_cls in [
        ("*.txt", "langchain_community.document_loaders.TextLoader"),
        ("*.md", "langchain_community.document_loaders.TextLoader"),
    ]:
        for file_path in domain_dir.glob(ext):
            try:
                module_path, class_name = loader_cls.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                loader = getattr(module, class_name)(str(file_path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata.update({
                        "source": str(file_path.name),
                        "domain": domain,
                    })
                docs.extend(loaded)
            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {e}")

    # PDF
    try:
        from langchain_community.document_loaders import PyPDFLoader
        for file_path in domain_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata.update({
                        "source": str(file_path.name),
                        "domain": domain,
                    })
                docs.extend(loaded)
            except Exception as e:
                logger.warning(f"加载 PDF 失败 {file_path}: {e}")
    except ImportError:
        logger.debug("PyPDFLoader 未安装，跳过 PDF 加载")

    return docs

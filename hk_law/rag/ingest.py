"""
法律文档索引工具
将 documents/<domain>/ 下的文档批量索引到 Elasticsearch。

用法:
    # 索引单个法域
    python -m hk_law.rag.ingest criminal

    # 索引所有法域
    python -m hk_law.rag.ingest --all

    # 重建索引（先删除再重建）
    python -m hk_law.rag.ingest criminal --rebuild
"""
import argparse
import asyncio
import sys

from hk_law.rag.engine import check_es_health, delete_index, index_documents
from core.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="香港法律文档索引工具")
    parser.add_argument("domain", nargs="?", help="法域名称，如 criminal, company 等")
    parser.add_argument("--all", action="store_true", help="索引所有法域")
    parser.add_argument("--rebuild", action="store_true", help="重建索引（先删除）")
    args = parser.parse_args()

    # 检查 ES 连接
    if not check_es_health():
        logger.error("无法连接到 Elasticsearch。请确认：")
        logger.error("  1. docker-compose up -d 已启动")
        logger.error("  2. ELASTICSEARCH_URL 环境变量配置正确（默认 http://localhost:9200）")
        sys.exit(1)

    domains = []
    if args.all:
        from hk_law.agents import list_domains
        domains = list_domains()
    elif args.domain:
        domains = [args.domain]
    else:
        parser.print_help()
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"开始索引法域: {', '.join(domains)}")
    logger.info("=" * 60)

    for domain in domains:
        logger.info(f"[索引] 法域: {domain}")

        if args.rebuild:
            logger.info(f"[索引] 删除旧索引: {domain}")
            delete_index(domain)

        try:
            index_documents(domain)
            logger.info(f"[完成] {domain} 索引成功")
        except ValueError as e:
            logger.warning(f"[跳过] {e}")
        except Exception as e:
            logger.error(f"[错误] {domain} 索引失败: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

"""
香港法律文档自动下载脚本

从香港律政司电子版香港法例 (e-Legislation) 自动下载各法域的 PDF 文档。

用法:
    python -m hk_law.rag.download --all          # 下载所有法域
    python -m hk_law.rag.download criminal       # 下载单个法域
    python -m hk_law.rag.download criminal civil # 下载多个法域

说明:
    e-Legislation 网站有客户端配置检测机制（CSRF + Cookie + JS 重定向），
    本脚本使用 curl 完整模拟浏览器行为，自动完成验证并下载 PDF。
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List

from core.utils.logger import get_logger

logger = get_logger(__name__)

# 文档根目录
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"

# 法域 → 条例清单映射
# key: 法域目录名
# value: list of {"cap": "章节号", "name": "条例名称"}
ORDINANCES = {
    "criminal": [
        {"cap": "cap200", "name": "刑事罪行条例"},
        {"cap": "cap210", "name": "盗窃罪条例"},
    ],
    "civil": [
        {"cap": "cap623", "name": "合约(第三方权利)条例"},
        {"cap": "cap284", "name": "失实陈述条例"},
    ],
    "company": [
        {"cap": "cap622", "name": "公司条例"},
    ],
    "employment": [
        {"cap": "cap57", "name": "雇佣条例"},
    ],
    "property": [
        {"cap": "cap219", "name": "物业转易及财产条例"},
        {"cap": "cap344", "name": "建筑物管理条例"},
    ],
}


def _run_curl(args: List[str]) -> subprocess.CompletedProcess:
    """运行 curl 命令"""
    cmd = ["curl", "-sL"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def _extract_csrf(html: str) -> str:
    """从 HTML 中提取 _CSRF_TOKEN"""
    patterns = [
        r'name="_CSRF_TOKEN"\s+value="([^"]+)"',
        r'value="([^"]+)"\s+.*_CSRF_TOKEN',
        r'_CSRF_TOKEN.*value="([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return ""


def _extract_js_redirect(html: str) -> str:
    """从 HTML 中提取 JS 重定向 URL"""
    match = re.search(r'document\.location\.href="([^"]+)"', html)
    if match:
        return match.group(1)
    return ""


def download_ordinance(domain: str, cap: str, name: str, cookies_file: Path) -> Path:
    """
    下载单个条例的 PDF。

    流程:
        1. 访问条例页面 → 被重定向到 checkClientConfig
        2. 提取 CSRF 令牌
        3. POST 客户端配置表单
        4. 跟随 JS 重定向回条例页面
        5. 下载 PDF
    """
    domain_dir = DOCUMENTS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    output_path = domain_dir / f"{cap}.pdf"
    if output_path.exists():
        logger.info(f"  [{domain}] {name} ({cap}.pdf) 已存在，跳过")
        return output_path

    base_url = f"https://www.elegislation.gov.hk/hk/{cap}"
    pdf_url = f"{base_url}!zh-Hant-HK.pdf"

    logger.info(f"  [{domain}] 正在下载: {name} ({cap})...")

    # Step 1: 访问条例页面，获取 CSRF（会经过 302 到 checkClientConfig）
    resp1 = _run_curl(["-c", str(cookies_file), "-b", str(cookies_file), base_url])
    csrf = _extract_csrf(resp1.stdout)
    if not csrf:
        logger.error(f"    无法获取 CSRF 令牌，跳过 {cap}")
        return output_path

    # Step 2: 提交客户端配置表单
    form_data = [
        "-X", "POST",
        "-d", "applicationId=RA001",
        "-d", "branchCode=00",
        "-d", "javascriptEnabled=true",
        "-d", "cookieEnabled=true",
        "-d", "appletLoadFailed=false",
        "-d", f"_CSRF_TOKEN={csrf}",
    ]
    resp2 = _run_curl(
        ["-c", str(cookies_file), "-b", str(cookies_file)]
        + form_data
        + ["https://www.elegislation.gov.hk/checkconfig/submitClientConfig.do"]
    )

    # Step 3: 跟随 JS 重定向
    js_url = _extract_js_redirect(resp2.stdout)
    if js_url:
        _run_curl(["-c", str(cookies_file), "-b", str(cookies_file), js_url])

    # Step 4: 下载 PDF
    resp4 = _run_curl(
        ["-c", str(cookies_file), "-b", str(cookies_file), "-o", str(output_path), pdf_url]
    )

    if resp4.returncode != 0:
        logger.error(f"    curl 下载失败: {resp4.stderr}")
        return output_path

    # 验证是否为 PDF（避免下载到 HTML 错误页）
    result = subprocess.run(
        ["file", str(output_path)], capture_output=True, text=True
    )
    if "PDF document" in result.stdout:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"    已保存 {output_path.name} ({size_mb} MB)")
    else:
        logger.error("    下载内容不是 PDF，可能是验证失败")
        # 删除错误文件
        output_path.unlink(missing_ok=True)

    return output_path


def download_domain(domain: str) -> None:
    """下载指定法域的所有条例"""
    if domain not in ORDINANCES:
        logger.error(f"未知法域: {domain}")
        logger.error(f"可用法域: {', '.join(ORDINANCES.keys())}")
        return

    logger.info(f"[开始] 下载法域: {domain}")
    cookies_file = Path(f"/tmp/hk_law_download_{domain}_cookies.txt")

    for item in ORDINANCES[domain]:
        download_ordinance(domain, item["cap"], item["name"], cookies_file)
        # 每个条例之间短暂间隔，避免对服务器造成压力

    # 清理 cookie 文件
    cookies_file.unlink(missing_ok=True)
    logger.info(f"[完成] 法域 {domain} 下载结束")


def download_all() -> None:
    """下载所有法域"""
    logger.info("=" * 50)
    logger.info("开始下载所有法域的法律文档")
    logger.info("=" * 50)

    for domain in ORDINANCES:
        download_domain(domain)

    logger.info("=" * 50)
    logger.info("全部下载完成")
    logger.info("=" * 50)

    # 打印汇总
    logger.info("文件汇总:")
    for domain in ORDINANCES:
        domain_dir = DOCUMENTS_DIR / domain
        if domain_dir.exists():
            files = list(domain_dir.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
            logger.info(f"  {domain}: {len(files)} 个文件, 共 {total_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="自动下载香港法律文档 (e-Legislation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --all                           # 下载所有法域
  %(prog)s criminal                        # 下载刑事法域
  %(prog)s criminal civil employment       # 下载多个法域
  %(prog)s company --output-dir ./docs     # 指定输出目录
        """,
    )
    parser.add_argument(
        "domains",
        nargs="*",
        help=f"要下载的法域，可选: {', '.join(ORDINANCES.keys())}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有法域",
    )

    args = parser.parse_args()

    if args.all:
        download_all()
    elif args.domains:
        for domain in args.domains:
            download_domain(domain)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

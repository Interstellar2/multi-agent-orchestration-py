# 香港法律文档官方来源

本文档列出香港法律多 Agent 系统中各法域对应的官方条例下载链接。

所有文档均来自香港律政司 [**电子版香港法例 (e-Legislation)**](https://www.elegislation.gov.hk/)，提供香港法例的官方版本，免费下载。

---

## 下载说明

### 官方下载步骤（手动）

1. 访问条例页面链接（下方表格中的「条例页面」）
2. 点击页面顶部的 **「下载 (Download)」** 按钮
3. 在弹出窗口中选择：
   - **语言**: 中文 (繁体) `zh-Hant-HK`
   - **格式**: PDF
4. 点击「下载」保存到对应法域目录

### 自动下载（推荐）

运行自动下载脚本，无需手动点击：

```bash
python -m hk_law.rag.download --all
```

或下载单个法域：

```bash
python -m hk_law.rag.download criminal
python -m hk_law.rag.download civil
python -m hk_law.rag.download company
python -m hk_law.rag.download employment
python -m hk_law.rag.download property
```

下载的文件会自动保存到 `hk_law/documents/<domain>/` 目录下。

---

## 法域文档清单

### 刑事法域 (criminal)

| 条例名称 | 章节编号 | 条例页面 | 直接 PDF 链接 |
|----------|----------|----------|---------------|
| 刑事罪行条例 | Cap. 200 | https://www.elegislation.gov.hk/hk/cap200 | https://www.elegislation.gov.hk/hk/cap200!zh-Hant-HK.pdf |
| 盗窃罪条例 | Cap. 210 | https://www.elegislation.gov.hk/hk/cap210 | https://www.elegislation.gov.hk/hk/cap210!zh-Hant-HK.pdf |

### 民事法域 (civil)

| 条例名称 | 章节编号 | 条例页面 | 直接 PDF 链接 |
|----------|----------|----------|---------------|
| 合约(第三方权利)条例 | Cap. 623 | https://www.elegislation.gov.hk/hk/cap623 | https://www.elegislation.gov.hk/hk/cap623!zh-Hant-HK.pdf |
| 失实陈述条例 | Cap. 284 | https://www.elegislation.gov.hk/hk/cap284 | https://www.elegislation.gov.hk/hk/cap284!zh-Hant-HK.pdf |

### 公司法域 (company)

| 条例名称 | 章节编号 | 条例页面 | 直接 PDF 链接 |
|----------|----------|----------|---------------|
| 公司条例 | Cap. 622 | https://www.elegislation.gov.hk/hk/cap622 | https://www.elegislation.gov.hk/hk/cap622!zh-Hant-HK.pdf |

### 雇佣法域 (employment)

| 条例名称 | 章节编号 | 条例页面 | 直接 PDF 链接 |
|----------|----------|----------|---------------|
| 雇佣条例 | Cap. 57 | https://www.elegislation.gov.hk/hk/cap57 | https://www.elegislation.gov.hk/hk/cap57!zh-Hant-HK.pdf |

### 物业法域 (property)

| 条例名称 | 章节编号 | 条例页面 | 直接 PDF 链接 |
|----------|----------|----------|---------------|
| 物业转易及财产条例 | Cap. 219 | https://www.elegislation.gov.hk/hk/cap219 | https://www.elegislation.gov.hk/hk/cap219!zh-Hant-HK.pdf |
| 建筑物管理条例 | Cap. 344 | https://www.elegislation.gov.hk/hk/cap344 | https://www.elegislation.gov.hk/hk/cap344!zh-Hant-HK.pdf |

---

## 技术说明

### 下载机制

电子版香港法例网站 (`elegislation.gov.hk`) 启用了客户端配置检测机制：

1. 首次访问时会检查浏览器是否支持 JavaScript 和 Cookies
2. 需要通过一个配置验证表单（`_CSRF_TOKEN`）
3. 验证通过后，会话 Cookie 会记录验证状态
4. 之后即可正常下载 PDF 文件

自动下载脚本 (`download.py`) 已完整模拟上述流程，使用 `curl` 处理 Cookie、CSRF 令牌和 JavaScript 重定向。

### 注意事项

- 所有 PDF 均为官方中文（繁体）版本
- 文件较大（5MB - 30MB），请确保网络稳定
- 官方条例会不定期更新，建议定期重新下载以确保内容最新
- 下载内容受香港法例版权保护，仅供个人学习及研究用途

---

## 参考链接

- 电子版香港法例首页: https://www.elegislation.gov.hk/
- 香港律政司: https://www.doj.gov.hk/
- 按章节编号检索: https://www.elegislation.gov.hk/hk/quicksearch?lang=zh-Hant-HK

---

*最后更新: 2026-05-31*

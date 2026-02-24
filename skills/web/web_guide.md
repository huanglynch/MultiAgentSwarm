# 网络工具使用指南

## web_search 工具
**功能**：实时搜索网络最新信息

**数据源**：DuckDuckGo（无需 API Key）

**参数**：
- `query` (必需)：搜索关键词
- `num_results` (可选)：返回结果数量 (1-10)

**返回**：
- 每条结果包含：标题、URL、摘要

**适用场景**：
- 查询最新新闻和事件
- 获取技术文档和教程
- 收集市场和行业信息

**示例**：
```python
web_search(query="Python 最新版本特性", num_results=5)
```

## browse_page 工具
**功能**：获取指定网页的纯文本内容

**参数**：
- `url` (必需)：目标网页地址
- `max_length` (可选)：最大返回长度

**特性**：
- 自动移除脚本和样式
- 提取标题和正文
- 智能截断过长内容

**适用场景**：
- 分析特定网页内容
- 提取文章正文
- 监控网页变化

**示例**：
```python
browse_page(url="https://example.com/article", max_length=3000)
```

## 最佳实践
1. 搜索使用具体关键词，避免过于宽泛
2. 浏览网页前验证 URL 有效性
3. 注意遵守网站的 robots.txt 规则
4. 大量请求时添加延迟避免被封禁
5. 优先使用搜索，明确目标后再浏览

## 依赖安装
```bash
pip install duckduckgo-search requests beautifulsoup4
```

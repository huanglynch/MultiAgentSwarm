"""
Skills 目录结构初始化脚本
自动创建完整的工具和知识文件
"""

import os

# 目录结构
STRUCTURE = {
    "file": ["read_file.py", "write_file.py", "file_guide.md"],
    "web": ["web_search.py", "browse_page.py", "web_guide.md"],
    "data": ["csv_reader.py", "json_parser.py", "data_format.md"],
    "knowledge": ["ai_basics.md", "coding_standards.md"]
}

# 文件内容模板
FILE_CONTENTS = {
    # ========== File 工具 ==========
    "file/read_file.py": '''"""
文件读取工具
功能：读取指定文件的内容
"""

def tool_function(file_path: str):
    """读取文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"success": True, "content": content, "length": len(content)}
    except FileNotFoundError:
        return {"success": False, "error": f"文件不存在: {file_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取指定文件的内容",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要读取的文件路径"
                }
            },
            "required": ["file_path"]
        }
    }
}
''',

    "file/write_file.py": '''"""
文件写入工具
功能：将内容写入指定文件
"""

import os

def tool_function(file_path: str, content: str, mode: str = "w"):
    """写入文件内容"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        with open(file_path, mode, encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True, 
            "file_path": file_path, 
            "bytes_written": len(content.encode("utf-8"))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "将内容写入指定文件，自动创建目录",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "目标文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容"
                },
                "mode": {
                    "type": "string",
                    "description": "写入模式：w(覆盖) 或 a(追加)",
                    "enum": ["w", "a"],
                    "default": "w"
                }
            },
            "required": ["file_path", "content"]
        }
    }
}
''',

    "file/file_guide.md": '''# 文件操作工具使用指南

## read_file 工具
**功能**：读取文本文件内容

**参数**：
- `file_path` (必需)：文件路径

**返回**：
- `success`: 操作是否成功
- `content`: 文件内容
- `length`: 内容长度

**示例**：
```python
read_file(file_path="./data/example.txt")
```

## write_file 工具
**功能**：写入内容到文件

**参数**：
- `file_path` (必需)：目标文件路径
- `content` (必需)：要写入的内容
- `mode` (可选)：`w` 覆盖写入，`a` 追加写入

**特性**：
- 自动创建不存在的目录
- 支持 UTF-8 编码

**示例**：
```python
write_file(file_path="./output/report.txt", content="报告内容", mode="w")
```

## 最佳实践
1. 读取前检查文件是否存在
2. 写入大文件时考虑分块处理
3. 使用绝对路径避免路径混淆
4. 敏感文件操作前备份
''',

    # ========== Web 工具 ==========
    "web/web_search.py": '''"""
网页搜索工具
功能：使用 DuckDuckGo 搜索实时信息
"""

def tool_function(query: str, num_results: int = 5):
    """实时网页搜索"""
    try:
        from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=num_results)
        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            }
            for r in results
        ]

        return {
            "success": True,
            "query": query,
            "results": formatted,
            "count": len(formatted)
        }
    except ImportError:
        return {
            "success": False,
            "error": "请安装依赖: pip install duckduckgo-search"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "使用 DuckDuckGo 搜索最新网络信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量 (1-10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}
''',

    "web/browse_page.py": '''"""
网页浏览工具
功能：获取网页内容并提取文本
"""

def tool_function(url: str, max_length: int = 5000):
    """获取网页内容"""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\\n", strip=True)

        # 限制长度
        if len(text) > max_length:
            text = text[:max_length] + "\\n... (内容已截断)"

        return {
            "success": True,
            "url": url,
            "title": soup.title.string if soup.title else "无标题",
            "content": text,
            "length": len(text)
        }
    except ImportError:
        return {
            "success": False,
            "error": "请安装依赖: pip install requests beautifulsoup4"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "browse_page",
        "description": "获取网页内容并提取纯文本",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "目标网页 URL"
                },
                "max_length": {
                    "type": "integer",
                    "description": "最大返回文本长度",
                    "default": 5000
                }
            },
            "required": ["url"]
        }
    }
}
''',

    "web/web_guide.md": '''# 网络工具使用指南

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
''',

    # ========== Data 工具 ==========
    "data/csv_reader.py": '''"""
CSV 读取工具
功能：读取和解析 CSV 文件
"""

def tool_function(file_path: str, max_rows: int = 100):
    """读取 CSV 文件"""
    try:
        import csv

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                data.append(dict(row))

        return {
            "success": True,
            "file_path": file_path,
            "rows": len(data),
            "columns": list(data[0].keys()) if data else [],
            "data": data
        }
    except FileNotFoundError:
        return {"success": False, "error": f"文件不存在: {file_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "csv_reader",
        "description": "读取 CSV 文件并解析为结构化数据",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "CSV 文件路径"
                },
                "max_rows": {
                    "type": "integer",
                    "description": "最大读取行数",
                    "default": 100
                }
            },
            "required": ["file_path"]
        }
    }
}
''',

    "data/json_parser.py": '''"""
JSON 解析工具
功能：读取和解析 JSON 文件或字符串
"""

import json

def tool_function(source: str, source_type: str = "file"):
    """解析 JSON 数据"""
    try:
        if source_type == "file":
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            source_info = f"文件: {source}"
        elif source_type == "string":
            data = json.loads(source)
            source_info = "JSON 字符串"
        else:
            return {"success": False, "error": "source_type 必须是 'file' 或 'string'"}

        return {
            "success": True,
            "source": source_info,
            "data": data,
            "type": type(data).__name__
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON 格式错误: {e}"}
    except FileNotFoundError:
        return {"success": False, "error": f"文件不存在: {source}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "json_parser",
        "description": "解析 JSON 文件或字符串",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "文件路径或 JSON 字符串"
                },
                "source_type": {
                    "type": "string",
                    "description": "数据源类型",
                    "enum": ["file", "string"],
                    "default": "file"
                }
            },
            "required": ["source"]
        }
    }
}
''',

    "data/data_format.md": '''# 数据处理工具说明

## csv_reader 工具
**功能**：读取 CSV 文件并转换为结构化数据

**参数**：
- `file_path` (必需)：CSV 文件路径
- `max_rows` (可选)：最大读取行数，默认 100

**返回格式**：
```python
{
    "success": True,
    "rows": 50,
    "columns": ["id", "name", "value"],
    "data": [
        {"id": "1", "name": "项目A", "value": "100"},
        ...
    ]
}
```

**适用场景**：
- 分析表格数据
- 数据预处理
- 生成统计报告

## json_parser 工具
**功能**：解析 JSON 文件或字符串

**参数**：
- `source` (必需)：文件路径或 JSON 字符串
- `source_type` (可选)：`file` 或 `string`

**返回格式**：
```python
{
    "success": True,
    "source": "文件: data.json",
    "data": {...},  # 解析后的数据
    "type": "dict"   # 数据类型
}
```

**适用场景**：
- API 响应解析
- 配置文件读取
- 数据验证

## 数据格式最佳实践

### CSV 文件
- 第一行必须是列名
- 使用 UTF-8 编码
- 避免单元格内换行
- 大文件考虑分批处理

### JSON 文件
- 使用标准 JSON 格式
- 避免过深的嵌套 (>5层)
- 大对象考虑流式解析
- 敏感数据注意加密

## 常见错误处理
1. **编码错误**：确保文件使用 UTF-8
2. **格式错误**：验证数据格式正确性
3. **内存溢出**：大文件使用 max_rows 限制
4. **路径错误**：使用绝对路径或相对路径
''',

    # ========== Knowledge 文件 ==========
    "knowledge/ai_basics.md": '''# 人工智能基础知识

## AI 核心概念

### 1. 机器学习 (Machine Learning)
- **监督学习**：从标注数据中学习
- **无监督学习**：从未标注数据中发现模式
- **强化学习**：通过试错学习最优策略

### 2. 深度学习 (Deep Learning)
- 基于人工神经网络
- 多层非线性变换
- 自动特征提取

### 3. 大语言模型 (LLM)
- 基于 Transformer 架构
- 预训练 + 微调范式
- 涌现能力：推理、生成、理解

## AI 应用领域

### 自然语言处理 (NLP)
- 文本生成
- 机器翻译
- 情感分析
- 问答系统

### 计算机视觉 (CV)
- 图像识别
- 目标检测
- 图像生成
- 视频分析

### 语音技术
- 语音识别 (ASR)
- 语音合成 (TTS)
- 声纹识别

## AI 发展趋势

1. **多模态融合**：文本、图像、音频联合理解
2. **小样本学习**：降低数据依赖
3. **可解释性**：提高模型透明度
4. **边缘计算**：模型轻量化和本地化
5. **AI 安全**：对抗攻击防御、隐私保护

## 重要里程碑

- **2012**：AlexNet 开启深度学习时代
- **2017**：Transformer 架构发布
- **2018**：BERT 预训练模型
- **2020**：GPT-3 展示涌现能力
- **2022**：ChatGPT 引发 AI 革命
- **2023**：多模态大模型爆发

## 伦理与挑战

### 伦理问题
- 算法偏见和公平性
- 隐私保护
- 就业影响
- 自主武器

### 技术挑战
- 数据质量和数量
- 计算资源需求
- 模型可解释性
- 安全性和鲁棒性

## 学习资源

### 在线课程
- Andrew Ng 的机器学习课程
- Fast.ai 深度学习课程
- Stanford CS231n (计算机视觉)
- Stanford CS224n (NLP)

### 经典书籍
- 《深度学习》(Goodfellow)
- 《统计学习方法》(李航)
- 《Python 机器学习》(Sebastian Raschka)

### 实践平台
- Kaggle 竞赛
- GitHub 开源项目
- Hugging Face 模型库
''',

    "knowledge/coding_standards.md": '''# 编码规范与最佳实践

## Python 编码规范 (PEP 8)

### 命名约定
```python
# 变量和函数：小写 + 下划线
user_name = "Alice"
def calculate_total(): pass

# 类名：大驼峰
class UserProfile: pass

# 常量：全大写 + 下划线
MAX_CONNECTIONS = 100

# 私有成员：单下划线前缀
def _internal_method(): pass
```

### 代码布局
```python
# 缩进：4 个空格
def example():
    if condition:
        do_something()

# 行长度：最多 79 字符
# 长行使用括号换行
result = some_function(
    argument1, argument2,
    argument3, argument4
)

# 空行：类和函数之间 2 行，方法之间 1 行
class MyClass:

    def method1(self):
        pass

    def method2(self):
        pass


class AnotherClass:
    pass
```

### 导入规范
```python
# 导入顺序：标准库 -> 第三方库 -> 本地模块
import os
import sys

import numpy as np
import pandas as pd

from .local_module import function
```

## 代码质量原则

### 1. SOLID 原则
- **S**ingle Responsibility：单一职责
- **O**pen/Closed：开闭原则
- **L**iskov Substitution：里氏替换
- **I**nterface Segregation：接口隔离
- **D**ependency Inversion：依赖倒置

### 2. DRY 原则
Don't Repeat Yourself - 避免代码重复

```python
# ❌ 不好的做法
def calculate_area_rectangle(width, height):
    return width * height

def calculate_area_square(side):
    return side * side

# ✅ 好的做法
def calculate_area(width, height=None):
    height = height or width
    return width * height
```

### 3. KISS 原则
Keep It Simple, Stupid - 保持简单

```python
# ❌ 过度复杂
result = [x for x in range(10) if x % 2 == 0 if x > 5]

# ✅ 清晰易懂
numbers = range(10)
even_numbers = [x for x in numbers if x % 2 == 0]
result = [x for x in even_numbers if x > 5]
```

## 文档注释

### 函数文档
```python
def process_data(data: list, threshold: float = 0.5) -> dict:
    """
    处理输入数据并返回统计结果

    Args:
        data: 待处理的数值列表
        threshold: 过滤阈值，默认 0.5

    Returns:
        包含统计信息的字典：
        {
            'count': int,
            'mean': float,
            'filtered': list
        }

    Raises:
        ValueError: 当 data 为空时抛出

    Example:
        >>> process_data([1, 2, 3, 4, 5], threshold=2.5)
        {'count': 5, 'mean': 3.0, 'filtered': [3, 4, 5]}
    """
    if not data:
        raise ValueError("数据不能为空")

    return {
        'count': len(data),
        'mean': sum(data) / len(data),
        'filtered': [x for x in data if x > threshold]
    }
```

## 错误处理

### 具体异常
```python
# ❌ 捕获所有异常
try:
    result = risky_operation()
except:
    pass

# ✅ 捕获具体异常
try:
    result = risky_operation()
except FileNotFoundError:
    logger.error("文件未找到")
    result = default_value
except PermissionError:
    logger.error("权限不足")
    raise
```

### 自定义异常
```python
class DataValidationError(Exception):
    """数据验证异常"""
    pass

def validate_data(data):
    if not isinstance(data, dict):
        raise DataValidationError(f"期望字典类型，得到 {type(data)}")
```

## 性能优化

### 使用生成器
```python
# ❌ 占用大量内存
def get_all_items():
    return [process(x) for x in range(1000000)]

# ✅ 节省内存
def get_all_items():
    return (process(x) for x in range(1000000))
```

### 列表推导式 vs 循环
```python
# ✅ 列表推导式更快
squares = [x**2 for x in range(1000)]

# ❌ 传统循环较慢
squares = []
for x in range(1000):
    squares.append(x**2)
```

## 测试最佳实践

### 单元测试
```python
import unittest

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DataProcessor()

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            self.processor.process([])

    def test_normal_case(self):
        result = self.processor.process([1, 2, 3])
        self.assertEqual(result['count'], 3)
        self.assertAlmostEqual(result['mean'], 2.0)
```

## Git 提交规范

### Commit Message 格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 示例
```
feat(auth): 添加 JWT 认证功能

实现基于 JWT 的用户认证系统，包括：
- Token 生成和验证
- 刷新 Token 机制
- 权限验证装饰器

Closes #123
```

## 代码审查清单

- [ ] 代码符合 PEP 8 规范
- [ ] 函数和类有完整文档
- [ ] 有适当的错误处理
- [ ] 有单元测试覆盖
- [ ] 没有硬编码的配置
- [ ] 没有安全漏洞
- [ ] 性能满足要求
- [ ] 代码可读性好
'''
}


def create_skills_structure():
    """创建 skills 目录结构和文件"""
    base_dir = "skills"

    print("🚀 开始创建 Skills 目录结构...\n")

    # 创建主目录
    os.makedirs(base_dir, exist_ok=True)
    print(f"📁 创建主目录: {base_dir}/")

    # 创建子目录和文件
    for category, files in STRUCTURE.items():
        category_path = os.path.join(base_dir, category)
        os.makedirs(category_path, exist_ok=True)
        print(f"📁 创建子目录: {category_path}/")

        for file_name in files:
            file_path = os.path.join(category_path, file_name)

            # 获取文件内容
            content_key = f"{category}/{file_name}"
            content = FILE_CONTENTS.get(content_key, f"# {file_name}\n\n这是一个占位文件\n")

            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # 图标显示
            icon = "📝" if file_name.endswith(".py") else "📚"
            print(f"   {icon} 创建文件: {file_path}")

    print("\n" + "=" * 60)
    print("✅ Skills 目录结构创建完成！")
    print("=" * 60)
    print("\n📊 目录结构：")
    print(f"""
skills/
├── file/          (文件操作工具)
│   ├── read_file.py
│   ├── write_file.py
│   └── file_guide.md
├── web/           (网络工具)
│   ├── web_search.py
│   ├── browse_page.py
│   └── web_guide.md
├── data/          (数据处理工具)
│   ├── csv_reader.py
│   ├── json_parser.py
│   └── data_format.md
└── knowledge/     (知识文档)
    ├── ai_basics.md
    └── coding_standards.md
    """)

    print("\n📦 需要安装的依赖：")
    print("pip install duckduckgo-search requests beautifulsoup4")

    print("\n🎯 下一步操作：")
    print("1. 运行主程序测试：python multi_agent_swarm_v2.py")
    print("2. 查看日志确认工具加载：检查 'skills 加载完成' 消息")
    print("3. 根据需要修改或添加自定义工具")


if __name__ == "__main__":
    try:
        create_skills_structure()
    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
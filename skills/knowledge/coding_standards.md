# 编码规范与最佳实践

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

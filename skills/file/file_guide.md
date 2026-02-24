# 文件操作工具使用指南

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

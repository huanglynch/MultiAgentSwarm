# 📁 文件操作工具使用指南

## 🔒 安全机制

所有文件操作工具都基于 **脚本根目录**（`D:/huang/data/working/python/openagent/`）进行：

- ✅ **只能访问脚本根目录及其子目录**
- ❌ **自动拒绝访问父目录或系统目录**
- 🔐 **文件类型白名单保护**（仅支持文本格式）
- 📂 **自动创建不存在的目录**

---

## 📖 read_file - 读取文件

### 功能
读取文本文件内容（UTF-8 编码）

### 参数
- `file_path` **(必需)**：文件路径（相对于脚本根目录）

### 返回值
```json
{
  "success": true,
  "content": "文件内容...",
  "length": 1234,
  "file_path": "D:/huang/.../openagent/data/input.txt",
  "relative_path": "data/input.txt"
}
```

### 使用示例

**示例 1：读取子目录文件**
```python
read_file(file_path="data/example.txt")
# 实际读取: D:/huang/.../openagent/data/example.txt
```

**示例 2：读取当前目录文件**
```python
read_file(file_path="config.yaml")
# 实际读取: D:/huang/.../openagent/config.yaml
```

**示例 3：使用 ./ 前缀**
```python
read_file(file_path="./reports/summary.md")
# 实际读取: D:/huang/.../openagent/reports/summary.md
```

### ⚠️ 安全限制示例

```python
# ❌ 尝试读取父目录（会被拒绝）
read_file(file_path="../secret.txt")
# 返回: {"success": false, "error": "安全错误：不允许读取脚本目录外的文件"}

# ❌ 尝试读取系统目录（会被拒绝）
read_file(file_path="C:/Windows/system.ini")
# 返回: {"success": false, "error": "安全错误：不允许读取脚本目录外的文件"}
```

---

## ✍️ write_file - 写入文件

### 功能
将内容写入文件（UTF-8 编码）

### 参数
- `file_path` **(必需)**：目标文件路径（相对于脚本根目录）
- `content` **(必需)**：要写入的文本内容
- `mode` **(可选)**：写入模式
  - `"w"` (默认)：覆盖写入（清空原内容）
  - `"a"`：追加写入（保留原内容）

### 返回值
```json
{
  "success": true,
  "file_path": "D:/huang/.../openagent/reports/ai_report.md",
  "relative_path": "reports/ai_report.md",
  "bytes_written": 2048,
  "file_size": 2048,
  "mode": "覆盖写入"
}
```

### 使用示例

**示例 1：创建新文件**
```python
write_file(
    file_path="reports/ai_analysis.md",
    content="# AI 分析报告\n\n这是报告内容...",
    mode="w"
)
# 自动创建 reports/ 目录（如不存在）
# 实际保存: D:/huang/.../openagent/reports/ai_analysis.md
```

**示例 2：追加内容到现有文件**
```python
write_file(
    file_path="logs/system.log",
    content="\n[2026-02-24] 新日志条目...",
    mode="a"
)
# 实际保存: D:/huang/.../openagent/logs/system.log (追加模式)
```

**示例 3：多级目录自动创建**
```python
write_file(
    file_path="output/data/results/final.json",
    content='{"status": "success"}',
    mode="w"
)
# 自动创建: output/data/results/ 目录结构
```

### 📝 支持的文件类型
```
.txt  .md   .json  .csv  .yaml  .yml
.log  .html .xml   .py   .sh    .sql  .rst
```

### ⚠️ 安全限制示例

```python
# ❌ 不支持的文件类型
write_file(file_path="malware.exe", content="binary data")
# 返回: {"success": false, "error": "不允许的文件类型: .exe"}

# ❌ 尝试写入父目录
write_file(file_path="../../../etc/passwd", content="hack")
# 返回: {"success": false, "error": "安全错误：不允许写入脚本目录外的文件"}
```

---

## 📂 list_dir - 列出目录

### 功能
列出指定目录下的文件和子目录

### 参数
- `directory` **(可选)**：目录路径（相对于脚本根目录）
  - 默认值：`"."` (脚本根目录)

### 返回值
```json
{
  "success": true,
  "directory": "reports",
  "files": [
    {
      "name": "ai_report.md",
      "path": "reports/ai_report.md",
      "size": 2048,
      "type": "file"
    }
  ],
  "directories": [
    {
      "name": "archives",
      "path": "reports/archives",
      "type": "directory"
    }
  ],
  "total_files": 1,
  "total_directories": 1
}
```

### 使用示例

**示例 1：列出根目录**
```python
list_dir()
# 或
list_dir(directory=".")
# 列出: D:/huang/.../openagent/ 下的所有文件和文件夹
```

**示例 2：列出子目录**
```python
list_dir(directory="reports")
# 列出: D:/huang/.../openagent/reports/ 的内容
```

**示例 3：列出多级子目录**
```python
list_dir(directory="output/data/results")
# 列出: D:/huang/.../openagent/output/data/results/ 的内容
```

### ⚠️ 安全限制示例

```python
# ❌ 尝试访问父目录
list_dir(directory="../../../")
# 返回: {"success": false, "error": "安全错误：不允许访问脚本目录外的路径"}
```

---

## 🎯 最佳实践

### 1️⃣ 路径规范
```python
# ✅ 推荐：相对路径（清晰简洁）
write_file("reports/output.md", content)

# ✅ 可以：使用 ./ 前缀（更明确）
write_file("./reports/output.md", content)

# ⚠️ 避免：绝对路径（除非必要）
write_file("D:/huang/.../reports/output.md", content)
```

### 2️⃣ 错误处理
```python
# ✅ 检查操作是否成功
result = read_file("data/input.txt")
if result["success"]:
    content = result["content"]
else:
    print(f"读取失败: {result['error']}")
```

### 3️⃣ 大文件处理
```python
# ⚠️ read_file 一次性读取全部内容，不适合大文件
# 对于超大文件（>10MB），建议分块处理或使用专用工具
```

### 4️⃣ 敏感操作
```python
# ✅ 写入前先读取备份（如需覆盖重要文件）
backup = read_file("important.txt")
if backup["success"]:
    write_file("important.txt.bak", backup["content"])
    write_file("important.txt", new_content)
```

### 5️⃣ 路径检查
```python
# ✅ 先列出目录确认文件存在
files = list_dir("data")
if any(f["name"] == "target.txt" for f in files["files"]):
    content = read_file("data/target.txt")
```

---

## 🚨 常见错误

### 错误 1：文件不存在
```
{"success": false, "error": "文件不存在: data/missing.txt"}
```
**解决**：检查文件名拼写、路径是否正确

---

### 错误 2：权限错误
```
{"success": false, "error": "权限错误：无法写入文件 logs/system.log"}
```
**解决**：确保脚本目录有写入权限

---

### 错误 3：不支持的文件类型
```
{"success": false, "error": "不允许的文件类型: .exe"}
```
**解决**：只写入文本格式文件（.txt, .md, .json 等）

---

### 错误 4：编码错误
```
{"success": false, "error": "文件不是 UTF-8 文本格式或包含二进制内容"}
```
**解决**：确保文件是 UTF-8 编码的纯文本

---

## 📦 目录结构建议

```
openagent/                   # 脚本根目录
├── multi_agent_swarm_v2.py  # 主程序
├── swarm_config.yaml        # 配置
│
├── data/                    # 输入数据
│   ├── input.txt
│   └── config.json
│
├── reports/                 # 输出报告
│   ├── ai_report.md
│   └── analysis.txt
│
├── logs/                    # 日志文件
│   └── system.log
│
├── output/                  # 临时输出
│   └── results/
│       └── final.json
│
└── skills/                  # 工具目录（只读）
    └── file/
        ├── read_file.py
        ├── write_file.py
        └── list_dir.py
```

---

## 🔧 调试技巧

### 打印实际路径
```python
# 查看文件实际保存位置
result = write_file("test.txt", "hello")
print(result["file_path"])
# 输出: D:/huang/data/working/python/openagent/test.txt
```

### 验证路径安全性
```python
# 测试各种路径输入
test_paths = [
    "normal.txt",           # ✅ 应该成功
    "./reports/file.md",    # ✅ 应该成功
    "../secret.txt",        # ❌ 应该被拒绝
    "output/../../hack.py"  # ❌ 应该被拒绝
]

for path in test_paths:
    result = write_file(path, "test")
    print(f"{path}: {result['success']}")
```

---

## 📚 相关资源

- [Python pathlib 文档](https://docs.python.org/3/library/pathlib.html)
- [文件 I/O 最佳实践](https://realpython.com/read-write-files-python/)
- [安全编程指南](https://owasp.org/www-project-secure-coding-practices/)

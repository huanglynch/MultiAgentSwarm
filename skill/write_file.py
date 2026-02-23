def execute(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"写入成功: {path}"
    except Exception as e:
        return f"写入失败: {str(e)}"

schema = { ... }  # 与之前完全一样，省略重复

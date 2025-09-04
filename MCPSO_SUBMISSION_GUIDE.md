# mcp.so 发布指南

本指南详细说明如何将MongoDB自然语言查询MCP服务器发布到mcp.so平台。

## 📋 发布前检查清单

### 必需文件
- [x] `mcp.json` - MCP服务器配置文件
- [x] `MCP_SO_README.md` - mcp.so专用README文档
- [x] `src/mongo_query_server.py` - 核心服务器代码
- [x] `src/start_mcp_server.py` - 启动脚本
- [x] `requirements.txt` - Python依赖
- [x] `setup.py` - 安装配置
- [x] `LICENSE` - 开源许可证
- [x] `.gitignore` - 排除模型文件和缓存
- [x] `download_models.sh` - 模型下载脚本

### ⚠️ 模型文件处理策略

**不要将模型文件打包进仓库！** 原因：

1. **文件大小限制**: 模型文件通常几GB，超出Git和GitHub限制
2. **版本控制效率**: 大文件会让仓库克隆和更新极慢
3. **存储成本**: GitHub对大文件收费，影响开源项目可持续性
4. **许可证问题**: 模型可能有不同的许可证要求

**推荐做法:**
- [x] 使用模型下载脚本自动获取
- [x] 在文档中明确指出模型依赖
- [x] 提供模型下载和配置说明
- [x] 使用.gitignore排除模型文件

### 代码质量要求
- [x] 代码包含完整的错误处理
- [x] 函数和类有详细的文档字符串
- [x] 遵循MCP 1.0.0协议标准
- [x] 支持异步操作
- [x] 包含详细的日志输出

## 🚀 发布步骤

### 1. 准备GitHub仓库

```bash
# 创建新的Git仓库
git init
git add .
git commit -m "Initial commit: MongoDB Natural Language Query MCP Server"

# 推送到GitHub
git remote add origin https://github.com/your-username/mongo-query-mcp.git
git branch -M main
git push -u origin main
```

### 2. 更新配置文件

确保 `mcp.json` 中的URL正确：

```json
{
  "name": "mongo-query-mcp",
  "repository": {
    "type": "git",
    "url": "https://github.com/your-username/mongo-query-mcp.git"
  },
  "homepage": "https://github.com/your-username/mongo-query-mcp",
  "bugs": {
    "url": "https://github.com/your-username/mongo-query-mcp/issues"
  }
}
```

### 3. 创建发布标签

```bash
# 创建版本标签
git tag v1.0.0
git push origin v1.0.0
```

### 4. 提交到mcp.so

访问 [mcp.so](https://mcp.so) 并：

1. **注册账号**
   - 使用GitHub账号登录
   - 完善个人资料

2. **提交服务器**
   - 点击"Submit Server"
   - 输入GitHub仓库URL
   - 系统会自动读取 `mcp.json` 配置

3. **填写额外信息**
   - 添加项目截图（如有）
   - 提供演示视频链接（可选）
   - 填写详细的项目描述

## 📝 mcp.so 平台要求

### 技术要求
- ✅ 必须实现标准MCP协议
- ✅ 支持stdio通信方式
- ✅ 工具定义符合JSON Schema规范
- ✅ 错误处理完善
- ✅ 文档完整清晰

### 质量标准
- ✅ 代码结构清晰
- ✅ 注释完整
- ✅ 测试覆盖（推荐）
- ✅ 示例丰富
- ✅ 文档详细

### 内容规范
- ✅ 项目名称唯一
- ✅ 描述准确详细
- ✅ 标签相关性强
- ✅ 许可证明确
- ✅ 联系方式有效

## 🏷️ 推荐标签

在mcp.so上使用以下标签提高可发现性：

**主要标签:**
- `mongodb`
- `database`
- `nlp`
- `vector-search`
- `ai`

**功能标签:**
- `natural-language`
- `chinese-support`
- `academic-research`
- `text-embedding`
- `similarity-search`

**技术标签:**
- `python`
- `ollama`
- `transformers`
- `pytorch`

## 📊 性能优化建议

### 启动时间优化
```python
# 延迟加载大型模型
class MongoQueryServer:
    def __init__(self):
        self._reranker_model = None
        self._tokenizer = None
    
    @property
    def reranker_model(self):
        if self._reranker_model is None:
            self._load_reranker()
        return self._reranker_model
```

### 内存使用优化
```python
# 使用半精度推理
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
```

### 错误恢复机制
```python
async def query_with_fallback(self, query_text: str):
    try:
        return await self.enhanced_query(query_text)
    except Exception as e:
        logger.warning(f"Enhanced query failed: {e}")
        return await self.basic_query(query_text)
```

## 🎯 用户体验优化

### 1. 清晰的错误消息
```python
if not query_text.strip():
    return [TextContent(
        type='text',
        text="错误：查询文本不能为空。请提供要搜索的内容。"
    )]
```

### 2. 进度指示
```python
async def query_mongo_with_natural_language(self, query_text: str):
    yield TextContent(type='text', text="正在处理查询...")
    
    # 执行查询
    results = await self.perform_search(query_text)
    
    yield TextContent(type='text', text="正在优化结果...")
    
    # 重排序
    optimized_results = await self.rerank_results(results)
```

### 3. 结果预览
```python
def format_result_preview(self, content: str, max_length: int = 200):
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."
```

## 📈 发布后维护

### 1. 监控使用情况
- 关注GitHub仓库的Star和Fork数量
- 收集用户反馈和Issue
- 监控下载和使用统计

### 2. 持续更新
```bash
# 定期更新版本
git tag v1.0.1
git push origin v1.0.1

# 更新mcp.so上的信息
# 修改mcp.json中的版本号
```

### 3. 社区互动
- 及时回复Issues和PR
- 在社交媒体分享使用案例
- 参与MCP社区讨论

## 🔧 故障排除

### 常见发布问题

**1. mcp.json验证失败**
```bash
# 使用在线JSON验证器检查格式
# 确保所有必需字段都已填写
```

**2. 仓库访问问题**
```bash
# 确保仓库是公开的
# 检查GitHub URL拼写正确
```

**3. 依赖安装失败**
```bash
# 测试requirements.txt在干净环境中的安装
pip install -r requirements.txt
```

## 📞 获取帮助

如果在发布过程中遇到问题：

1. **查看官方文档**: [mcp.so/docs](https://mcp.so/docs)
2. **联系支持**: support@mcp.so
3. **社区论坛**: [mcp.so/community](https://mcp.so/community)
4. **GitHub Issues**: 在项目仓库中创建Issue

## ✅ 发布成功标志

发布成功后，您将看到：

- ✅ 服务器出现在mcp.so搜索结果中
- ✅ 项目页面正确显示所有信息
- ✅ 安装说明清晰可行
- ✅ 用户可以正常下载和使用
- ✅ 获得社区反馈和评价

---

**祝您发布成功！** 🎉

如有任何问题，欢迎在项目仓库中创建Issue或联系维护者。

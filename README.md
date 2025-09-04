# MongoDB自然语言查询MCP服务器

一个强大的MCP服务器，支持使用自然语言查询MongoDB数据库，专门为生物入侵研究等学术领域设计。集成向量相似度搜索、AI结果重排序和智能分析功能。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![MCP](https://img.shields.io/badge/MCP-1.0.0-orange.svg)

## ✨ 核心功能

- 🔍 **自然语言查询**: 将自然语言转换为向量进行智能搜索
- 📊 **向量相似度搜索**: 基于余弦相似度的精准文档检索
- 🎯 **AI重排序**: 使用Qwen3-Reranker模型优化搜索结果
- 🤖 **智能分析**: DeepSeek API驱动的专业分析报告
- 🌏 **中文优化**: 专为中文学术文献设计优化
- 📦 **即插即用**: 标准MCP协议，兼容所有MCP客户端

## 🚀 快速开始

### 系统要求

- Python 3.8+
- MongoDB 4.4+
- Ollama (本地模型推理)

### 安装步骤

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置MongoDB**
   ```bash
   # 启动MongoDB服务
   mongod
   ```

3. **安装Ollama并下载模型**
   ```bash
   # 安装Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # 下载嵌入模型
   ollama pull dengcao/Qwen3-Embedding-8B:Q5_K_M
   ```

4. **创建环境配置**
   ```bash
   # 创建 .env 文件
   cat > .env << EOF
   MONGO_URI=mongodb://localhost:27017/
   MONGO_DB_NAME=your_database_name
   MONGO_COLLECTION=your_collection_name
   
   # 可选：AI增强功能
   DEEPSEEK_API_KEY=your_api_key
   DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
   EOF
   ```

### MCP客户端配置

#### Cursor配置
```json
{
  "mcpServers": {
    "mongo-query-server": {
      "command": "python",
      "args": ["start_mcp_server.py"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

#### Claude Desktop配置
```json
{
  "mcpServers": {
    "mongo-query-server": {
      "command": "python",
      "args": ["/path/to/start_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/project"
      }
    }
  }
}
```

## 🛠️ 可用工具

### query_mongo_with_natural_language
使用自然语言查询MongoDB数据库

**参数:**
- `query_text` (必需): 自然语言查询文本
- `db_name` (可选): 数据库名称
- `collection_name` (可选): 集合名称  
- `limit` (可选): 返回结果数量，默认5
- `use_reranker` (可选): 是否使用重排序，默认true
- `enhance_results` (可选): 是否AI增强，默认true

**示例:**
```
查询：外来物种入侵对生态系统的影响
```

### text_to_vector
将文本转换为向量表示

**参数:**
- `text` (必需): 要转换的文本

**示例:**
```
文本：生物入侵监测技术
```

### get_database_info
获取数据库和集合信息

**参数:** 无

## 📊 使用场景

### 学术研究
- 文献检索和相似论文推荐
- 研究趋势分析
- 知识图谱构建

### 政策分析
- 法规文档智能搜索
- 政策关联性分析
- 影响评估报告

### 数据挖掘
- 大规模文档集合分析
- 内容相似度计算
- 智能分类和标注

## 🔧 高级配置

### 自定义模型
```python
# 在代码中修改模型配置
LOCAL_EMBEDDING_MODEL = 'your-custom-model'
```

### 性能优化
- GPU加速推理
- 批量处理优化
- 内存使用控制

### 数据导入
支持批量导入.npy向量文件：
```bash
python 向量输入数据库_批量版.py /path/to/data
```

## 📈 性能特性

- **向量维度**: 支持任意维度嵌入向量
- **查询速度**: 平均响应时间 < 2秒
- **并发支持**: 多客户端同时访问
- **内存优化**: FP16推理降低内存使用
- **错误恢复**: 完善的异常处理机制

## 🌟 特色功能

### 智能重排序
使用Qwen3-Reranker-4B模型对搜索结果进行二次排序，提高相关性。

### AI分析报告
集成DeepSeek API，自动生成专业的分析报告：
- 关键信息提取
- 内容深度分析  
- 专业见解总结
- 结构化输出

### 中文优化
专门针对中文学术文献进行优化：
- 中文分词处理
- 学术术语识别
- 语义理解增强

## 🔍 示例查询

```
用户：使用mongo-query-server查询"入侵物种对农业的影响"

结果：
查询: 入侵物种对农业的影响
数据库: 生物入侵研究, 集合: 研究文献
共找到 156 个相关结果，显示前 5 个:

结果 #1:
相似度评分: 0.8742
相关程度: 高度相关

向量文档元数据信息:
{
  "title": "外来入侵植物对农业生态系统的影响研究",
  "author": "张某某",
  "year": "2023"
}

原文档分段内容:
外来入侵植物通过与作物竞争养分、水分和光照资源，
严重影响农业生产效率。研究表明，某些入侵植物
可使作物产量下降20-40%...

AI整合分析报告:
基于检索到的156篇相关文献，入侵物种对农业的影响
主要体现在以下几个方面：1）直接经济损失，年均
造成农业损失约500亿元...
```

## 📖 项目结构

```
mongo-query-mcp/
├── src/
│   ├── __init__.py
│   ├── mongo_query_server.py    # 核心MCP服务器实现
│   └── start_mcp_server.py      # 服务器启动脚本
├── mcp.json                     # MCP服务器配置文件
├── requirements.txt             # Python依赖
├── setup.py                     # 安装配置
├── .env.example                 # 环境变量示例
└── README.md                    # 项目文档
```

## 🐛 故障排除

### 常见问题

**1. 模型加载失败**
```bash
# 检查Ollama服务状态
ollama list
# 重新下载模型
ollama pull dengcao/Qwen3-Embedding-8B:Q5_K_M
```

**2. 数据库连接失败**
```bash
# 检查MongoDB服务
systemctl status mongod
# 验证连接字符串
mongo mongodb://localhost:27017/
```

**3. MCP连接问题**
- 确保Python路径正确设置
- 检查环境变量配置
- 验证依赖包版本

## 🤝 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Ollama](https://ollama.ai/) - 本地模型推理
- [MongoDB](https://mongodb.com/) - 数据库支持
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP协议
- [Anthropic](https://anthropic.com/) - MCP标准制定

## 📞 联系方式

- 项目地址: [GitHub Repository](https://github.com/your-username/mongo-query-mcp)
- 问题反馈: [Issues](https://github.com/your-username/mongo-query-mcp/issues)
- 邮箱: your.email@example.com

---

**标签**: `#MCP` `#MongoDB` `#NLP` `#VectorSearch` `#AI` `#Chinese` `#AcademicResearch`

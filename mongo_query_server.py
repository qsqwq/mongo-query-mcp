#!/usr/bin/env python3
"""
MongoDB自然语言查询MCP服务器
将自然语言查询MongoDB的功能封装为Model Context Protocol工具
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# 加载环境变量
load_dotenv()

# 本地模型配置
LOCAL_EMBEDDING_MODEL = 'dengcao/Qwen3-Embedding-8B:Q5_K_M'

class MongoQueryServer:
    def __init__(self):
        self.server = Server("mongo-query-server")
        
        # 注册工具列表处理器
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="query_mongo_with_natural_language",
                    description="使用自然语言查询MongoDB数据库，支持向量相似度搜索和结果优化",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string", "description": "自然语言查询文本"},
                            "db_name": {"type": "string", "description": "数据库名称（可选）"},
                            "collection_name": {"type": "string", "description": "集合名称（可选）"},
                            "limit": {"type": "integer", "description": "返回结果数量限制", "default": 5},
                            "use_reranker": {"type": "boolean", "description": "是否使用重排序模型", "default": True},
                            "enhance_results": {"type": "boolean", "description": "是否使用AI优化结果", "default": True}
                        },
                        "required": ["query_text"]
                    }
                ),
                Tool(
                    name="text_to_vector",
                    description="将文本转换为向量表示",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "要转换为向量的文本"}
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="get_database_info",
                    description="获取MongoDB数据库和集合信息",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        # 注册工具处理器
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "query_mongo_with_natural_language":
                return await self.query_mongo_with_natural_language(
                    arguments.get("query_text", ""),
                    arguments.get("db_name"),
                    arguments.get("collection_name"),
                    arguments.get("limit", 5),
                    arguments.get("use_reranker", True),
                    arguments.get("enhance_results", True)
                )
            elif name == "text_to_vector":
                return await self.text_to_vector_tool(arguments.get("text", ""))
            elif name == "get_database_info":
                return await self.get_database_info()
            else:
                raise ValueError(f"未知工具: {name}")

    async def text_to_vector(self, text: str) -> np.ndarray:
        """使用本地模型将文本转换为向量"""
        try:
            import ollama
            import re
            
            # 文本归一化处理
            text = text.lower()  # 统一小写
            text = re.sub(r'[^\w\s]', '', text)  # 去除标点
            text = re.sub(r'\s+', ' ', text).strip()  # 标准化空格
            
            # 生成向量
            response = ollama.embeddings(
                model=LOCAL_EMBEDDING_MODEL,
                prompt=text
            )
            
            # 调试：检查响应格式
            if 'embedding' not in response:
                raise Exception(f"Ollama响应缺少embedding字段: {response}")
                
            vector = np.array(response['embedding'])
            
            # 调试：检查向量类型和形状
            if not isinstance(vector, np.ndarray):
                raise Exception(f"向量不是numpy数组: {type(vector)}")
                
            # 向量归一化处理
            return vector / np.linalg.norm(vector)
        except Exception as e:
            raise Exception(f"本地文本转向量失败: {str(e)}")

    async def enhance_results_with_llm(self, query_text: str, original_docs: List) -> str:
        """使用DeepSeek API优化输出结果，结合原文档内容"""
        DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
        DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL')
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 准备包含原文档内容的详细结果摘要
        results_summary = "\n".join(
            f"结果{i} (相似度:{sim:.4f}):\n"
            f"- 分段编号: {original_doc['chunk_number'] if original_doc else '无'}\n"
            f"- 来源文件: {original_doc['source'] if original_doc else '未知'}\n"
            f"- 内容预览: {original_doc['content'][:200] + '...' if original_doc and len(original_doc['content']) > 200 else original_doc['content'] if original_doc else '无内容'}"
            for i, (doc, sim, original_doc) in enumerate(original_docs, 1)
        )
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的生物入侵研究专家。请基于检索到的实际文档内容，生成关于查询主题的深度整合分析报告。报告要求：\n\n1. 关键信息提取：准确提取文档中的事实数据、时间、地点、影响范围等核心信息\n2. 内容深度分析：分析文档间的关联性、数据一致性、研究趋势\n3. 专业见解：提供基于文档证据的专业判断和风险评估\n4. 结构化输出：使用清晰的章节结构，包括摘要、分析、结论和建议\n5. 准确性：严格基于提供的文档内容，不添加外部知识或假设\n\n输出语言：中文\n报告风格：学术专业，数据驱动"
                },
                {
                    "role": "user",
                    "content": f"查询主题：{query_text}\n\n检索到的相关文档内容（按相似度排序）：\n{results_summary}\n\n请基于以上实际文档内容，生成一份专业的整合分析报告。要求：\n- 严格基于提供的文档内容进行分析\n- 提取关键数据和事实信息\n- 分析不同文档间的关联和一致性\n- 评估信息的完整性和可靠性\n- 提供专业的结论和建议\n- 使用清晰的章节结构组织内容"
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1500
        }
        
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"结果优化失败: {str(e)}")

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        # 调试：检查输入参数类型
        if not isinstance(vec1, np.ndarray):
            raise Exception(f"vec1不是numpy数组: {type(vec1)}, 值: {vec1}")
        if not isinstance(vec2, np.ndarray):
            raise Exception(f"vec2不是numpy数组: {type(vec2)}, 值: {vec2}")
            
        # 调试：检查数组内容
        if vec1.dtype == object:
            raise Exception(f"vec1包含对象类型数据: {vec1}")
        if vec2.dtype == object:
            raise Exception(f"vec2包含对象类型数据: {vec2}")
            
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def load_reranker_model(self):
        """加载Qwen3-Reranker-4B模型"""
        model_name = "C:\\Users\\admin\\.cache\\modelscope\\hub\\models\\dengcao\\Qwen3-Reranker-4B\\models\\Qwen3-Reranker-4B"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # 加载tokenizer和模型
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                pad_token='[PAD]'
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None,
                ignore_mismatched_sizes=True
            )
            model.eval()
            
            return tokenizer, model
        except Exception as e:
            print(f"[警告] 模型加载失败: {str(e)}")
            return None, None

    async def rerank_results(self, query_text: str, results: List, tokenizer, model):
        """使用reranker模型对结果进行重排序"""
        try:
            pairs = []
            for doc, _ in results:
                # 安全访问metadata字段
                metadata = doc.get('metadata', {})
                title = metadata.get('title', '')
                abstract = metadata.get('abstract', '')
                pairs.append((query_text, f"{title} {abstract}"))
            
            with torch.no_grad():
                # 减少最大长度以降低GPU内存使用
                inputs = tokenizer(
                    pairs,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    max_length=256,
                    pad_to_multiple_of=8
                )
                
                # 分批处理以避免内存溢出
                batch_size = 4
                scores = []
                for i in range(0, len(pairs), batch_size):
                    batch_inputs = {k: v[i:i+batch_size].to(model.device) 
                                  for k, v in inputs.items()}
                    batch_scores = model(**batch_inputs, return_dict=True).logits.view(-1,).float()
                    scores.append(batch_scores.cpu())
                
                scores = torch.cat(scores)
        except Exception as e:
            print(f"[警告] 重排序失败: {str(e)}")
            return results
            
        # 将分数转换为numpy数组并归一化到0-1范围
        scores = torch.sigmoid(scores).numpy()
        
        # 更新结果分数(提高reranker权重)
        reranked = [(doc, sim * 0.3 + score * 0.7)
                   for (doc, sim), score in zip(results, scores)]
        
        # 按新分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    async def query_mongo_with_natural_language(
        self,
        query_text: str,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        limit: int = 5,
        use_reranker: bool = True,
        enhance_results: bool = True
    ) -> List[TextContent]:
        """MCP工具：使用自然语言查询MongoDB"""
        client = None
        try:
            # 将文本转换为向量
            query_vector = await self.text_to_vector(query_text)
            
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            default_db = os.getenv('MONGO_DB_NAME', '中国生物入侵研究')
            default_collection = os.getenv('MONGO_COLLECTION', '生物入侵研究')
            
            client = MongoClient(mongo_uri)
            db = client[db_name or default_db]
            collection = db[collection_name or default_collection]
            
            results = []
            for doc in collection.find({}):
                if 'data' in doc:
                    # 处理不同的数据格式：可能是数组或包含vector字段的字典
                    data = doc['data']
                    if isinstance(data, dict) and 'vector' in data:
                        # 如果是字典格式，提取vector字段
                        vector_data = np.array(data['vector'])
                    else:
                        # 如果是数组格式，直接使用
                        vector_data = np.array(data)
                    
                    similarity = self.cosine_similarity(vector_data, query_vector)
                    results.append((doc, similarity))
            
            # 初始排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 使用reranker进行重排序
            if use_reranker:
                tokenizer, model = await self.load_reranker_model()
                if tokenizer and model:
                    results = await self.rerank_results(query_text, results, tokenizer, model)
            
            # 查找对应的原文档分段
            original_docs = []
            for doc, sim in results[:limit]:
                vector_source = doc.get('source', '')
                if vector_source.startswith('ias_cn_') and vector_source.endswith('.npy'):
                    chunk_number = vector_source.replace('ias_cn_', '').replace('.npy', '')
                    try:
                        chunk_number = int(chunk_number)
                        original_doc = collection.find_one({
                            'chunk_number': chunk_number,
                            'file_type': 'markdown_chunk'
                        })
                        original_docs.append((doc, sim, original_doc))
                    except ValueError:
                        original_docs.append((doc, sim, None))
                else:
                    original_docs.append((doc, sim, None))
            
            # 构建结果输出
            output_content = []
            
            # 添加查询信息
            output_content.append(TextContent(
                type='text',
                text=f"查询: {query_text}\n数据库: {db.name}, 集合: {collection.name}\n共找到 {len(results)} 个相关结果，显示前 {limit} 个:\n\n"
            ))
            
            # 添加每个结果
            for i, (doc, sim, original_doc) in enumerate(original_docs, 1):
                result_text = f"结果 #{i}:\n"
                result_text += f"相似度评分: {sim:.4f}\n"
                result_text += f"相关程度: {'高度相关' if sim > 0.8 else '中等相关' if sim > 0.5 else '低相关'}\n\n"
                
                result_text += "向量文档元数据信息:\n"
                # 安全访问metadata字段
                metadata = doc.get('metadata', {})
                result_text += json.dumps(metadata, ensure_ascii=False, indent=2) + "\n\n"
                
                if original_doc:
                    result_text += f"原文档分段内容 (分段编号: {original_doc['chunk_number']}):\n"
                    content_preview = original_doc['content'][:200] + "..." if len(original_doc['content']) > 200 else original_doc['content']
                    result_text += f"{content_preview}\n\n"
                    result_text += f"完整内容长度: {len(original_doc['content'])} 字符\n"
                    result_text += f"来源文件: {original_doc['source']}\n"
                else:
                    result_text += "未找到对应的原文档分段\n"
                
                result_text += "-" * 50 + "\n\n"
                
                output_content.append(TextContent(
                    type='text',
                    text=result_text
                ))
            
            # 使用API优化结果展示
            if enhance_results and len(results) > 0:
                try:
                    enhanced_report = await self.enhance_results_with_llm(query_text, original_docs[:limit])
                    if enhanced_report:
                        output_content.append(TextContent(
                            type='text',
                            text=f"AI整合分析报告:\n{enhanced_report}\n"
                        ))
                except Exception as e:
                    output_content.append(TextContent(
                        type='text',
                        text=f"结果优化失败: {str(e)}\n"
                    ))
            
            return output_content
            
        except Exception as e:
            return [TextContent(
                type='text',
                text=f"查询失败: {str(e)}"
            )]
        finally:
            if client is not None:
                client.close()

    async def text_to_vector_tool(self, text: str) -> List[TextContent]:
        """MCP工具：将文本转换为向量"""
        try:
            vector = await self.text_to_vector(text)
            return [TextContent(
                type='text',
                text=f"文本: {text}\n向量维度: {vector.shape}\n向量值 (前10维): {vector[:10].tolist()}"
            )]
        except Exception as e:
            return [TextContent(
                type='text',
                text=f"文本转向量失败: {str(e)}"
            )]

    async def get_database_info(self) -> List[TextContent]:
        """MCP工具：获取数据库信息"""
        try:
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            client = MongoClient(mongo_uri)
            
            db_info = []
            for db_name in client.list_database_names():
                db = client[db_name]
                collections = db.list_collection_names()
                db_info.append(f"数据库: {db_name}")
                db_info.append(f"集合: {', '.join(collections)}")
                db_info.append("")
            
            client.close()
            
            return [TextContent(
                type='text',
                text="\n".join(db_info)
            )]
        except Exception as e:
            return [TextContent(
                type='text',
                text=f"获取数据库信息失败: {str(e)}"
            )]

async def main():
    """启动MCP服务器"""
    server = MongoQueryServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

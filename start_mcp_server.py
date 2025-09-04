#!/usr/bin/env python3
"""
MongoDB自然语言查询MCP服务器启动脚本
"""
import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mongo_query_server import main

if __name__ == "__main__":
    print("启动MongoDB自然语言查询MCP服务器...")
    print("服务器名称: mongo-query-server")
    print("可用工具:")
    print("  - query_mongo_with_natural_language: 使用自然语言查询MongoDB")
    print("  - text_to_vector: 将文本转换为向量")
    print("  - get_database_info: 获取数据库信息")
    print("\n按Ctrl+C停止服务器")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)
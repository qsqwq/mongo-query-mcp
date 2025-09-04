#!/bin/bash
# download_models.sh
# 模型下载脚本

echo "开始下载必要的模型..."

# 检查Ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "错误: 未找到Ollama，请先安装Ollama"
    echo "安装命令: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# 下载嵌入模型
echo "下载文本嵌入模型: dengcao/Qwen3-Embedding-8B:Q5_K_M"
ollama pull dengcao/Qwen3-Embedding-8B:Q5_K_M

# 提示用户下载reranker模型
echo "请手动下载Qwen3-Reranker-4B模型并放置在合适的位置"
echo "或者修改src/mongo_query_server.py中的模型路径"

echo "模型下载完成!"
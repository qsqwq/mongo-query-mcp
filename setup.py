from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mongo-query-mcp",
    version="1.0.0",
    description="MongoDB自然语言查询MCP服务器",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pymongo>=4.6.2",
        "python-dotenv>=1.0.0", 
        "numpy>=1.26.0",
        "torch>=2.7.1",
        "requests>=2.32.3",
        "transformers>=4.55.0",
        "ollama>=0.5.1",
        "mcp>=1.0.0",
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
        "aiohttp>=3.9.0",
        "psutil>=5.9.0",
        "python-dateutil>=2.9.0",
        "sentence-transformers>=5.0.0"
    ],
    entry_points={
        'console_scripts': [
            'mongo-query-mcp=start_mcp_server:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.8',
    keywords="mongodb, mcp, nlp, vector-search, ai",
    url="https://github.com/your-username/mongo-query-mcp",
    project_urls={
        "Bug Reports": "https://github.com/your-username/mongo-query-mcp/issues",
        "Source": "https://github.com/your-username/mongo-query-mcp",
        "Documentation": "https://github.com/your-username/mongo-query-mcp/blob/main/DATA_PREPARATION.md",
    },
)

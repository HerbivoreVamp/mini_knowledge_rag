# Mini Knowledge RAG

基于 LangChain + LangGraph 的本地知识库 RAG 问答系统。

## 功能

- Hierarchical Retrieval
- Parent Document Retrieval
- Hybrid Search（Dense + BM25，RRF 融合）
- 检索链（Dense/BM25 混合召回 → Rerank → Parent 扩展）
- Agent Tool 调用知识库
- 本地模型运行
- SQLite 对话状态持久化

## 项目结构

```
main.py                      # 程序入口，交互式问答
README.md
requirements.txt             # 依赖
pytest.ini                   # pytest配置
backend/
├── config/                  # 项目配置
│   ├── settings.py          # 模型、路径等配置
│   ├── model.py             # 模型工厂（Embedding + LLM）
│   ├── prompts.py           # Prompt 模板
│   ├── embedding.py         # Embedding 模型配置
│   └── reranker.py          # Reranker 配置
│
├── core/                    # 核心模块
│   ├── utils.py             # 通用工具函数
│   ├── exceptions.py        # 统一异常定义
│   └── logger.py            # 日志配置
│
├── ingestion/               # 文档导入模块
│   ├── service.py           # 导入流程编排（load → split → store → embed → save）
│   ├── loader.py            # Markdown 文档加载
│   ├── splitter.py          # 文本切分
│   ├── hierarchical.py      # 文档分层导入（parent/child split + store + embed）
│   └── utils.py             # chunk_id/doc_id 生成
│
├── storage/                 # 存储模块
│   ├── sqlite_docstore.py   # SQLite 存储 parent/child 文档（BaseStore 实现）
│   ├── docstore.py          # 旧版 JSON 存储（已被 sqlite_docstore 替代）
│   └── vectorstore.py       # FAISS 向量库构建、加载、删除
│
├── retrieval/                    # 检索模块
│   ├── factory.py                # 检索器组装工厂（Dense → Hybrid → Rerank → Hierarchical）
│   ├── dense_retriever.py        # 基础向量检索器（VectorStore 封装）
│   ├── hybrid_retriever.py       # 混合检索器（Dense + BM25，RRF 融合）
│   ├── rerank_retriever.py       # 重排序检索器（装饰器模式）
│   ├── hierarchical_retriever.py # 分层检索器（child → parent 扩展）
│   ├── search.py                 # 检索工具封装（LangChain Tool）
│   └── fusion/
│       └── rrf.py                # Reciprocal Rank Fusion 融合算法
│
├── agent/                   # Agent 模块
│   └── agent.py             # RAG Agent 创建
│
├── application/             # 应用层
│   ├── service.py           # 生成服务封装
│   └── chat.py              # 对话生成
│
├── data/                    # 数据目录
│   ├── document/            # Markdown 知识文档
│   ├── database/            # 数据库（vectorstore + parent_store + child_store）
│   └── memory/              # LangGraph 对话状态持久化
└── logs/                    # 运行日志

tests/
├── test_xxx                 # 各类单元测试
└── ...
```

## 数据流

**文档导入**

```
    Markdown 文件
      │
    Loader
      │
ingest_documents (parent/child split)
      │
      ├──────────────────+
      │                  │
Parent Document    Child Document
      │                  ├──────────────────+
      │                  │                  │
SqliteDocStore     SqliteDocStore        Embedding
(parent_store)     (child_store)            │
                                          FAISS
```

**查询问答**

```
用户问题
 |
Agent
 |
Retriever Tool
 |
 ├─────────────────────+
 |                     |
DenseRetriever   BM25Retriever
 |                     |
 ├─────────────────────+
HybridRetriever (RRF 融合)
 |
RerankRetriever (Reranker 重排序)
 |
HierarchicalRetriever (Parent Document Lookup)
 |
Context
 |
LLM
```
## Retrieval Pipeline
```
Indexing:
Markdown
 ↓
Loader
 ↓
Parent/Child Split
 ↓
Embedding
 ↓
FAISS(child)
 ↓
SqliteDocStore(parent)


Query:
Question
 ↓
DenseRetriever (FAISS similarity search)
 +
BM25Retriever (sparse retrieval)
 ↓
HybridRetriever (RRF 融合)
 ↓
RerankRetriever (BGE Reranker)
 ↓
HierarchicalRetriever (Parent lookup)
 ↓
LLM
```
对话状态通过 SQLite Checkpoint 持久化。

## 技术栈

- **框架**:LangChain + LangGraph
- **Embedding**: BAAI/bge-small-zh-v1.5 (HuggingFace)
- **向量库**: FAISS
- **文档存储**: SQLite（SqliteDocStore，parent/child 文档独立存储）
- **稀疏检索**: BM25
- **Retriever**: DenseRetriever + HybridRetriever(RRF) + RerankRetriever + HierarchicalRetriever 检索链
- **LLM**: LangChain ChatModel（支持 OpenAI Compatible API）
- **记忆**: SQLite (LangGraph Checkpoint)

## 使用

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 .env 文件中的 embedding 模型路径 rerank 模型路径 和 LLM 参数
# 如果想调整读取文档的位置 可以配置config.settings的内容

# 运行
python main.py

# 运行测试
pytest
```

按提示选择：1 导入文档 / 2 查询知识库 / 3 删除数据库和记忆 / 4 仅删除记忆。
将 Markdown .md文件放入 data/document/ 目录后运行导入流程 支持嵌套文件夹。

## 环境

依赖见 `requirements.txt`。

## 特点

- 支持本地 Embedding 模型
- FAISS 本地向量检索
- SQLite 持久化 parent/child 文档（SqliteDocStore）
- BM25 稀疏检索语料从 child_store 读取，不依赖 FAISS 内部 docstore
- 检索结果保留文档来源信息
- 使用 Agent Tool 动态调用知识库
- SQLite 持久化保存对话状态
- 统一异常处理（RAGError 异常体系）
- 完善的日志记录
- 支持基于配置切换不同知识库索引，每个知识库独立维护 FAISS 向量索引和 parent/child 文档存储
## Logging

项目使用 Python logging 记录运行状态，包括：

- 配置加载
- 模型初始化
- 文档处理流程
- 向量库构建与保存
- Agent 工具调用
- 模型生成过程

日志同时支持控制台输出和文件滚动记录。

## 下一步工作 
- [x] 配置系统重构
- [x] 日志系统完善
- [x] 统一异常处理（error handling）
- [x] 封装 ingestion/retrieval/generation 模块
- [x] 优化项目目录结构
- [x] 增加 pytest 单元测试
- [x] 修改vectorstore的创建和导入 文档的导入
- [x] 完成HierarchicalRetriever
- [x] 完成Reranker
- [x] 完成Retriever模块化 为hybridsearch准备
- [x] 完成hybridsearch
- [x] SqliteDocStore 替代 JsonDocStore，parent/child 文档独立 SQLite 存储
- [ ] 增加基础 evaluation 流程
  - [ ] RAGAS 评测
  - [ ] 自定义检索准确率测试
- [ ] 增加可选的SemanticChunker
- [ ] 完善 README 和项目文档
- [ ] Docker 化部署（可选）
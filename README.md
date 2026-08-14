# Mini Knowledge RAG

基于 LangChain + LangGraph 的本地知识库 RAG 问答系统。

## 功能

- 导入 Markdown 文档到 FAISS 向量库
- 基于向量检索的智能问答
- 对话记忆（SQLite 持久化）

## 项目结构

```
backend/
├── main.py                  # 程序入口，交互式问答
│
├── config/                  # 项目配置
│   ├── settings.py          # 模型、路径等配置
│   ├── model.py             # 模型工厂（Embedding + LLM）
│   ├── prompts.py           # Prompt 模板
│   └── embedding.py         # Embedding 模型配置
│
├── core/                    # 核心模块
│   ├── exceptions.py        # 统一异常定义
│   └── logger.py            # 日志配置
│
├── ingestion/               # 文档导入模块
│   ├── service.py           # 导入流程编排（load → parent_retriever → embed → save）
│   ├── loader.py            # Markdown 文档加载
│   ├── splitter.py          # 文本切分
│   └── hierarchical.py      # ParentDocumentRetriever 分层检索
│
├── storage/                 # 存储模块
│   └── vectorstore.py       # FAISS 向量库构建、加载、删除
│
├── retrieval/               # 检索模块
│   └── search.py            # 检索工具封装
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
│   ├── database/            # 向量库
│   └── memory/              # LangGraph 对话状态持久化
└── logs/                    # 运行日志

tests/
├── test_xxx                 # 各类单元测试
└── ......
```

## 数据流

**文档导入**

```
    Markdown 文件
      │
    Loader
      │
ParentDocumentRetriever
      │
      ├────────────────+
      │                │
      │                │
Parent Document     Child Document
      │                │
      │                │
JsonDocStore        Embedding
                       │
                       │
                     FAISS
```

**查询问答**

```
用户问题
  │
Agent
  │
Retriever Tool
  │
FAISS（child chunk 向量搜索）
  │
Parent Document (parent document lookup)
  │
Context
  │
LLM
  │
Answer
```

对话状态通过 SQLite Checkpoint 持久化。

## 技术栈

- **框架**:LangChain + LangGraph
- **Embedding**: BAAI/bge-small-zh-v1.5 (HuggingFace)
- **向量库**: FAISS
- **Retriever**: LangChain ParentDocumentRetriever
- **LLM**: LangChain ChatModel（支持 OpenAI Compatible API）
- **记忆**: SQLite (LangGraph Checkpoint)

## 使用

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 .env 文件中的 embedding 模型路径和 LLM 参数
# 如果想调整读取文档的位置 可以配置config.settings的内容

# 运行
python backend/main.py

# 运行测试
pytest
```

按提示选择：1 导入文档 / 2 查询知识库 / 3 删除数据库。
将 Markdown .md文件放入 data/document/ 目录后运行导入流程。

## 环境

依赖见 `requirements.txt`。

## 特点

- 支持本地 Embedding 模型
- FAISS 本地向量检索
- 检索结果保留文档来源信息
- 使用 Agent Tool 动态调用知识库
- SQLite 持久化保存对话状态
- 统一异常处理（RAGError 异常体系）
- 完整日志记录
- 支持基于配置切换不同知识库索引，每个知识库独立维护 FAISS 向量索引和 Parent Document 存储
## Logging

项目使用 Python logging 记录运行状态，包括：

- 配置加载
- 模型初始化
- 文档处理流程
- 向量库构建与保存
- Agent 工具调用
- 模型生成过程

日志默认输出到控制台（或 logs 文件夹）。

## 下一步工作 
- [x] 配置系统重构
- [x] 日志系统完善
- [x] 统一异常处理（error handling）
- [x] 封装 ingestion/retrieval/generation 模块
- [x] 优化项目目录结构
- [x] 增加 pytest 单元测试
- [x] 修改vectorstore的创建和导入 文档的导入
- [x] HierarchicalRetriever
  - [x] ParentDocumentRetriever 集成
  - [x] JsonDocStore 持久化
  - [x] Parent/Child 文档检索流程
  - [ ] 异常处理完善
  - [ ] pytest测试完善
- [ ] 增加SemanticChunker
- [ ] 增加基础 evaluation 流程
  - [ ] RAGAS 评测
  - [ ] 自定义检索准确率测试
- [ ] 完善 README 和项目文档
- [ ] Docker 化部署（可选）
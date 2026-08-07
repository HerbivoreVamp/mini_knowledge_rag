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
│   └── prompts.py           # Prompt 模板
│
├── utils/                   # 通用工具
│   └── logger.py            # 日志配置
│
├── ingestion/               # 文档处理流程
│   ├── loader.py            # Markdown 文档加载
│   ├── splitter.py          # 文本切分
│   └── embedding.py         # 文档向量化
│
├── retrieval/               # 检索模块
│   ├── vectorstore_manage.py # FAISS 向量库管理
│   ├── search.py            # 检索工具封装
│   └── delete.py            # 向量库删除
│
├── generation/              # 生成模块
│   └── chat.py              # Agent 对话封装
│
├── database/                # FAISS 向量库存储
├── document/                # Markdown 知识文档
├── memory/                  # LangGraph 对话状态持久化
└── logs/                    # 运行日志
```

## 数据流

**文档导入**

```
Markdown 文件
  │
Loader（文档加载）
  │
Splitter（文本切分）
  │
Embedding（向量化）
  │
FAISS（向量存储）
```

**查询问答**

```
用户问题
  │
Agent
  │
Retriever Tool
  │
FAISS
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
- **Embedding**: BGE-small-zh-v1.5 (HuggingFace)
- **向量库**: FAISS
- **LLM**: LangChain ChatModel（支持 OpenAI Compatible API）
- **记忆**: SQLite (LangGraph Checkpoint)

## 使用

```bash
# 安装依赖
pip install -r requirements.txt

# 修改 config/settings.py 中的 embedding 模型路径和 LLM 配置

# 运行
python backend/main.py
```

按提示选择：1 导入文档 / 2 查询知识库 / 3 删除数据库。
将 Markdown .md文件放入 document/ 目录后运行导入流程。

## 环境

依赖见 `requirements.txt`。

## 特点

- 支持本地 Embedding 模型
- FAISS 本地向量检索
- 检索结果保留文档来源信息
- 使用 Agent Tool 动态调用知识库
- SQLite 持久化保存对话状态
- 完整日志记录

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
- [x] 配置重构
- [x] 日志系统
- [ ] 异常处理（统一 error handling）
- [ ] 把 ingestion/retrieval/generation 封装成 service/class
- [ ] 加 pytest 测试
- [ ] 做 evaluation（RAGAS 或自己的评测）
- [ ] 再进入 Agent workflow
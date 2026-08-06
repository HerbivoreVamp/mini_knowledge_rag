# Mini Knowledge RAG

基于 LangChain + LangGraph 的本地知识库 RAG 问答系统。

## 功能

- 导入 Markdown 文档到 FAISS 向量库
- 基于向量检索的智能问答
- 对话记忆（SQLite 持久化）

## 项目结构

```
backend/
├── main.py              # 入口，交互式菜单
├── config/              # 配置
│   ├── settings.py      # 模型、路径等集中配置
│   └── prompts.py       # 系统提示词
├── ingestion/           # 文档导入
│   ├── loader.py        # 加载 .md 文件
│   ├── splitter.py      # 文本分块
│   ├── embedding.py     # HuggingFace Embedding
│   └── delete.py        # 删除向量库
├── retrieval/           # 检索
│   ├── vectorstore_manage.py  # FAISS 构建/保存/加载
│   └── search.py        # 检索 Tool 封装
├── generation/          # 对话生成
│   └── chat.py          # Agent 对话封装
├── database/            # 向量库存储
├── document/            # 知识文档
└── memory/              # 对话记忆持久化
```

## 数据流

```
Markdown 文件
     |
  Loader(加载)
     |
  Splitter(分块)
     |
  Embedding(向量化)
     |
  FAISS(存储/检索)
     |
  Retriever Tool(检索工具)
     |
   Agent（调度）
     |
     ├── Retriever Tool → FAISS
     |
     ├── LLM（生成回答）
     |
     └── SQLite Checkpoint（状态持久化）
```

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

# 修改 config/settings_init.py 中的嵌入(embedding)模型路径和 LLM 配置

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

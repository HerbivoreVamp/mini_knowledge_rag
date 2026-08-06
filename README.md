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

## 架构图
```
Markdown
   |
Loader
   |
Splitter
   |
Embedding
   |
FAISS
   |
Retriever Tool
   |
Agent
   |
LLM
   |
SQLite Memory
```

## 技术栈

- **框架**: LangChain + LangGraph(SQLite)
- **Embedding**: BGE-small-zh-v1.5 (HuggingFace)
- **向量库**: FAISS
- **LLM**: 通过 OpenAI 兼容 API 调用
- **记忆**: SQLite (LangGraph Checkpoint)

## 使用

```bash
# 安装依赖
pip install -r requirements.txt

# 完成embedding的路径配置
# 见main.py

# 完成model的配置
# 见main.py

# 运行
python backend/main.py
```

按提示选择：1 导入文档 / 2 查询知识库 / 3 删除数据库。


也可以进入小模块运行测试
## 环境

依赖见 `requirements.txt`。


## Roadmap

- [x] Markdown RAG
- [x] FAISS persistence
- [x] SQLite conversation memory

- [ ] FastAPI interface
- [ ] Reranker
- [ ] Multi knowledge base
- [ ] LangGraph workflow
- [ ] Docker deployment
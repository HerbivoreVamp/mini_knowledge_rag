class RAGError(Exception):
    """基础异常"""


class LoaderError(RAGError):
    pass


class EmbeddingError(RAGError):
    pass


class RetrievalError(RAGError):
    pass


class SplitError(RAGError):
    pass


class VectorStoreError(RAGError):
    pass


class VectorStoreLoadError(RAGError):
    pass


class VectorStoreNotFoundError(RAGError):
    pass


class VectorStoreDeleteError(RAGError):
    pass


class GenerationError(RAGError):
    pass


class AgentError(RAGError):
    pass


class LLMError(RAGError):
    pass


class DocStoreError(RAGError):
    """文档存储相关异常"""
    pass

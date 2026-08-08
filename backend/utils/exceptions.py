class RAGError(Exception):
    """基础异常"""


class LoaderError(RAGError):
    pass


class EmbeddingError(RAGError):
    pass


class RetrievalError(RAGError):
    pass

from pathlib import Path

from sentence_transformers import CrossEncoder

from core.logger import logger
from core.exceptions import RerankerError


class BGEReranker:
    def __init__(self, model_name_or_path: Path, device: str):
        self.model = CrossEncoder(
            model_name_or_path=str(model_name_or_path),
            max_length=1024,
            device=device
        )

    def rerank(
            self,
            query,
            docs,
            top_k=5
    ):
        pairs = [
            (
                query,
                doc.page_content
            )
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, _ in ranked[:top_k]
        ]


def init_reranker_model(dir_name: Path, model_name: str, device="cpu") -> BGEReranker:
    model_name_or_path = dir_name / model_name
    try:
        reranker = BGEReranker(model_name_or_path, device=device)
    except Exception as e:
        logger.exception(f"rerank模型加载失败 model_name={model_name}")
        raise RerankerError(f"rerank模型加载失败: {model_name}") from e
    return reranker

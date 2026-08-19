from sentence_transformers import CrossEncoder


def create_reranker(model_name_or_path: str):
    reranker = CrossEncoder(
        model_name_or_path,
        max_length=1024
    )
    return reranker

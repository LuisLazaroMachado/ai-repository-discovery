from functools import lru_cache


@lru_cache(maxsize=1)
def get_state():
    """Recomendador preparado una sola vez para todos los tests."""
    from src.recommender import load_catalog, prepare_recommender

    return prepare_recommender(load_catalog())

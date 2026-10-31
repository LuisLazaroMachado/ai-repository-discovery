import numpy as np

from src.recommender import METHODS, recommend, recommend_by_preferences, top_indices
from tests.helpers import get_state

QUERY = "huggingface/transformers"


def test_every_method_returns_ranked_rows_without_the_query():
    state = get_state()
    for method in METHODS:
        result = recommend(QUERY, method, top_n=7, recommender=state)
        assert len(result) == 7, method
        assert QUERY not in set(result["Repositorio recomendado"]), method
        assert list(result["Posición"]) == list(range(1, 8)), method


def test_all_methods_share_one_schema():
    state = get_state()
    schemas = {
        tuple(sorted(set(recommend(QUERY, m, 3, state).columns) - {"Score híbrido"})) for m in METHODS
    }
    assert len(schemas) == 1


def test_scores_are_sorted_descending():
    state = get_state()
    for method in METHODS:
        scores = recommend(QUERY, method, top_n=10, recommender=state)["Score"].tolist()
        assert scores == sorted(scores, reverse=True), method


def test_popularity_baseline_does_not_depend_on_the_query():
    state = get_state()
    popularity = np.asarray(state["popularity_score"])
    obscure = [int(i) for i in np.argsort(popularity)[:2]]  # fuera del top de popularidad
    names = state["catalog"]["nombre"]
    first = recommend(names[obscure[0]], "popularity", 10, state)["Repositorio recomendado"].tolist()
    second = recommend(names[obscure[1]], "popularity", 10, state)["Repositorio recomendado"].tolist()
    assert first == second


def test_unknown_repository_returns_empty_frame():
    assert recommend("no-existe/repo", "hybrid", 5, get_state()).empty


def test_ranking_matches_top_indices():
    state = get_state()
    query_idx = int(state["catalog"].index[state["catalog"]["nombre"] == QUERY][0])
    expected = [state["catalog"].at[i, "nombre"] for i in top_indices(state, "hybrid", query_idx, 5)]
    assert recommend(QUERY, "hybrid", 5, state)["Repositorio recomendado"].tolist() == expected


def test_preferences_respect_every_filter():
    state = get_state()
    result = recommend_by_preferences(
        language="Python", topics=["computer-vision"], min_stars=1000,
        max_days_without_commit=180, top_n=10, recommender=state,
    )
    assert 0 < len(result) <= 10
    assert (result["Lenguaje"] == "Python").all()
    assert (result["Estrellas"] >= 1000).all()
    assert (result["Días sin commit"] <= 180).all()


def test_preferences_with_impossible_filters_return_empty():
    result = recommend_by_preferences(min_stars=10**9, recommender=get_state())
    assert result.empty


def test_cluster_names_come_from_the_catalog():
    state = get_state()
    assert set(state["cluster_names"]) == set(state["catalog"]["cluster_kmeans"].astype(int))
    assert all(isinstance(name, str) and name for name in state["cluster_names"].values())

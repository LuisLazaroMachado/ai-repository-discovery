import math

import numpy as np
import pandas as pd

from src import common as c
from src.evaluation import (
    HYBRID_NAME, CONTENT_NAME, POPULARITY_NAME, average_precision_at_k, build_relevance,
    ndcg_at_k, precision_at_k, recall_at_k, run_evaluation,
)
from tests.helpers import get_state

# Mismas definiciones del notebook 04: 3 relevantes (0, 2, 4) y 5 recomendados con aciertos en las posiciones 1 y 3.
MASK = np.array([True, False, True, False, True, False])
RECOMMENDED = [0, 1, 2, 3, 5]


def test_metrics_on_a_hand_computed_case():
    assert math.isclose(precision_at_k(RECOMMENDED, MASK, 5), 0.4)
    assert math.isclose(recall_at_k(RECOMMENDED, MASK, 5), 2 / 3)
    assert math.isclose(average_precision_at_k(RECOMMENDED, MASK, 5), (1 + 2 / 3) / 3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(ndcg_at_k(RECOMMENDED, MASK, 5), dcg / idcg)


def test_metrics_with_no_relevant_items_are_zero():
    empty = np.zeros(6, dtype=bool)
    assert recall_at_k(RECOMMENDED, empty, 5) == 0.0
    assert average_precision_at_k(RECOMMENDED, empty, 5) == 0.0
    assert ndcg_at_k(RECOMMENDED, empty, 5) == 0.0


def test_relevance_rule_depends_on_the_number_of_query_topics():
    catalog = pd.DataFrame({"topics_list": [["a", "b", "c"], ["a", "b"], ["a"], ["x"], []]})
    relevance = build_relevance(catalog)
    # la consulta 0 tiene 3 topics: hay que compartir 2, así que solo el repositorio 1 es relevante
    assert relevance.mask(0).tolist() == [False, True, False, False, False]
    # la consulta 2 tiene 1 topic: basta compartir 1 (comparten 'a' los repositorios 0 y 1)
    assert relevance.mask(2).tolist() == [True, True, False, False, False]
    # sin topics significativos no hay candidatos relevantes
    assert not relevance.mask(4).any()


def test_reproduces_the_metrics_published_in_notebook_04():
    summary = run_evaluation(get_state())["summary"].set_index("model")
    published = {POPULARITY_NAME: 0.2058, CONTENT_NAME: 0.7652, HYBRID_NAME: 0.7726}
    for model, value in published.items():
        assert abs(summary.at[model, "Precision@10"] - value) < 5e-4, model


def test_saved_artifact_matches_a_fresh_evaluation():
    fresh = run_evaluation(get_state())["summary"].set_index("model")
    saved = pd.read_csv(c.EVAL_SUMMARY_PATH).set_index("model")
    for model in fresh.index:
        assert abs(fresh.at[model, "NDCG@10"] - saved.at[model, "NDCG@10"]) < 2e-3, model

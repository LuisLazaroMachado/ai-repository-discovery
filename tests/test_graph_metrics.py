import pandas as pd

from src.common import CATALOG_PATH, EDGE_LIST_PATH
from src.graph_metrics import compute_graph_metrics


def test_ranks_are_global_and_pagerank_matches_published_top():
    catalog = pd.read_csv(CATALOG_PATH)
    table = compute_graph_metrics(pd.read_csv(EDGE_LIST_PATH), catalog)
    assert len(table) == 1861
    assert table.iloc[0]["nombre"] == "rasbt/LLMs-from-scratch"
    # la posición por estrellas es contra los 3034 repositorios, no contra un subconjunto
    thinc = table.set_index("nombre").loc["explosion/thinc"]
    assert thinc["rank_stars"] == int((catalog["stars"] > thinc["stars"]).sum()) + 1
    assert table["rank_stars"].max() > 2000
    assert table["pagerank"].is_monotonic_decreasing

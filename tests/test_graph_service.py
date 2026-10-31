import pytest

pytest.importorskip("pyvis")

from src.graph_service import build_interactive_subgraph, load_centrality_ranking, load_graph_edges  # noqa: E402


def test_graph_artifacts_load():
    assert len(load_graph_edges()) > 20000
    assert not load_centrality_ranking().empty


def test_subgraph_for_a_known_repository():
    payload = build_interactive_subgraph("tensorflow/tensorflow", max_neighbors=8, min_weight=4)
    assert payload["html"].strip()
    assert 0 < len(payload["neighbors_table"]) <= 8
    assert payload["graph_stats"]["subgrafo_nodos"] >= 2


def test_unknown_repository_returns_empty_payload():
    assert build_interactive_subgraph("no-existe/repo")["html"] == ""

"""Métricas de grafo sobre TODO el componente principal (sin pyvis, se puede probar sin red).

Corrige dos limitaciones del notebook 05: el ranking por estrellas se calculaba solo entre los 200
repositorios con mayor weighted degree (no entre los 3034), y la betweenness se calculaba sobre ese
subgrafo de 200 nodos. Aquí PageRank y betweenness se calculan sobre los 1861 nodos del grafo y las
posiciones por estrellas se comparan contra el catálogo completo.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.src), str(row.dst), weight=float(row.weight), distance=1.0 / float(row.weight))
    return graph


def compute_graph_metrics(edges: pd.DataFrame, catalog: pd.DataFrame, alpha: float = 0.85) -> pd.DataFrame:
    """Una fila por repositorio del grafo, ordenada por PageRank."""
    graph = build_graph(edges)
    pagerank = nx.pagerank(graph, weight="weight", alpha=alpha)
    betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    weighted_degree = dict(graph.degree(weight="weight"))

    base = catalog[["nombre", "stars"]].copy()
    base["repo_idx"] = range(len(base))
    base["rank_stars"] = base["stars"].rank(ascending=False, method="min").astype(int)
    table = pd.DataFrame(
        {
            "nombre": list(pagerank),
            "weighted_degree": [weighted_degree[n] for n in pagerank],
            "pagerank": list(pagerank.values()),
            "betweenness": [betweenness[n] for n in pagerank],
        }
    ).merge(base, on="nombre", how="left")
    table["stars"] = table["stars"].astype(int)
    table["repo_idx"] = table["repo_idx"].astype(int)
    table["rank_pagerank"] = table["pagerank"].rank(ascending=False, method="min").astype(int)
    table = table.sort_values("pagerank", ascending=False).reset_index(drop=True)
    return table[["repo_idx", "nombre", "stars", "weighted_degree", "pagerank", "betweenness", "rank_pagerank", "rank_stars"]]


def graph_summary(edges: pd.DataFrame, table: pd.DataFrame, total_repositories: int) -> pd.DataFrame:
    graph = build_graph(edges)
    components = sorted((len(x) for x in nx.connected_components(graph)), reverse=True)
    spearman = table["pagerank"].corr(table["stars"], method="spearman")
    rows = [
        ("Repositorios en el catálogo", int(total_repositories)),
        ("Nodos del componente principal", int(graph.number_of_nodes())),
        ("Aristas", int(graph.number_of_edges())),
        ("Densidad", round(nx.density(graph), 4)),
        ("Componentes conectados en el archivo de aristas", len(components)),
        ("Correlación de Spearman PageRank vs estrellas", round(float(spearman), 3)),
    ]
    return pd.DataFrame([(m, str(v)) for m, v in rows], columns=["métrica", "valor"])

"""Regenera los artefactos de recomendación y evaluación que consume el dashboard.

    python -m src.build_artifacts            # reconstruye y escribe
    python -m src.build_artifacts --check    # solo verifica que los artefactos publicados coinciden

Entrada: data/processed/repos_clean.csv (fotografía de datos de GitHub, ver data/README.md).
Salida (artifacts/recommendation/): repos_clustered.csv, cluster_profile.csv,
offline_evaluation_{summary,detail}.csv, strong_cases.csv, failure_cases.csv,
evaluation_robustness.csv, signal_ablation.csv, data_quality_impact.csv.

Grafo (artifacts/graph/): la lista de aristas la genera el notebook 05; aquí se recalculan
centrality_ranking.csv, pagerank_vs_stars.csv y graph_summary.csv sobre todo el componente principal
y la figura reports/graph/ranking_comparison.png. Las figuras de EDA, PCA y K-means las generan los notebooks 01-03.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src import common as c
from src.evaluation import (
    build_cluster_profile,
    run_data_quality_experiment,
    run_evaluation,
    run_signal_ablation,
)
from src.graph_metrics import compute_graph_metrics, graph_summary
from src.features import PCA_COMPONENTS, add_clusters, name_clusters, prepare_frame
from src.recommender import prepare_recommender

PCA_COLUMNS = [f"pca_kmeans_{i}" for i in range(1, PCA_COMPONENTS + 1)]


def build_clustered_catalog() -> pd.DataFrame:
    """PCA(10) + K-means(5) sobre el dataset limpio, conservando pca_x/pca_y del notebook 02."""
    clean = pd.read_csv(c.CATALOG_PATH)
    clustered, _ = add_clusters(prepare_frame(clean))
    reference = pd.read_csv(c.CLUSTERED_CATALOG_PATH) if c.CLUSTERED_CATALOG_PATH.exists() else None
    clustered = name_clusters(clustered, reference)
    if reference is not None and {"pca_x", "pca_y"} <= set(reference.columns):
        clustered = clustered.merge(reference[["repo_id", "pca_x", "pca_y"]], on="repo_id", how="left")
    clustered["topics_list"] = clustered["topics_list"].astype(str)
    return clustered


def same_as_published(new: pd.DataFrame, pca_tol: float = 1e-6) -> bool:
    """True si el clustering reconstruido es el publicado (misma partición y mismas componentes salvo signo)."""
    if not c.CLUSTERED_CATALOG_PATH.exists():
        return False
    old = pd.read_csv(c.CLUSTERED_CATALOG_PATH)
    if len(old) != len(new) or not (old["nombre"].values == new["nombre"].values).all():
        return False
    same_partition = adjusted_rand_score(old["cluster_kmeans"], new["cluster_kmeans"]) >= 0.999
    same_pca = np.abs(np.abs(old[PCA_COLUMNS].to_numpy()) - np.abs(new[PCA_COLUMNS].to_numpy())).max() <= pca_tol
    return bool(same_partition and same_pca)


def build_graph_artifacts(catalog: pd.DataFrame) -> None:
    """PageRank y betweenness sobre los 1861 nodos; posiciones por estrellas contra los 3034 repositorios."""
    edges = pd.read_csv(c.EDGE_LIST_PATH)
    table = compute_graph_metrics(edges, catalog)
    table.to_csv(c.CENTRALITY_PATH, index=False)
    table.head(15).to_csv(c.PAGERANK_STARS_PATH, index=False)
    graph_summary(edges, table, len(catalog)).to_csv(c.GRAPH_SUMMARY_PATH, index=False)
    _plot_graph_vs_popularity(table, c.REPORTS_DIR / "graph" / "ranking_comparison.png")
    _plot_graph_visualization(edges, table, c.REPORTS_DIR / "graph" / "graph_visualization.png")
    print(f"[OK] grafo: {len(table)} nodos con PageRank y betweenness del grafo completo")


def _plot_graph_vs_popularity(table: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table.head(10)[::-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.8))
    left.scatter(table["stars"], table["pagerank"], s=10, alpha=0.4, color="#0072B2")
    for row in table.head(5).itertuples():
        left.annotate(row.nombre.split("/")[-1], (row.stars, row.pagerank), fontsize=8, xytext=(4, 4), textcoords="offset points")
    left.set_xscale("log")
    left.set_xlabel("Estrellas (escala log)")
    left.set_ylabel("PageRank")
    left.set_title("Popularidad vs importancia estructural (1861 repositorios)")
    left.grid(alpha=0.3)
    labels = [f"{n.split('/')[-1]}  (#{r:,} en estrellas)" for n, r in zip(top["nombre"], top["rank_stars"])]
    right.barh(labels, top["pagerank"], color="#D55E00")
    right.set_xlabel("PageRank")
    right.set_title("Top 10 por PageRank y su posición por estrellas\n(entre los 3034 repositorios)")
    right.grid(axis="x", alpha=0.3)
    fig.suptitle("AI Repository Discovery: grafo frente a popularidad")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_graph_visualization(edges: pd.DataFrame, table: pd.DataFrame, path, top_n: int = 45) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    from src.graph_metrics import build_graph

    top = table.head(top_n)
    sub = build_graph(edges).subgraph(top["nombre"]).copy()
    pos = nx.spring_layout(sub, seed=42, weight="weight", k=0.9)
    pagerank = top.set_index("nombre")["pagerank"]
    degree = top.set_index("nombre")["weighted_degree"]
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(sub, pos, alpha=0.15, width=0.6, ax=ax)
    nodes = nx.draw_networkx_nodes(
        sub, pos, node_size=[degree[n] / 3 for n in sub], node_color=[pagerank[n] for n in sub], cmap="viridis", ax=ax
    )
    nx.draw_networkx_labels(sub, pos, labels={n: n.split("/")[-1] for n in list(sub)[:0] or top["nombre"].head(12)}, font_size=7, ax=ax)
    fig.colorbar(nodes, ax=ax, shrink=0.7, label="PageRank")
    ax.set_title(f"AI Repository Discovery: los {top_n} repositorios con mayor PageRank\n"
                 "arista = al menos 4 topics compartidos · tamaño = weighted degree")
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_system_comparison() -> None:
    """Figuras de comparación de sistemas a partir de offline_evaluation_summary.csv (protocolo único)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(c.EVAL_SUMMARY_PATH).set_index("model")
    order = ["Popularity baseline", "Content-based (description only)", "Content-based baseline",
             "Hybrid PCA + K-means ranking", "SVD (repo-topic matrix)"]
    summary = summary.loc[order]
    ax = summary[["Precision@10", "MAP@10", "NDCG@10"]].plot.bar(figsize=(10, 4.8), color=["#0072B2", "#E69F00", "#009E73"], rot=20)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Evaluación offline: 500 consultas, K=10 (relevancia: 100 topics más frecuentes)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for name in ("recommendation/metrics_comparison.png", "system_comparison/systems_comparison.png"):
        target = c.REPORTS_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target, dpi=150)
    plt.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="no escribe; falla si el artefacto publicado difiere")
    args = parser.parse_args(argv)

    clustered = build_clustered_catalog()
    unchanged = same_as_published(clustered)
    print(f"[{'OK' if unchanged else 'CAMBIO'}] repos_clustered.csv "
          f"{'coincide con el artefacto publicado' if unchanged else 'difiere del artefacto publicado'}")
    if args.check:
        return 0 if unchanged else 1

    c.RECOMMENDATION_DIR.mkdir(parents=True, exist_ok=True)
    if not unchanged:  # evita ruido en Git: solo se reescribe si realmente cambió
        clustered.to_csv(c.CLUSTERED_CATALOG_PATH, index=False)
        print("      repos_clustered.csv reescrito")

    state = prepare_recommender(clustered)
    catalog = state["catalog"]
    build_cluster_profile(catalog).to_csv(c.CLUSTER_PROFILE_PATH, index=False)

    main_eval = run_evaluation(state, "top100")
    main_eval["summary"].to_csv(c.EVAL_SUMMARY_PATH, index=False)
    main_eval["detail"].to_csv(c.EVAL_DETAIL_PATH, index=False)
    main_eval["strong"].to_csv(c.STRONG_CASES_PATH, index=False)
    main_eval["failure"].to_csv(c.FAILURE_CASES_PATH, index=False)
    print(f"[OK] evaluación: {len(main_eval['queries'])} consultas de {main_eval['eligible']} evaluables")

    robust = run_evaluation(state, "tail")
    robust["summary"].assign(relevancia="topics ranking 31-100").to_csv(c.ROBUSTNESS_PATH, index=False)
    run_signal_ablation(state, main_eval["relevance"], main_eval["queries"]).to_csv(c.ABLATION_PATH, index=False)
    run_data_quality_experiment().to_csv(c.DATA_QUALITY_PATH, index=False)
    print("[OK] robustez, ablación de señales y calidad de datos")

    _plot_system_comparison()
    build_graph_artifacts(catalog)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Valida de punta a punta los artefactos y servicios que consume el dashboard.

    python src/run_pipeline.py

No descarga datos de GitHub (eso es `src/ingest.py`) ni modifica artefactos: comprueba que
todo lo que necesita el dashboard existe y funciona, y escribe artifacts/pipeline/pipeline_report.json
con rutas relativas al proyecto.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import common as c  # noqa: E402

REQUIRED_FILES = [
    c.CATALOG_PATH, c.CLUSTERED_CATALOG_PATH, c.EVAL_SUMMARY_PATH, c.EVAL_DETAIL_PATH,
    c.STRONG_CASES_PATH, c.FAILURE_CASES_PATH, c.ABLATION_PATH, c.DATA_QUALITY_PATH,
    c.ROBUSTNESS_PATH, c.CLUSTER_PROFILE_PATH, c.EDGE_LIST_PATH, c.CENTRALITY_PATH, c.PAGERANK_STARS_PATH,
]
REQUIRED_CATALOG_COLUMNS = [
    "nombre", "descripcion", "lenguaje", "stars", "forks", "topics", "dias_sin_commit",
    "antiguedad_dias", "num_topics", "ratio_forks_stars", "stars_por_dia", "cluster_kmeans", "cluster_nombre",
]
MIN_CATALOG_ROWS = 3000
MIN_GRAPH_EDGES = 20000
QUERY_REPOSITORY = "huggingface/transformers"
GRAPH_REPOSITORY = "tensorflow/tensorflow"


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _validate_files() -> None:
    missing = [_relative(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos (genera los de recomendación con `python -m src.build_artifacts` "
            "y los del grafo con el notebook 05):\n- " + "\n- ".join(missing)
        )


def _validate_recommenders() -> dict:
    from src.recommender import METHODS, prepare_recommender, recommend, recommend_by_preferences

    state = prepare_recommender()
    catalog = state["catalog"]
    if len(catalog) < MIN_CATALOG_ROWS:
        raise ValueError(f"El catálogo tiene {len(catalog)} filas y se esperaban al menos {MIN_CATALOG_ROWS}.")
    missing = [col for col in REQUIRED_CATALOG_COLUMNS if col not in catalog.columns]
    if missing:
        raise ValueError("Faltan columnas en el catálogo: " + ", ".join(missing))
    if QUERY_REPOSITORY not in set(catalog["nombre"]):
        raise KeyError(f"No se encontró el repositorio de prueba: {QUERY_REPOSITORY}")

    report = {"catalog_rows": len(catalog), "methods": {}}
    for method in METHODS:
        result = recommend(QUERY_REPOSITORY, method, top_n=5, recommender=state)
        if len(result) != 5:
            raise ValueError(f"{method}: se esperaban 5 resultados y se obtuvieron {len(result)}.")
        if QUERY_REPOSITORY in set(result["Repositorio recomendado"]):
            raise ValueError(f"{method}: incluyó el repositorio de consulta.")
        report["methods"][method] = result["Repositorio recomendado"].tolist()

    prefs = recommend_by_preferences(
        language="Python", topics=["computer-vision"], min_stars=1000,
        max_days_without_commit=180, top_n=5, recommender=state,
    )
    report["preferences_rows"] = int(len(prefs))
    return report


def _validate_evaluation() -> dict:
    import pandas as pd

    summary = pd.read_csv(c.EVAL_SUMMARY_PATH)
    if summary.empty or "Precision@10" not in summary.columns:
        raise ValueError("offline_evaluation_summary.csv está vacío o sin Precision@10.")
    return {row["model"]: round(float(row["Precision@10"]), 4) for _, row in summary.iterrows()}


def _validate_graph() -> dict:
    from src.graph_service import (
        build_interactive_subgraph, load_centrality_ranking, load_graph_edges, load_pagerank_vs_stars,
    )

    edges = load_graph_edges()
    if len(edges) <= MIN_GRAPH_EDGES:
        raise ValueError(f"El grafo tiene {len(edges)} aristas y se esperaban más de {MIN_GRAPH_EDGES}.")
    if load_centrality_ranking().empty or load_pagerank_vs_stars().empty:
        raise ValueError("Las tablas de centralidad o PageRank vs estrellas están vacías.")
    payload = build_interactive_subgraph(GRAPH_REPOSITORY, max_neighbors=8, min_weight=4)
    if not payload.get("html", "").strip():
        raise ValueError("El HTML del grafo interactivo está vacío.")
    stats = payload.get("graph_stats", {})
    return {"edges": len(edges), "subgraph_nodes": stats.get("subgrafo_nodos", 0), "subgraph_edges": stats.get("subgrafo_aristas", 0)}


def main() -> int:
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "files_validated": [], "errors": []}
    try:
        _validate_files()
        report["files_validated"] = [_relative(p) for p in REQUIRED_FILES]
        _ok("Archivos validados")

        report["recommenders"] = _validate_recommenders()
        _ok(f"Recomendadores validados ({', '.join(report['recommenders']['methods'])})")

        report["evaluation_precision_at_10"] = _validate_evaluation()
        _ok("Evaluación offline disponible")

        report["graph"] = _validate_graph()
        _ok("Grafo validado")
        status = 0
    except Exception as error:  # el reporte registra el motivo y el proceso termina con código 1
        report["errors"].append(str(error))
        print(f"[ERROR] {error}")
        status = 1

    c.PIPELINE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    c.PIPELINE_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if status == 0:
        _ok("Pipeline completado")
    return status


if __name__ == "__main__":
    sys.exit(main())

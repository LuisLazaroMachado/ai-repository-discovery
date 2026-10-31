"""Rutas del proyecto y utilidades compartidas por src/ y dashboard/."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- datos
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "repos_clean.csv"
RAW_SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "repos_raw.csv"

# --- artefactos generados (los consume el dashboard)
RECOMMENDATION_DIR = PROJECT_ROOT / "artifacts" / "recommendation"
CLUSTERED_CATALOG_PATH = RECOMMENDATION_DIR / "repos_clustered.csv"
EVAL_SUMMARY_PATH = RECOMMENDATION_DIR / "offline_evaluation_summary.csv"
EVAL_DETAIL_PATH = RECOMMENDATION_DIR / "offline_evaluation_detail.csv"
STRONG_CASES_PATH = RECOMMENDATION_DIR / "strong_cases.csv"
FAILURE_CASES_PATH = RECOMMENDATION_DIR / "failure_cases.csv"
SYSTEMS_COMPARISON_PATH = PROJECT_ROOT / "artifacts" / "system_comparison" / "systems_comparison.csv"
ABLATION_PATH = RECOMMENDATION_DIR / "signal_ablation.csv"
DATA_QUALITY_PATH = RECOMMENDATION_DIR / "data_quality_impact.csv"
CLUSTER_PROFILE_PATH = RECOMMENDATION_DIR / "cluster_profile.csv"

GRAPH_DIR = PROJECT_ROOT / "artifacts" / "graph"
EDGE_LIST_PATH = GRAPH_DIR / "edge_list.csv"
CENTRALITY_PATH = GRAPH_DIR / "centrality_ranking.csv"
PAGERANK_STARS_PATH = GRAPH_DIR / "pagerank_vs_stars.csv"
GRAPH_SUMMARY_PATH = GRAPH_DIR / "graph_summary.csv"

PIPELINE_REPORT_PATH = PROJECT_ROOT / "artifacts" / "pipeline" / "pipeline_report.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42


def parse_topics(value) -> List[str]:
    """Convierte la columna `topics` (lista o texto "['a', 'b']") en una lista de str."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(t).strip().lower() for t in value if str(t).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if text in ("", "[]", "nan", "None"):
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(t).strip().lower() for t in parsed if str(t).strip()]
    return [t.strip().lower() for t in text.split(",") if t.strip()]


def numeric_column(dataframe: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Columna numérica sin inf/NaN (se rellenan con `default`)."""
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype=float)
    return (
        pd.to_numeric(dataframe[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
    )

ROBUSTNESS_PATH = RECOMMENDATION_DIR / "evaluation_robustness.csv"

# Fotografía oficial de datos (ver data/README.md)
SNAPSHOT_DATE = "30 de abril de 2026"
SNAPSHOT_SOURCE = "API REST de GitHub (search/repositories) con PyGithub"

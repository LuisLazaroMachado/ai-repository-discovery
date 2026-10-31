"""Evaluación offline de los recomendadores y experimentos de calidad de datos.

Protocolo (el mismo del notebook 04, ahora ejecutable desde `src/`):

- No hay usuarios, clics ni ratings, así que la relevancia es un *proxy*: un candidato es
  relevante si comparte topics "significativos" con la consulta (los 100 topics más
  frecuentes; se exige compartir 2 si la consulta tiene >= 3 topics significativos y 1 si
  tiene menos).
- Una consulta es evaluable si tiene al menos 3 candidatos relevantes. Se evalúa una muestra
  reproducible de hasta 500 consultas (semilla 42).
- Métricas @10: Precision, Recall, MAP, NDCG y Coverage.

Limitación importante: los métodos que usan topics (contenido, SVD, PCA/K-means) están
favorecidos porque la relevancia también se define con topics. Por eso se agrega una variante
de robustez (`scope="tail"`) que mide relevancia solo con los topics ranking 31-100, que ni el
SVD ni el PCA/K-means llegan a ver.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from src.common import CATALOG_PATH, RANDOM_SEED
from src.features import add_clusters, cluster_profile, name_clusters, prepare_frame
from src.recommender import HYBRID_WEIGHTS, prepare_recommender, top_indices

K = 10
TOP_EVAL_TOPICS = 100
MAX_EVAL_QUERIES = 500
MIN_RELEVANT = 3

HYBRID_NAME = "Hybrid PCA + K-means ranking"
CONTENT_NAME = "Content-based baseline"
POPULARITY_NAME = "Popularity baseline"
SVD_NAME = "SVD (repo-topic matrix)"
DESCRIPTION_NAME = "Content-based (description only)"
METRICS = [f"Precision@{K}", f"Recall@{K}", f"MAP@{K}", f"NDCG@{K}", f"Coverage@{K}"]


# --------------------------------------------------------------------------- #
# Relevancia proxy
# --------------------------------------------------------------------------- #
@dataclass
class Relevance:
    """Relevancia por consulta: matriz de solapamiento de topics y umbral por consulta."""

    overlaps: np.ndarray          # (n, n) topics significativos compartidos
    thresholds: np.ndarray        # (n,) 0 = consulta sin topics significativos
    eligible: List[int]           # consultas con >= MIN_RELEVANT candidatos relevantes
    scope: str

    def mask(self, query_idx: int) -> np.ndarray:
        mask = self.overlaps[query_idx] >= self.thresholds[query_idx]
        mask[query_idx] = False
        return mask


def build_relevance(catalog: pd.DataFrame, scope: str = "top100") -> Relevance:
    """`scope="top100"`: protocolo publicado. `scope="tail"`: solo topics ranking 31-100."""
    all_topics = [t for topics in catalog["topics_list"] for t in topics]
    ranked = [t for t, _ in Counter(all_topics).most_common(TOP_EVAL_TOPICS)]
    if scope == "top100":
        vocabulary = ranked
    elif scope == "tail":
        vocabulary = ranked[30:]
    else:
        raise ValueError("scope debe ser 'top100' o 'tail'")

    vocab = set(vocabulary)
    eval_topics = catalog["topics_list"].apply(lambda ts: sorted(set(ts) & vocab))
    binary = csr_matrix(MultiLabelBinarizer(classes=vocabulary).fit_transform(eval_topics))
    overlaps = binary.dot(binary.T).toarray().astype(np.int16)

    counts = eval_topics.apply(len).to_numpy()
    thresholds = np.where(counts == 0, 10**6, np.where(counts >= 3, 2, 1)).astype(np.int32)

    relevance = Relevance(overlaps, thresholds, [], scope)
    relevance.eligible = [i for i in range(len(catalog)) if relevance.mask(i).sum() >= MIN_RELEVANT]
    return relevance


def select_queries(relevance: Relevance, max_queries: int = MAX_EVAL_QUERIES, seed: int = RANDOM_SEED) -> List[int]:
    eligible = relevance.eligible
    if len(eligible) <= max_queries:
        return list(eligible)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=max_queries, replace=False).tolist())


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _hits(recommended: List[int], mask: np.ndarray, k: int) -> np.ndarray:
    return mask[np.asarray(recommended[:k], dtype=int)]


def precision_at_k(recommended, mask, k=K) -> float:
    hits = _hits(recommended, mask, k)
    return float(hits.sum() / len(hits)) if len(hits) else 0.0


def recall_at_k(recommended, mask, k=K) -> float:
    relevant = int(mask.sum())
    return float(_hits(recommended, mask, k).sum() / relevant) if relevant else 0.0


def average_precision_at_k(recommended, mask, k=K) -> float:
    relevant = int(mask.sum())
    if relevant == 0:
        return 0.0
    hits = _hits(recommended, mask, k)
    score, running = 0.0, 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            running += 1
            score += running / rank
    return score / min(relevant, k)


def ndcg_at_k(recommended, mask, k=K) -> float:
    relevant = int(mask.sum())
    ideal = min(relevant, k)
    if ideal == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(_hits(recommended, mask, k), start=1) if hit)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg


def evaluate_ranker(
    name: str,
    ranker: Callable[[int], List[int]],
    queries: List[int],
    relevance: Relevance,
    catalog: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (resumen de una fila, detalle por consulta) con el esquema del notebook 04."""
    rows, recommended_all = [], []
    for q in queries:
        recommended = ranker(q)[:K]
        mask = relevance.mask(q)
        recommended_all.extend(recommended)
        rows.append(
            {
                "model": name,
                "query_idx": q,
                "query_repo": catalog.at[q, "nombre"],
                "relevant_candidates": int(mask.sum()),
                f"Precision@{K}": precision_at_k(recommended, mask),
                f"Recall@{K}": recall_at_k(recommended, mask),
                f"MAP@{K}": average_precision_at_k(recommended, mask),
                f"NDCG@{K}": ndcg_at_k(recommended, mask),
            }
        )
    detail = pd.DataFrame(rows)
    summary = detail.drop(columns=["query_idx", "query_repo"]).groupby("model").mean(numeric_only=True).reset_index()
    summary[f"Coverage@{K}"] = len(set(recommended_all)) / (len(catalog) - 1)
    return summary, detail


# --------------------------------------------------------------------------- #
# Sistemas evaluados
# --------------------------------------------------------------------------- #
def description_only_ranker(catalog: pd.DataFrame) -> Callable[[int], List[int]]:
    """TF-IDF solo con la descripción (la línea base de contenido de la comparación del notebook 04, parte 11)."""
    matrix = TfidfVectorizer(max_features=5000, stop_words="english").fit_transform(catalog["descripcion"].fillna(""))

    def ranker(q: int) -> List[int]:
        scores = matrix[q].dot(matrix.T).toarray().ravel()
        scores[q] = -np.inf
        return np.argsort(scores)[::-1][:K].tolist()

    return ranker


def official_systems(state: Dict[str, object]) -> Dict[str, Callable[[int], List[int]]]:
    return {
        POPULARITY_NAME: lambda q: top_indices(state, "popularity", q, K),
        CONTENT_NAME: lambda q: top_indices(state, "content", q, K),
        DESCRIPTION_NAME: description_only_ranker(state["catalog"]),
        SVD_NAME: lambda q: top_indices(state, "svd", q, K),
        HYBRID_NAME: lambda q: top_indices(state, "hybrid", q, K),
    }


def run_evaluation(state: Dict[str, object], scope: str = "top100") -> Dict[str, object]:
    catalog = state["catalog"]
    relevance = build_relevance(catalog, scope)
    queries = select_queries(relevance)

    summaries, details = [], []
    for name, ranker in official_systems(state).items():
        summary, detail = evaluate_ranker(name, ranker, queries, relevance, catalog)
        summaries.append(summary)
        details.append(detail)

    summary = pd.concat(summaries, ignore_index=True)
    detail = pd.concat(details, ignore_index=True)
    summary = summary[["model", "relevant_candidates"] + METRICS]

    hybrid = detail[detail["model"] == HYBRID_NAME].sort_values([f"Precision@{K}", f"NDCG@{K}"], ascending=False)
    strong = hybrid.head(5).copy()
    failure = hybrid.tail(5).sort_values([f"Precision@{K}", f"NDCG@{K}"], ascending=True).copy()
    for table in (strong, failure):
        table["diagnosis"] = [diagnose_case(state, int(q), relevance) for q in table["query_idx"]]

    return {
        "summary": summary,
        "detail": detail,
        "strong": strong,
        "failure": failure,
        "queries": queries,
        "eligible": len(relevance.eligible),
        "relevance": relevance,
    }


def diagnose_case(state: Dict[str, object], query_idx: int, relevance: Relevance) -> str:
    """Causas probables de un caso fuerte o fallido (misma lógica del notebook 04)."""
    catalog = state["catalog"]
    topics = set(catalog.at[query_idx, "topics_list"])
    all_topics = [t for ts in catalog["topics_list"] for t in ts]
    generic = {t for t, _ in Counter(all_topics).most_common(10)}
    cluster_size = int((state["clusters"] == state["clusters"][query_idx]).sum())
    recommended = top_indices(state, "hybrid", query_idx, 5)
    avg_shared = np.mean([len(topics & set(catalog.at[i, "topics_list"])) for i in recommended])
    has_eval_topics = relevance.thresholds[query_idx] < 10**6

    popularity = np.asarray(state["popularity_score"])
    popularity_gap = float(np.mean(popularity[recommended]) - popularity[query_idx])
    top10_hits = int(relevance.mask(query_idx)[top_indices(state, "hybrid", query_idx, K)].sum())
    sharing_any = sum(1 for i in recommended if topics & set(catalog.at[i, "topics_list"]))

    causes = []
    if len(topics) <= 1:
        causes.append("metadata limitada: el repositorio tiene pocos topics")
    if not has_eval_topics:
        causes.append("no tiene topics significativos dentro del vocabulario de evaluación")
    if len(topics & generic) / max(len(topics), 1) >= 0.5:
        causes.append("topics demasiado genéricos")
    if cluster_size <= 10:
        causes.append("clúster pequeño, con pocos candidatos cercanos")
    if avg_shared < 1:
        causes.append("bajo solapamiento promedio de topics con los recomendados")
    if popularity_gap >= 0.3:
        causes.append("sesgo hacia popularidad: los recomendados son mucho más populares que la consulta")
    if top10_hits == 0 and sharing_any >= 3:
        causes.append("posible límite del proxy: los recomendados comparten topics pero fuera del vocabulario de evaluación")
    return "; ".join(causes) if causes else "caso estable: suficiente metadata y candidatos similares"


# --------------------------------------------------------------------------- #
# Experimentos
# --------------------------------------------------------------------------- #
def run_signal_ablation(state: Dict[str, object], relevance: Relevance, queries: List[int]) -> pd.DataFrame:
    """Qué aporta cada señal del híbrido: se anula una a la vez y se mide el cambio."""
    catalog = state["catalog"]
    variants = {"Híbrido completo": dict(HYBRID_WEIGHTS)}
    labels = {
        "content": "contenido (TF-IDF)", "pca_similarity": "similitud PCA", "same_cluster": "mismo clúster",
        "popularity": "popularidad", "activity": "actividad", "growth": "crecimiento",
    }
    for key, label in labels.items():
        variants[f"Sin {label}"] = {**HYBRID_WEIGHTS, key: 0.0}
    variants["Solo contenido"] = {**{k: 0.0 for k in HYBRID_WEIGHTS}, "content": 1.0}

    rows = []
    for name, weights in variants.items():
        ranker = lambda q, w=weights: top_indices(state, "hybrid", q, K, weights=w)
        summary, _ = evaluate_ranker(name, ranker, queries, relevance, catalog)
        rows.append(summary.iloc[0].to_dict())
    table = pd.DataFrame(rows).rename(columns={"model": "variante"}).drop(columns=["relevant_candidates"])
    full = table.loc[table["variante"] == "Híbrido completo"].iloc[0]
    for metric in (f"Precision@{K}", f"NDCG@{K}"):
        table[f"Δ {metric}"] = table[metric] - full[metric]
    return table


def _evaluate_catalog_variant(label: str, clean: pd.DataFrame, log_counts: bool, note: str) -> Dict[str, object]:
    frame = prepare_frame(clean)
    clustered, stats = add_clusters(frame, log_counts=log_counts)
    state = prepare_recommender(name_clusters(clustered, reference=None))
    catalog = state["catalog"]
    relevance = build_relevance(catalog)
    queries = select_queries(relevance)
    results = {}
    for method in ("hybrid", "content"):
        summary, _ = evaluate_ranker(method, lambda q, m=method: top_indices(state, m, q, K), queries, relevance, catalog)
        results[method] = summary.iloc[0]
    return {
        "variante": label,
        "repositorios": len(catalog),
        "clúster más pequeño": stats["smallest_cluster"],
        "silhouette": round(stats["silhouette"], 3),
        f"Híbrido Precision@{K}": round(float(results["hybrid"][f"Precision@{K}"]), 4),
        f"Híbrido NDCG@{K}": round(float(results["hybrid"][f"NDCG@{K}"]), 4),
        f"Contenido Precision@{K}": round(float(results["content"][f"Precision@{K}"]), 4),
        "nota": note,
    }


def run_data_quality_experiment(clean: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Mide el efecto de las decisiones de preprocesamiento sobre cobertura, clústeres y precisión."""
    clean = pd.read_csv(CATALOG_PATH) if clean is None else clean
    incomplete = (clean["lenguaje"].fillna("Unknown") == "Unknown") | (clean["descripcion"].fillna("No description") == "No description")
    rows = [
        _evaluate_catalog_variant(
            "Dataset limpio (publicado)", clean, False,
            "Valores vacíos imputados (lenguaje='Unknown', descripción='No description'); estrellas/forks sin log en PCA.",
        ),
        _evaluate_catalog_variant(
            "Sin imputar vacíos (se descartan los repos incompletos)", clean.loc[~incomplete], False,
            f"Se pierden {int(incomplete.sum())} repositorios ({incomplete.mean():.1%}). Otro catálogo, otras consultas: comparación aproximada.",
        ),
        _evaluate_catalog_variant(
            "Estrellas, forks y estrellas/día en escala log (PCA y K-means)", clean, True,
            "Mismo catálogo y mismas consultas; solo cambia la escala de las variables con cola pesada.",
        ),
    ]
    return pd.DataFrame(rows)


def build_cluster_profile(catalog: pd.DataFrame) -> pd.DataFrame:
    return cluster_profile(catalog)

"""Etapa de representación: features -> PCA(10) -> K-means(K=5).

Reproduce exactamente la construcción del notebook 04 (que a su vez reconstruye el
clustering del notebook 03), para que el artefacto `repos_clustered.csv` se pueda
regenerar desde `src/` sin depender de ningún notebook.
"""
from __future__ import annotations

import warnings
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from src.common import RANDOM_SEED, parse_topics

NUMERIC_VARS = [
    "stars", "forks", "dias_sin_commit", "antiguedad_dias",
    "num_topics", "ratio_forks_stars", "stars_por_dia",
]
HEAVY_TAILED = ["stars", "forks", "stars_por_dia"]
N_TOP_TOPICS = 30
PCA_COMPONENTS = 10
N_CLUSTERS = 5

# Nombres asignados a los clusters publicados (ver reports/clustering/report.md).
# Solo son válidos para el clustering original: `name_clusters` comprueba que
# el reclustering reproduzca esa partición antes de aplicarlos.
PUBLISHED_CLUSTER_NAMES: Dict[int, str] = {
    0: "Herramientas de IA modernas",
    1: "Proyectos establecidos y educativos",
    2: "Ecosistema LLM emergente",
    3: "Proyectos masivos e influyentes",
    4: "Repos virales recientes",
}


def prepare_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Tipos mínimos: topics como lista, lenguaje sin vacíos, numéricas sin NaN."""
    df = dataframe.copy().reset_index(drop=True)
    df["topics_list"] = df["topics"].apply(parse_topics)
    df["lenguaje"] = df["lenguaje"].fillna("Unknown").replace("", "Unknown")
    for col in NUMERIC_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    return df


def build_cluster_features(df: pd.DataFrame, log_counts: bool = False) -> np.ndarray:
    """Matriz combinada: numéricas estandarizadas + lenguaje one-hot + 30 topics.

    `log_counts=True` aplica log1p a estrellas, forks y estrellas/día antes de
    estandarizar (variante usada solo en el experimento de calidad de datos).
    """
    numeric = df[NUMERIC_VARS].copy()
    if log_counts:
        for col in HEAVY_TAILED:
            numeric[col] = np.log1p(numeric[col].clip(lower=0))
    x_num = StandardScaler().fit_transform(numeric)
    x_lang = pd.get_dummies(df["lenguaje"], prefix="lang").astype(int).values

    all_topics = [t for topics in df["topics_list"] for t in topics]
    top_topics = [t for t, _ in Counter(all_topics).most_common(N_TOP_TOPICS)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_topics = MultiLabelBinarizer(classes=top_topics).fit_transform(df["topics_list"])
    return np.hstack([x_num, x_lang, x_topics])


def fit_pca_kmeans(
    features: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    n_clusters: int = N_CLUSTERS,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, PCA, KMeans]:
    pca = PCA(n_components=n_components, random_state=seed)
    embedding = pca.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=100)
    labels = kmeans.fit_predict(embedding)
    return embedding, labels, pca, kmeans


def add_clusters(df: pd.DataFrame, log_counts: bool = False) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Devuelve el catálogo con columnas pca_kmeans_1..10, pca_x/pca_y y cluster_kmeans."""
    features = build_cluster_features(df, log_counts=log_counts)
    embedding, labels, pca, kmeans = fit_pca_kmeans(features)
    out = df.copy()
    for i in range(embedding.shape[1]):
        out[f"pca_kmeans_{i + 1}"] = embedding[:, i]
    out["cluster_kmeans"] = labels
    stats = {
        "silhouette": float(silhouette_score(embedding, labels)),
        "inertia": float(kmeans.inertia_),
        "explained_variance_10": float(pca.explained_variance_ratio_.sum()),
        "smallest_cluster": int(np.bincount(labels).min()),
    }
    return out, stats


def name_clusters(df: pd.DataFrame, reference: pd.DataFrame | None) -> pd.DataFrame:
    """Asigna `cluster_nombre`.

    Si hay un catálogo de referencia (el artefacto publicado) y la partición coincide
    en >= 99 % de los repositorios, se usan los nombres publicados. Si no, nombres
    genéricos ("Cluster 0"...), porque los nombres solo describen el clustering original.
    """
    out = df.copy()
    names = {c: f"Cluster {c}" for c in sorted(out["cluster_kmeans"].unique())}
    if reference is not None and "cluster_kmeans" in reference.columns:
        merged = out[["nombre", "cluster_kmeans"]].merge(
            reference[["nombre", "cluster_kmeans"]], on="nombre", suffixes=("", "_ref")
        )
        if len(merged) and (merged["cluster_kmeans"] == merged["cluster_kmeans_ref"]).mean() >= 0.99:
            names = dict(PUBLISHED_CLUSTER_NAMES)
    out["cluster_nombre"] = out["cluster_kmeans"].map(names)
    return out


def cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen por clúster (tamaño y medianas) para el dashboard."""
    grouped = df.groupby(["cluster_kmeans", "cluster_nombre"])
    profile = grouped.agg(
        repositorios=("nombre", "size"),
        estrellas_mediana=("stars", "median"),
        estrellas_max=("stars", "max"),
        dias_sin_commit_mediana=("dias_sin_commit", "median"),
        antiguedad_dias_mediana=("antiguedad_dias", "median"),
        stars_por_dia_mediana=("stars_por_dia", "median"),
    ).reset_index()
    profile["lenguaje_principal"] = [
        df.loc[df["cluster_kmeans"] == c, "lenguaje"].mode().iat[0] for c in profile["cluster_kmeans"]
    ]
    return profile.round(2)

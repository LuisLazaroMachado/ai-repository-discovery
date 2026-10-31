"""Recomendadores de repositorios.

Una sola implementación sirve al dashboard, a los tests y a la evaluación offline
(`src/evaluation.py`). Los cuatro métodos disponibles son:

- ``popularity``: baseline que ignora la consulta y ordena por popularidad.
- ``content``: TF-IDF sobre descripción + topics (x3) + lenguaje.
- ``svd``: factorización SVD de la matriz repo-topic (30 topics más frecuentes).
- ``hybrid``: contenido + similitud PCA + mismo clúster + popularidad + actividad + crecimiento.

La fórmula del híbrido es la misma que se evaluó en el notebook 04
(similitud PCA llevada a [0, 1] con (coseno + 1) / 2 y actividad = 1 - minmax(días sin commit)).
`tests/test_evaluation.py` comprueba que reproduce las métricas publicadas.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, normalize

from src.common import (
    CATALOG_PATH,
    CLUSTERED_CATALOG_PATH,
    RANDOM_SEED,
    numeric_column,
    parse_topics,
)
from src.features import PCA_COMPONENTS, add_clusters, name_clusters, prepare_frame

TOPIC_COUNT = 30
SVD_COMPONENTS = 10

TFIDF_MAX_FEATURES = 1200
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_TOKEN_PATTERN = r"(?u)\b[\w\-\.\+/#]+\b"

HYBRID_WEIGHTS = {
    "content": 0.35,
    "pca_similarity": 0.20,
    "same_cluster": 0.15,
    "popularity": 0.12,
    "activity": 0.10,
    "growth": 0.08,
}

METHODS = {
    "hybrid": "Híbrido (contenido + PCA + clúster + señales de vigencia)",
    "content": "Contenido (TF-IDF de descripción, topics y lenguaje)",
    "svd": "SVD (factorización de la matriz repo-topic)",
    "popularity": "Popularidad (baseline, ignora el repositorio elegido)",
}

PCA_COLUMNS = [f"pca_kmeans_{i}" for i in range(1, PCA_COMPONENTS + 1)]


# --------------------------------------------------------------------------- #
# Carga del catálogo
# --------------------------------------------------------------------------- #
def _prepare_base_catalog(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    required = ["nombre", "descripcion", "lenguaje", "topics", "stars", "forks"]
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing))

    dataframe["nombre"] = dataframe["nombre"].fillna("Repositorio sin nombre").astype(str).str.strip()
    dataframe["descripcion"] = dataframe["descripcion"].fillna("No description").astype(str).str.strip()
    dataframe["lenguaje"] = dataframe["lenguaje"].fillna("Unknown").astype(str).str.strip()
    dataframe["topics_list"] = dataframe["topics"].apply(parse_topics)
    dataframe["stars"] = numeric_column(dataframe, "stars")
    dataframe["forks"] = numeric_column(dataframe, "forks")
    dataframe["dias_sin_commit"] = numeric_column(dataframe, "dias_sin_commit")
    dataframe["antiguedad_dias"] = numeric_column(dataframe, "antiguedad_dias", default=1).replace(0, 1)

    if "num_topics" in dataframe.columns:
        dataframe["num_topics"] = numeric_column(dataframe, "num_topics")
    else:
        dataframe["num_topics"] = dataframe["topics_list"].apply(len).astype(float)

    if "ratio_forks_stars" in dataframe.columns:
        dataframe["ratio_forks_stars"] = numeric_column(dataframe, "ratio_forks_stars")
    else:
        dataframe["ratio_forks_stars"] = dataframe["forks"] / dataframe["stars"].replace(0, 1)

    if "stars_por_dia" in dataframe.columns:
        dataframe["stars_por_dia"] = numeric_column(dataframe, "stars_por_dia")
    else:
        dataframe["stars_por_dia"] = dataframe["stars"] / dataframe["antiguedad_dias"]

    return dataframe.drop_duplicates(subset=["nombre"], keep="first").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    """Catálogo con PCA y clústeres.

    Usa el artefacto `repos_clustered.csv` si existe. Si no, parte del dataset limpio y
    calcula PCA + K-means con `src/features.py` (mismo resultado que el artefacto publicado).
    """
    if CLUSTERED_CATALOG_PATH.exists():
        return _prepare_base_catalog(pd.read_csv(CLUSTERED_CATALOG_PATH))
    if CATALOG_PATH.exists():
        return _with_clusters(_prepare_base_catalog(pd.read_csv(CATALOG_PATH)))
    raise FileNotFoundError(
        "No se encontró artifacts/recommendation/repos_clustered.csv ni data/processed/repos_clean.csv"
    )


def _with_clusters(dataframe: pd.DataFrame) -> pd.DataFrame:
    clustered, _ = add_clusters(prepare_frame(dataframe))
    clustered = name_clusters(clustered, reference=None)
    return _prepare_base_catalog(clustered)


# --------------------------------------------------------------------------- #
# Preparación del recomendador
# --------------------------------------------------------------------------- #
def _build_content_text(dataframe: pd.DataFrame) -> pd.Series:
    topics_text = dataframe["topics_list"].apply(lambda t: " ".join(t))
    return (
        dataframe["descripcion"].fillna("").astype(str)
        + " "
        + ((topics_text + " ") * 3)
        + dataframe["lenguaje"].fillna("Unknown").astype(str)
    )


def _top_topics(dataframe: pd.DataFrame, limit: int = TOPIC_COUNT) -> List[str]:
    all_topics = [t for topics in dataframe["topics_list"] for t in topics]
    return pd.Series(all_topics).value_counts().head(limit).index.tolist()


def _topic_matrix(dataframe: pd.DataFrame, top_topics: List[str]) -> csr_matrix:
    encoder = MultiLabelBinarizer(classes=top_topics, sparse_output=True)
    filtered = dataframe["topics_list"].apply(lambda ts: [t for t in ts if t in top_topics])
    return csr_matrix(encoder.fit_transform(filtered), dtype=np.float32)


def prepare_recommender(catalog: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Calcula las matrices y señales que usan los cuatro métodos."""
    dataframe = _prepare_base_catalog(catalog if catalog is not None else load_catalog())

    if not all(c in dataframe.columns for c in PCA_COLUMNS + ["cluster_kmeans"]):
        dataframe = _with_clusters(dataframe)
    if "cluster_nombre" not in dataframe.columns:
        dataframe = name_clusters(dataframe, reference=None)

    # contenido
    dataframe["recommendation_text"] = _build_content_text(dataframe)
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        token_pattern=TFIDF_TOKEN_PATTERN,
        lowercase=True,
    )
    content_matrix = tfidf.fit_transform(dataframe["recommendation_text"])  # filas con norma L2 = 1

    # PCA (ya calculado en el artefacto) normalizado para el coseno
    latent_norm = normalize(dataframe[PCA_COLUMNS].astype(float).to_numpy(), norm="l2", axis=1)

    # SVD sobre la matriz repo-topic
    top_topics = _top_topics(dataframe)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)
    svd_norm = normalize(svd.fit_transform(_topic_matrix(dataframe, top_topics)), norm="l2", axis=1)

    # señales de relevancia y vigencia
    popularity_features = pd.DataFrame(
        {
            "log_stars": np.log1p(dataframe["stars"]),
            "log_forks": np.log1p(dataframe["forks"]),
            "log_stars_por_dia": np.log1p(dataframe["stars_por_dia"].clip(lower=0)),
        }
    )
    scaled = MinMaxScaler().fit_transform(popularity_features)
    popularity_score = 0.60 * scaled[:, 0] + 0.25 * scaled[:, 1] + 0.15 * scaled[:, 2]

    activity_score = 1 - MinMaxScaler().fit_transform(dataframe[["dias_sin_commit"]])[:, 0]
    growth_raw = np.log1p(dataframe[["stars_por_dia"]].clip(lower=0))
    growth_score = MinMaxScaler().fit_transform(growth_raw)[:, 0]

    cluster_names = (
        dataframe.drop_duplicates("cluster_kmeans")
        .set_index("cluster_kmeans")["cluster_nombre"]
        .astype(str)
        .to_dict()
    )

    return {
        "catalog": dataframe,
        "content_matrix": content_matrix,
        "latent_norm": latent_norm,
        "svd_norm": svd_norm,
        "clusters": dataframe["cluster_kmeans"].astype(int).to_numpy(),
        "cluster_names": {int(k): v for k, v in cluster_names.items()},
        "popularity_score": popularity_score,
        "activity_score": activity_score,
        "growth_score": growth_score,
        "top_topics": top_topics,
        "tfidf": tfidf,
    }


def _get_state(recommender: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    return prepare_recommender() if recommender is None else recommender


# --------------------------------------------------------------------------- #
# Puntuación por método (usada por el dashboard y por la evaluación)
# --------------------------------------------------------------------------- #
def content_scores(state: Dict[str, object], query_idx: int) -> np.ndarray:
    matrix = state["content_matrix"]
    return matrix[query_idx].dot(matrix.T).toarray().ravel()


def pca_scores(state: Dict[str, object], query_idx: int) -> np.ndarray:
    cosine = state["latent_norm"] @ state["latent_norm"][query_idx]
    return (cosine + 1) / 2


def svd_scores(state: Dict[str, object], query_idx: int) -> np.ndarray:
    return state["svd_norm"] @ state["svd_norm"][query_idx]


def hybrid_scores(
    state: Dict[str, object],
    query_idx: int,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    w = weights or HYBRID_WEIGHTS
    clusters = state["clusters"]
    same_cluster = (clusters == clusters[query_idx]).astype(float)
    return (
        w["content"] * content_scores(state, query_idx)
        + w["pca_similarity"] * pca_scores(state, query_idx)
        + w["same_cluster"] * same_cluster
        + w["popularity"] * state["popularity_score"]
        + w["activity"] * state["activity_score"]
        + w["growth"] * state["growth_score"]
    )


def score_vector(
    state: Dict[str, object],
    method: str,
    query_idx: int,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Puntaje de todos los candidatos para un método; la consulta queda en -inf."""
    if method == "hybrid":
        scores = hybrid_scores(state, query_idx, weights)
    elif method == "content":
        scores = content_scores(state, query_idx)
    elif method == "svd":
        scores = svd_scores(state, query_idx)
    elif method == "popularity":
        scores = np.array(state["popularity_score"], dtype=float)
    else:
        raise ValueError(f"Método desconocido: {method!r}. Opciones: {sorted(METHODS)}")
    scores = np.array(scores, dtype=float)
    scores[query_idx] = -np.inf
    return scores


def top_indices(
    state: Dict[str, object],
    method: str,
    query_idx: int,
    k: int = 10,
    weights: Optional[Dict[str, float]] = None,
) -> List[int]:
    return np.argsort(score_vector(state, method, query_idx, weights))[::-1][:k].tolist()


# --------------------------------------------------------------------------- #
# Recomendaciones item-a-item
# --------------------------------------------------------------------------- #
_FALLBACK_REASON = {
    "hybrid": "Alta similitud global en contenido, PCA y señales de vigencia",
    "content": "Texto, topics y lenguaje similares",
    "svd": "Perfil de topics similar en el espacio latente",
    "popularity": "Repositorio popular; este baseline no considera similitud",
}


def recommend(
    repository_name: str,
    method: str = "hybrid",
    top_n: int = 10,
    recommender: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Top-N repositorios para `repository_name` con el método elegido.

    Todos los métodos devuelven el mismo esquema de columnas, de modo que el
    dashboard los muestra igual; solo cambia cómo se calcula `Score`.
    """
    state = _get_state(recommender)
    catalog = state["catalog"]
    matches = catalog.index[catalog["nombre"] == repository_name].tolist()
    if not matches:
        return pd.DataFrame()

    query_idx = matches[0]
    scores = score_vector(state, method, query_idx)
    ranked = np.argsort(scores)[::-1][:top_n]

    content = content_scores(state, query_idx)
    pca = pca_scores(state, query_idx)
    clusters = state["clusters"]
    names = state["cluster_names"]
    query_topics = set(catalog.iloc[query_idx]["topics_list"])
    query_language = catalog.iloc[query_idx]["lenguaje"]

    rows = []
    for position, idx in enumerate(ranked, start=1):
        candidate = catalog.iloc[idx]
        shared = sorted(query_topics & set(candidate["topics_list"]))
        same_cluster = int(clusters[idx] == clusters[query_idx])

        reasons = []
        if shared:
            reasons.append("Topics compartidos: " + ", ".join(shared[:5]))
        if same_cluster and method != "popularity":
            reasons.append("Mismo clúster")
        if candidate["lenguaje"] == query_language and query_language != "Unknown":
            reasons.append("Mismo lenguaje principal")
        if not reasons:
            reasons.append(_FALLBACK_REASON[method])

        rows.append(
            {
                "Posición": position,
                "Repositorio recomendado": candidate["nombre"],
                "Método": method,
                "Score": round(float(scores[idx]), 4),
                "Lenguaje": candidate["lenguaje"],
                "Estrellas": int(candidate["stars"]),
                "Forks": int(candidate["forks"]),
                "Clúster": names.get(int(clusters[idx]), str(int(clusters[idx]))),
                "Mismo clúster": same_cluster,
                "Topics compartidos": ", ".join(shared),
                "Motivo de recomendación": ". ".join(reasons),
                "URL": candidate.get("url", ""),
                # señales individuales (el desglose se muestra para cualquier método)
                "Similitud de contenido": round(float(content[idx]), 4),
                "Similitud PCA": round(float(pca[idx]), 4),
                "Bonus clúster": float(HYBRID_WEIGHTS["same_cluster"] * same_cluster),
                "Popularidad": float(state["popularity_score"][idx]),
                "Actividad": float(state["activity_score"][idx]),
                "Crecimiento": float(state["growth_score"][idx]),
            }
        )

    result = pd.DataFrame(rows)
    if method == "hybrid":
        result["Score híbrido"] = result["Score"]
    return result


def recommend_hybrid(repository_name: str, top_n: int = 10, recommender=None) -> pd.DataFrame:
    return recommend(repository_name, "hybrid", top_n, recommender)


def recommend_content(repository_name: str, top_n: int = 10, recommender=None) -> pd.DataFrame:
    return recommend(repository_name, "content", top_n, recommender)


def recommend_svd(repository_name: str, top_n: int = 10, recommender=None) -> pd.DataFrame:
    return recommend(repository_name, "svd", top_n, recommender)


def recommend_popularity(repository_name: str, top_n: int = 10, recommender=None) -> pd.DataFrame:
    return recommend(repository_name, "popularity", top_n, recommender)


# --------------------------------------------------------------------------- #
# Búsqueda por preferencias (no parte de un repositorio de referencia)
# --------------------------------------------------------------------------- #
PREFERENCE_COLUMNS = [
    "Posición", "Repositorio", "Score de preferencias", "Coincidencia de topics",
    "Coincidencia de lenguaje", "Popularidad", "Actividad", "Crecimiento", "Lenguaje",
    "Estrellas", "Forks", "Días sin commit", "Clúster", "Topics coincidentes", "Motivo", "URL",
]


def recommend_by_preferences(
    language: Optional[str] = None,
    topics: Optional[List[str]] = None,
    min_stars: float = 0,
    max_days_without_commit: Optional[float] = None,
    cluster: Optional[int] = None,
    top_n: int = 10,
    recommender: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    state = _get_state(recommender)
    filtered = state["catalog"].copy()

    if language:
        filtered = filtered[filtered["lenguaje"] == language]
    filtered = filtered[filtered["stars"] >= float(min_stars)]
    if max_days_without_commit is not None:
        filtered = filtered[filtered["dias_sin_commit"] <= float(max_days_without_commit)]
    if cluster is not None:
        filtered = filtered[filtered["cluster_kmeans"].astype(int) == int(cluster)]

    if filtered.empty:
        return pd.DataFrame(columns=PREFERENCE_COLUMNS)

    selected_topics = [t for t in (topics or []) if t]
    if selected_topics:
        topic_match = np.array(
            [len(set(selected_topics) & set(ts)) / len(selected_topics) for ts in filtered["topics_list"]]
        )
    else:
        topic_match = np.zeros(len(filtered))
    language_match = np.full(len(filtered), 1.0 if language else 0.0)

    positions = filtered.index.to_numpy()
    popularity = np.asarray(state["popularity_score"])[positions]
    activity = np.asarray(state["activity_score"])[positions]
    growth = np.asarray(state["growth_score"])[positions]

    filtered["preferences_score"] = (
        0.40 * topic_match + 0.20 * language_match + 0.15 * popularity + 0.15 * activity + 0.10 * growth
    )
    filtered["topic_match_score"] = topic_match
    filtered["language_match_score"] = language_match
    filtered["popularidad"], filtered["actividad"], filtered["crecimiento"] = popularity, activity, growth
    filtered = filtered.sort_values(
        ["preferences_score", "stars", "forks"], ascending=[False, False, False]
    ).head(top_n)

    names = state["cluster_names"]
    rows = []
    for position, (_, row) in enumerate(filtered.iterrows(), start=1):
        matched = sorted(set(selected_topics) & set(row["topics_list"])) if selected_topics else []
        reasons = []
        if selected_topics:
            reasons.append(f"Coincide {len(matched)} de {len(selected_topics)} topics seleccionados")
        if language:
            reasons.append("Coincide con el lenguaje seleccionado")
        if min_stars:
            reasons.append(f"Cumple el mínimo de {int(min_stars)} estrellas")
        if max_days_without_commit is not None:
            reasons.append(f"Se mantiene dentro de {int(max_days_without_commit)} días sin commit")
        if cluster is not None:
            reasons.append(f"Pertenece al clúster {int(cluster)}")
        if not reasons:
            reasons.append("Cumple los criterios de preferencias seleccionados")

        cluster_id = int(row["cluster_kmeans"])
        rows.append(
            {
                "Posición": position,
                "Repositorio": row["nombre"],
                "Score de preferencias": round(float(row["preferences_score"]), 4),
                "Coincidencia de topics": round(float(row["topic_match_score"]), 4),
                "Coincidencia de lenguaje": round(float(row["language_match_score"]), 4),
                "Popularidad": round(float(row["popularidad"]), 4),
                "Actividad": round(float(row["actividad"]), 4),
                "Crecimiento": round(float(row["crecimiento"]), 4),
                "Lenguaje": row["lenguaje"],
                "Estrellas": int(row["stars"]),
                "Forks": int(row["forks"]),
                "Días sin commit": int(row["dias_sin_commit"]),
                "Clúster": names.get(cluster_id, str(cluster_id)),
                "Topics coincidentes": ", ".join(matched),
                "Motivo": ". ".join(reasons),
                "URL": row.get("url", ""),
            }
        )
    return pd.DataFrame(rows, columns=PREFERENCE_COLUMNS)


def get_repository_row(repository_name: str, recommender: Optional[Dict[str, object]] = None) -> pd.Series:
    catalog = _get_state(recommender)["catalog"]
    matches = catalog.index[catalog["nombre"] == repository_name].tolist()
    if not matches:
        raise KeyError(f"No se encontró el repositorio: {repository_name}")
    return catalog.iloc[matches[0]]


def list_repository_names(recommender: Optional[Dict[str, object]] = None) -> List[str]:
    return sorted(_get_state(recommender)["catalog"]["nombre"].dropna().astype(str).unique().tolist())

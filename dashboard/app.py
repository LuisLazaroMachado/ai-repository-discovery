"""AI Repository Discovery: dashboard de Streamlit.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.sections import clusters, evaluation_page, graph_page, overview, recommender_page  # noqa: E402
from src.graph_service import load_centrality_ranking, load_graph_edges, load_pagerank_vs_stars  # noqa: E402
from src.recommender import load_catalog, prepare_recommender  # noqa: E402

st.set_page_config(page_title="AI Repository Discovery", page_icon="🔎", layout="wide")

SECTIONS = {
    "Resumen": overview,
    "Recomendador": recommender_page,
    "Clústeres y PCA": clusters,
    "Evaluación": evaluation_page,
    "Grafo interactivo": graph_page,
}


@st.cache_data(show_spinner=False)
def read_csv(path: str):
    """CSV de artefactos; None si todavía no existe."""
    return pd.read_csv(path) if Path(path).exists() else None


@st.cache_resource(show_spinner="Preparando el recomendador...")
def get_recommender():
    return prepare_recommender(load_catalog())


@st.cache_data(show_spinner=False)
def get_centrality() -> pd.DataFrame:
    return load_centrality_ranking()


@st.cache_data(show_spinner=False)
def get_pagerank_vs_stars() -> pd.DataFrame:
    return load_pagerank_vs_stars()


@st.cache_data(show_spinner=False)
def get_graph_nodes() -> list:
    edges = load_graph_edges()
    return sorted(set(edges["src"].astype(str)) | set(edges["dst"].astype(str)))


recommender = get_recommender()
catalog = recommender["catalog"]
context = {
    "catalog": catalog,
    "recommender": recommender,
    "read_csv": read_csv,
    "centrality": get_centrality(),
    "pagerank_vs_stars": get_pagerank_vs_stars(),
    "graph_nodes": get_graph_nodes(),
}

st.title("AI Repository Discovery")
st.caption("Descubre repositorios de IA parecidos entre sí y explora la red de topics que los conecta.")

section = st.sidebar.radio("Navegación", list(SECTIONS))
st.sidebar.caption("Datos reales de la API de GitHub · fotografía del 30 de abril de 2026.")
SECTIONS[section].render(context)

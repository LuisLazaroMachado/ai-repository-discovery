"""Recomendador: cuatro métodos item-a-item y búsqueda por preferencias."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.ui import bordered, fmt_float, fmt_int, show_table
from src.recommender import (
    HYBRID_WEIGHTS, METHODS, get_repository_row, recommend, recommend_by_preferences,
)

METHOD_HELP = {
    "hybrid": "Combina contenido (35 %), similitud PCA (20 %), mismo clúster (15 %), popularidad (12 %), actividad (10 %) y crecimiento (8 %).",
    "content": "Compara el texto de la descripción, los topics y el lenguaje (TF-IDF + coseno).",
    "svd": "Compara el perfil de topics de cada repositorio en un espacio latente (SVD).",
    "popularity": "Baseline: devuelve siempre los más populares, sin importar el repositorio elegido.",
}
SIGNALS = [
    ("Similitud de contenido", "content", "Similitud de contenido"),
    ("Similitud PCA", "pca_similarity", "Similitud PCA"),
    ("Bonus de clúster", "same_cluster", "Bonus clúster"),
    ("Popularidad", "popularity", "Popularidad"),
    ("Actividad", "activity", "Actividad"),
    ("Crecimiento", "growth", "Crecimiento"),
]


def _repository_filters(df: pd.DataFrame) -> pd.DataFrame:
    languages = ["Todos"] + sorted(df["lenguaje"].dropna().astype(str).unique().tolist())
    language = st.selectbox("Filtro por lenguaje", languages, help="Reduce el catálogo por lenguaje principal.")
    topic_options = ["Todos"] + sorted({t for topics in df["topics_list"] for t in topics})
    topic = st.selectbox("Filtro por topic", topic_options, help="Opcional: solo repositorios con ese topic.")
    name_query = st.text_input("Buscar por nombre", placeholder="Parte del nombre del repositorio")

    filtered = df
    if language != "Todos":
        filtered = filtered[filtered["lenguaje"] == language]
    if topic != "Todos":
        filtered = filtered[filtered["topics_list"].apply(lambda topics: topic in topics)]
    if name_query.strip():
        filtered = filtered[filtered["nombre"].str.contains(name_query.strip(), case=False, na=False)]
    return filtered.reset_index(drop=True)


def _repository_card(row: pd.Series, cluster_name: str) -> None:
    with bordered():
        st.subheader(row["nombre"])
        st.write(row["descripcion"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Lenguaje", row["lenguaje"])
        c2.metric("Estrellas", fmt_int(row["stars"]))
        c3.metric("Forks", fmt_int(row["forks"]))
        c4, c5, c6 = st.columns(3)
        c4.metric("Días sin commit", fmt_int(row["dias_sin_commit"]))
        c5.metric("Estrellas por día", fmt_float(row["stars_por_dia"], 3))
        c6.metric("Topics", fmt_int(row["num_topics"]))
        st.markdown(f"**Clúster:** {cluster_name}")
        topics = ", ".join(row["topics_list"]) if row["topics_list"] else "Sin topics disponibles"
        st.markdown(f"**Topics:** {topics}")
        if isinstance(row.get("url", ""), str) and row["url"].strip():
            st.markdown(f"[Abrir en GitHub]({row['url']})")


def _recommendation_card(row: pd.Series) -> None:
    with bordered():
        st.markdown(f"### {int(row['Posición'])}. {row['Repositorio recomendado']}")
        st.progress(min(max(float(row["Score"]), 0.0), 1.0))
        c1, c2 = st.columns(2)
        c1.metric("Score", fmt_float(row["Score"], 4))
        c2.metric("Lenguaje", row["Lenguaje"])
        st.write(f"**Clúster:** {row['Clúster']}")
        st.write(f"**Estrellas:** {fmt_int(row['Estrellas'])} | **Forks:** {fmt_int(row['Forks'])}")
        if row.get("Topics compartidos"):
            st.write(f"**Topics compartidos:** {row['Topics compartidos']}")
        st.write(f"**Motivo:** {row['Motivo de recomendación']}")
        if isinstance(row.get("URL", ""), str) and row["URL"].strip():
            st.markdown(f"[Abrir en GitHub]({row['URL']})")


def _signal_breakdown(row: pd.Series) -> None:
    with st.expander(f"Desglose de señales: {row['Repositorio recomendado']}"):
        for label, weight_key, column in SIGNALS:
            value = float(row.get(column, 0.0))
            weight = HYBRID_WEIGHTS[weight_key]
            st.caption(f"{label}: {fmt_float(value, 4)}  ·  peso {weight:.0%}  ·  aporta {fmt_float(value * weight, 4)}")
            st.progress(min(max(value, 0.0), 1.0))


def _similar_tab(ctx: dict) -> None:
    catalog, state = ctx["catalog"], ctx["recommender"]
    st.write("Elige un repositorio y un método; el sistema devuelve los repositorios más parecidos.")

    method = st.radio(
        "Método de recomendación", list(METHODS), format_func=lambda key: METHODS[key],
        help="Los cuatro métodos se evalúan en la sección Evaluación.",
    )
    st.caption(METHOD_HELP[method])
    if method == "popularity":
        st.info("El baseline de popularidad devuelve la misma lista para cualquier repositorio; sirve como punto de comparación.")
    if method == "hybrid":
        st.caption(
            "Popularidad, actividad, crecimiento y clúster priorizan repositorios vigentes; "
            "según la evaluación offline no aumentan la precisión temática (ver Evaluación › señales)."
        )

    filtered = _repository_filters(catalog)
    if filtered.empty:
        st.warning("No hay repositorios con los filtros seleccionados.")
        return
    options = {f"{r['nombre']} — {r['lenguaje']} — {fmt_int(r['stars'])} estrellas": r["nombre"] for _, r in filtered.iterrows()}
    choice = st.selectbox("Repositorio de referencia", list(options), help="Es el punto de partida del ranking.")
    repository = options[choice]
    top_n = st.slider("Número de recomendaciones", min_value=5, max_value=20, value=10, step=1)

    index = catalog.index[catalog["nombre"] == repository][0]
    _repository_card(get_repository_row(repository, state), state["cluster_names"].get(int(state["clusters"][index]), "—"))

    results = recommend(repository, method, top_n=top_n, recommender=state)
    st.subheader("Recomendaciones principales")
    if results.empty:
        st.warning("No se encontraron recomendaciones.")
        return
    for column, (_, row) in zip(st.columns(3), results.head(3).iterrows()):
        with column:
            _recommendation_card(row)
    if method == "hybrid":
        st.subheader("Detalle del Top 5")
        for _, row in results.head(5).iterrows():
            _signal_breakdown(row)
    st.subheader("Tabla completa")
    show_table(results[["Posición", "Repositorio recomendado", "Score", "Lenguaje", "Estrellas", "Forks", "Clúster", "Topics compartidos", "Motivo de recomendación"]])


def _preferences_tab(ctx: dict) -> None:
    catalog, state = ctx["catalog"], ctx["recommender"]
    st.write("Configura lenguaje, topics, estrellas y actividad para encontrar repositorios que cumplan tus criterios.")
    with st.expander("¿Cómo se calcula esta búsqueda?"):
        st.markdown(
            "Primero se filtra por tus criterios y luego se ordena con: 40 % coincidencia de topics, 20 % lenguaje, "
            "15 % popularidad, 15 % actividad y 10 % crecimiento. No parte de un repositorio de referencia."
        )

    languages = ["Cualquiera"] + sorted(catalog["lenguaje"].dropna().astype(str).unique().tolist())
    cluster_names = state["cluster_names"]
    cluster_labels = ["Cualquiera"] + [f"{cid} · {name}" for cid, name in sorted(cluster_names.items())]
    topic_options = sorted({t for topics in catalog["topics_list"] for t in topics})
    max_days = int(catalog["dias_sin_commit"].max())

    left, right = st.columns(2)
    with left:
        language = st.selectbox("Lenguaje principal", languages)
        topics = st.multiselect("Topics", topic_options, help="Elige uno o varios.")
        min_stars = st.number_input("Estrellas mínimas", min_value=0, value=0, step=50)
    with right:
        max_days_without_commit = st.slider("Máximo de días sin commit", min_value=0, max_value=max_days, value=min(180, max_days))
        cluster_label = st.selectbox("Clúster (opcional)", cluster_labels)
        top_n = st.slider("Cantidad de resultados", min_value=5, max_value=20, value=10)

    results = recommend_by_preferences(
        language=None if language == "Cualquiera" else language,
        topics=topics,
        min_stars=min_stars,
        max_days_without_commit=max_days_without_commit,
        cluster=None if cluster_label == "Cualquiera" else int(cluster_label.split(" · ")[0]),
        top_n=top_n,
        recommender=state,
    )
    if results.empty:
        st.warning("No hay resultados con los filtros seleccionados.")
        return
    st.subheader("Resultados")
    show_table(results[["Posición", "Repositorio", "Score de preferencias", "Lenguaje", "Estrellas", "Forks", "Días sin commit", "Clúster", "Topics coincidentes", "Motivo"]])


def render(ctx: dict) -> None:
    st.header("Recomendador de repositorios")
    tab_similar, tab_preferences = st.tabs(["Similar a un repositorio", "Buscar por preferencias"])
    with tab_similar:
        _similar_tab(ctx)
    with tab_preferences:
        _preferences_tab(ctx)

"""Resumen: de dónde salen los datos y cómo se conectan las etapas con el dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.ui import bordered, fmt_float, fmt_int, show_image, show_table
from src.common import REPORTS_DIR, SNAPSHOT_DATE, SNAPSHOT_SOURCE

PIPELINE = [
    ("1. Extracción", "src/ingest.py", "data/processed/repos_clean.csv", "Resumen"),
    ("2. Análisis exploratorio", "notebook 01", "reports/eda/*.png", "Resumen"),
    ("3. Reducción de dimensión (PCA)", "notebook 02", "reports/pca/*.png y coordenadas pca_x / pca_y", "Clústeres y PCA"),
    ("4. Clustering (K-means)", "notebook 03 + src/features.py", "repos_clustered.csv y reports/clustering/*.png", "Clústeres y PCA, Recomendador"),
    ("5. Recomendación", "src/recommender.py", "4 métodos: popularidad, contenido, SVD, híbrido", "Recomendador"),
    ("6. Evaluación offline", "src/evaluation.py", "métricas, robustez, ablación, calidad de datos", "Evaluación"),
    ("7. Análisis de grafos", "notebook 05 + src/graph_service.py", "artifacts/graph/*.csv", "Grafo interactivo"),
]


def render(ctx: dict) -> None:
    catalog: pd.DataFrame = ctx["catalog"]

    st.header("De los datos de GitHub a las recomendaciones")
    st.write(
        "El ecosistema de software de IA crece de forma rápida y desordenada. Este proyecto reúne repositorios "
        "públicos de investigación en IA, los analiza (texto, topics, actividad y popularidad) y ofrece "
        "recomendaciones de repositorios parecidos, además de un mapa de la red de topics que los conecta."
    )

    with bordered():
        st.markdown(f"**Fuente de datos:** {SNAPSHOT_SOURCE}.")
        st.caption(
            f"Fotografía del {SNAPSHOT_DATE}: repositorios públicos con al menos 50 estrellas y topics de investigación "
            "en IA. Los datos son reales y se extrajeron con la API oficial; no se recopiló información personal. "
            "Ver `data/README.md` para el detalle y `src/ingest.py` para repetir la extracción."
        )

    inactive = float((catalog["dias_sin_commit"] > 365).mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Repositorios", fmt_int(len(catalog)))
    c2.metric("Lenguajes distintos", fmt_int(catalog["lenguaje"].nunique()))
    c3.metric("Mediana de estrellas", fmt_int(catalog["stars"].median()))
    c4.metric("Sin commits hace más de 1 año", f"{inactive:.0%}")

    st.subheader("Cómo se conectan las etapas")
    st.caption("Cada etapa del proyecto alimenta una sección de este dashboard.")
    show_table(pd.DataFrame(PIPELINE, columns=["Etapa", "Dónde está", "Qué produce", "Se ve en"]))

    st.subheader("Análisis exploratorio")
    figures = [
        ("Estrellas", "distribution_stars.png", "Distribución de estrellas (escala log): muy sesgada, pocos repositorios concentran la popularidad."),
        ("Lenguajes", "top_languages.png", "Lenguajes principales."),
        ("Actividad", "activity_status.png", "Estado de actividad según el último commit."),
        ("Último commit", "days_since_last_commit.png", "Días desde el último commit."),
        ("Año de creación", "creation_year.png", "Repositorios por año de creación."),
    ]
    for tab, (_, filename, caption) in zip(st.tabs([f[0] for f in figures]), figures):
        with tab:
            show_image(REPORTS_DIR / "eda" / filename, caption)

    report = REPORTS_DIR / "final_report.pdf"
    if report.exists():
        st.download_button("Descargar el reporte final (PDF)", report.read_bytes(), file_name="final_report.pdf", mime="application/pdf")

"""Clústeres y PCA: cómo se agrupan los repositorios y por qué."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.ui import PALETTE, bordered, new_figure, show_figure, show_image, show_table
from src.common import CLUSTER_PROFILE_PATH, DATA_QUALITY_PATH, REPORTS_DIR


def _scatter(catalog: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, names: dict) -> None:
    fig, ax = new_figure(7.5, 4.8)
    ax.grid(axis="both", color="#e5e7eb", linewidth=0.6)
    sizes = catalog["cluster_kmeans"].value_counts()
    for cid in sorted(sizes.index, key=lambda c: -sizes[c]):  # los grandes primero, los pequeños encima
        part = catalog[catalog["cluster_kmeans"] == cid]
        small = len(part) < 100
        ax.scatter(
            part[x], part[y], s=46 if small else 10, alpha=0.9 if small else 0.45,
            color=PALETTE[int(cid) % len(PALETTE)], edgecolor="white" if small else "none", linewidth=0.5,
            label=f"{names.get(int(cid), cid)} ({len(part):,})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    show_figure(fig)


def render(ctx: dict) -> None:
    catalog, state, read_csv = ctx["catalog"], ctx["recommender"], ctx["read_csv"]
    names = state["cluster_names"]

    st.header("Clústeres y PCA")
    st.write(
        "Cada repositorio se describe con variables numéricas, lenguaje y los 30 topics más frecuentes; se reduce a "
        "10 componentes con PCA y se agrupa con K-means (K = 5). La pertenencia al clúster es una de las señales del "
        "recomendador híbrido."
    )

    st.subheader("Perfil de cada clúster")
    profile = read_csv(str(CLUSTER_PROFILE_PATH))
    if profile is not None:
        show_table(profile)
        smallest = profile.loc[profile["repositorios"].idxmin()]
        with bordered():
            st.warning(
                f"Los clústeres no están balanceados: «{smallest['cluster_nombre']}» tiene solo "
                f"{int(smallest['repositorios'])} repositorios. Estrellas y forks entran en PCA sin escala logarítmica, "
                "así que unos pocos repositorios gigantes definen clústeres propios."
            )
            quality = read_csv(str(DATA_QUALITY_PATH))
            if quality is not None and len(quality) >= 3:
                log_row = quality.iloc[2]
                st.caption(
                    "En el experimento de calidad de datos (sección Evaluación), aplicar log a estrellas, forks y estrellas por día "
                    f"deja el clúster más pequeño en {int(log_row['clúster más pequeño'])} repositorios. "
                    "No se aplicó al clustering publicado para mantener la coherencia con los reportes."
                )

    st.subheader("Mapas de los repositorios")
    tab_space, tab_embed = st.tabs(["Espacio del clustering (PC1 vs PC2)", "Mapa de embeddings de texto"])
    with tab_space:
        st.caption("Las dos primeras componentes PCA de las 10 que usa K-means.")
        _scatter(catalog, "pca_kmeans_1", "pca_kmeans_2", "Componente 1", "Componente 2", names)
    with tab_embed:
        if {"pca_x", "pca_y"} <= set(catalog.columns):
            st.caption("Proyección 2D de los embeddings de la descripción (notebook 02), coloreada por clúster.")
            _scatter(catalog, "pca_x", "pca_y", "PCA x", "PCA y", names)
        else:
            st.caption("No hay coordenadas pca_x / pca_y en el catálogo.")

    st.subheader("Cómo se eligió K y cómo se validó")
    tabs = st.tabs(["Elección de K", "Validación", "Iteraciones", "Resultado final", "Varianza PCA"])
    images = [
        (REPORTS_DIR / "clustering" / "kmeans_sweep.png", "Barrido de K: inercia y silhouette."),
        (REPORTS_DIR / "clustering" / "kmeans_validation.png", "Validación del clustering."),
        (REPORTS_DIR / "clustering" / "kmeans_iterations.png", "Convergencia de K-means."),
        (REPORTS_DIR / "clustering" / "kmeans_final.png", "Clustering final con K = 5."),
        (REPORTS_DIR / "pca" / "pca_variance.png", "Varianza explicada acumulada por PCA."),
    ]
    for tab, (path, caption) in zip(tabs, images):
        with tab:
            show_image(path, caption)

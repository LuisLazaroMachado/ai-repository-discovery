"""Red de repositorios por topics compartidos."""
from __future__ import annotations

import streamlit.components.v1 as components
import streamlit as st

from dashboard.ui import bordered, fmt_float, fmt_int, show_image, show_table
from src.common import REPORTS_DIR
from src.graph_service import build_interactive_subgraph


def render(ctx: dict) -> None:
    st.header("Red temática de repositorios")
    st.write(
        "Cada nodo es un repositorio y cada conexión indica topics compartidos; el grosor de la línea es la cantidad "
        "de topics en común. PageRank y las demás medidas muestran qué repositorios son estructuralmente importantes."
    )

    with bordered():
        a, b, c4, d = st.columns(4)
        a.markdown("**Nodo rojo**"); a.caption("Repositorio seleccionado.")
        b.markdown("**Nodos azules**"); b.caption("Vecinos; el tamaño crece con PageRank.")
        c4.markdown("**Línea gruesa**"); c4.caption("Más topics compartidos.")
        d.markdown("**Tooltip**"); d.caption("Pasa el cursor sobre un nodo para ver sus métricas.")

    nodes = ctx["graph_nodes"]
    st.caption(
        f"El grafo conecta repositorios que comparten al menos 4 topics, por eso solo {len(nodes):,} de los "
        f"{len(ctx['catalog']):,} repositorios del catálogo aparecen aquí."
    )
    default = nodes.index("tensorflow/tensorflow") if "tensorflow/tensorflow" in nodes else 0
    left, middle, right = st.columns([2.4, 1, 1])
    repository = left.selectbox("Repositorio central", nodes, index=default, key="graph_repository")
    min_weight = middle.slider("Peso mínimo", min_value=4, max_value=12, value=4, help="Topics compartidos mínimos por arista (el grafo parte de 4).")
    max_neighbors = right.slider("Vecinos máximos", min_value=5, max_value=30, value=12)

    payload = build_interactive_subgraph(repository, max_neighbors=max_neighbors, min_weight=min_weight)
    summary = payload["selected_summary"]

    with bordered():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PageRank", fmt_float(summary.get("pagerank", 0.0), 6))
        m2.metric("Weighted degree", fmt_float(summary.get("weighted_degree", 0.0), 3))
        betweenness = summary.get("betweenness")
        m3.metric("Betweenness", "N/A" if betweenness is None else fmt_float(betweenness, 6))
        m4.metric("Estrellas", "N/A" if summary.get("stars") is None else fmt_int(summary["stars"]))
        st.caption("PageRank: importancia estructural · Weighted degree: intensidad de conexiones · Betweenness: capacidad de conectar grupos.")

    if payload["html"]:
        components.html(payload["html"], height=760, scrolling=True)
    else:
        st.warning("No fue posible construir el subgrafo con los filtros seleccionados.")

    st.subheader("Vecinos")
    if payload["neighbors_table"].empty:
        st.info("No hay vecinos con el filtro actual.")
    else:
        show_table(payload["neighbors_table"].rename(columns={"Repositorio": "Repositorio vecino", "Peso": "Topics compartidos"}))

    with bordered():
        st.subheader("Popularidad frente a importancia estructural")
        st.write("Muchas estrellas no implican ser el más central de la red: PageRank mide conectividad temática, no popularidad.")
        st.caption("«rank_stars» es la posición por estrellas entre los 3034 repositorios del catálogo; PageRank y betweenness se calculan sobre los 1861 nodos del grafo.")
        show_table(ctx["pagerank_vs_stars"])
        left_img, right_img = st.columns(2)
        with left_img:
            show_image(REPORTS_DIR / "graph" / "ranking_comparison.png", "Ranking por PageRank frente a ranking por estrellas.")
        with right_img:
            show_image(REPORTS_DIR / "graph" / "graph_visualization.png", "Visualización del grafo completo.")

    with st.expander("Tabla de centralidades"):
        show_table(ctx["centrality"][["nombre", "weighted_degree", "pagerank", "betweenness", "stars"]])

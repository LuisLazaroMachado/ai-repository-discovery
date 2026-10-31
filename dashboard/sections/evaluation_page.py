"""Evaluación offline: qué se mide, qué resultados salen y cuáles son sus límites."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.ui import MUTED, PALETTE, bordered, new_figure, show_figure, show_table
from src import common as c
from src.recommender import recommend

SHORT = {
    "Popularity baseline": "Popularidad",
    "Content-based (description only)": "Contenido\n(solo descripción)",
    "Content-based baseline": "Contenido\n(texto + topics)",
    "Hybrid PCA + K-means ranking": "Híbrido",
    "SVD (repo-topic matrix)": "SVD",
}
ORDER = list(SHORT)


def _ordered(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary[summary["model"].isin(ORDER)].copy()
    summary["_order"] = summary["model"].map({m: i for i, m in enumerate(ORDER)})
    return summary.sort_values("_order").drop(columns="_order")


def _bars(summary: pd.DataFrame, columns: list, title: str, labels: list) -> None:
    summary = _ordered(summary)
    fig, ax = new_figure(7.8, 4.2)
    width = 0.8 / len(columns)
    xs = range(len(summary))
    for i, (column, label) in enumerate(zip(columns, labels)):
        positions = [x + (i - (len(columns) - 1) / 2) * width for x in xs]
        bars = ax.bar(positions, summary[column], width=width * 0.92, color=PALETTE[i], label=label)
        for bar, value in zip(bars, summary[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([SHORT[m] for m in summary["model"]], fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Valor (0 a 1)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=len(columns))
    show_figure(fig)


def _protocol_bars(main: pd.DataFrame, robust: pd.DataFrame) -> None:
    main, robust = _ordered(main), _ordered(robust)
    fig, ax = new_figure(7.8, 4.2)
    xs = list(range(len(main)))
    width = 0.38
    for offset, (frame, label, color) in enumerate(
        [(main, "Relevancia con los 100 topics más frecuentes", PALETTE[0]), (robust, "Relevancia solo con topics del puesto 31 al 100", PALETTE[1])]
    ):
        positions = [x + (offset - 0.5) * width for x in xs]
        bars = ax.bar(positions, frame["Precision@10"], width=width * 0.92, color=color, label=label)
        for bar, value in zip(bars, frame["Precision@10"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in main["model"]], fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Precision@10")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    show_figure(fig)


def _ablation_bars(ablation: pd.DataFrame) -> None:
    table = ablation[ablation["variante"] != "Híbrido completo"].iloc[::-1]
    fig, ax = new_figure(7.8, 4.0)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    values = table["Δ Precision@10"]
    bars = ax.barh(table["variante"], values, color=[PALETTE[0] if v >= 0 else PALETTE[4] for v in values])
    for bar, value in zip(bars, values):
        ax.text(value + (0.006 if value >= 0 else -0.006), bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Cambio en Precision@10 respecto al híbrido completo (azul: mejora; naranja: empeora)")
    lo, hi = float(values.min()), float(values.max())
    ax.set_xlim(lo - 0.08, hi + 0.06)
    show_figure(fig)


def _cases(ctx: dict, title: str, path, key: str) -> None:
    frame = ctx["read_csv"](str(path))
    if frame is None:
        return
    st.markdown(f"**{title}**")
    show_table(frame[["query_repo", "relevant_candidates", "Precision@10", "NDCG@10", "diagnosis"]].rename(
        columns={"query_repo": "Consulta", "relevant_candidates": "Candidatos relevantes", "diagnosis": "Diagnóstico"}
    ))
    pick = st.selectbox("Ver las recomendaciones híbridas de este caso", list(frame["query_repo"]), key=key)
    results = recommend(pick, "hybrid", top_n=5, recommender=ctx["recommender"])
    show_table(results[["Posición", "Repositorio recomendado", "Lenguaje", "Estrellas", "Topics compartidos"]])


def render(ctx: dict) -> None:
    read_csv = ctx["read_csv"]
    st.header("Evaluación offline")
    st.write(
        "El dataset no tiene usuarios, clics ni valoraciones, así que la relevancia es un *proxy*: un repositorio "
        "recomendado cuenta como relevante si comparte topics significativos con el de referencia (2 topics si la "
        "referencia tiene 3 o más; 1 si tiene menos). Se evalúan 500 consultas de las evaluables con semilla fija, "
        "y todos los sistemas usan el mismo conjunto de candidatos."
    )

    summary = read_csv(str(c.EVAL_SUMMARY_PATH))
    if summary is None:
        st.error("Falta artifacts/recommendation/offline_evaluation_summary.csv; ejecuta `python -m src.build_artifacts`.")
        return

    st.subheader("1. Comparación de sistemas")
    _bars(summary, ["Precision@10", "NDCG@10"], "Precision@10 y NDCG@10 por sistema", ["Precision@10", "NDCG@10"])
    table = _ordered(summary).assign(Sistema=lambda d: d["model"].map(lambda m: SHORT[m].replace("\n", " ")))
    show_table(table[["Sistema", "Precision@10", "Recall@10", "MAP@10", "NDCG@10", "Coverage@10"]].round(4))
    st.caption(
        "Recall@10 es bajo en todos los sistemas porque cada consulta tiene cientos de candidatos relevantes y solo se muestran 10. "
        "Coverage@10 es la fracción del catálogo que aparece en alguna recomendación."
    )

    st.subheader("2. ¿Los resultados son confiables? Prueba de robustez")
    with bordered():
        st.warning(
            "Los métodos que construyen su representación con topics (SVD, PCA y K-means, y en parte el contenido) "
            "están favorecidos cuando la relevancia también se define con topics. Por eso el SVD parece el mejor arriba."
        )
    robust = read_csv(str(c.ROBUSTNESS_PATH))
    if robust is not None:
        _protocol_bars(summary, robust)
        st.caption(
            "Al medir la relevancia solo con topics que el SVD y el PCA nunca vieron, el SVD cae de forma drástica y el "
            "contenido de texto pasa a ser el mejor. El híbrido queda por debajo del contenido en esta prueba. "
            "Conclusión honesta: sus señales extra no mejoran la precisión temática; aportan vigencia y popularidad, "
            "algo que este proxy no puede medir."
        )

    st.subheader("3. ¿Qué aporta cada señal del híbrido?")
    ablation = read_csv(str(c.ABLATION_PATH))
    if ablation is not None:
        _ablation_bars(ablation)
        st.caption(
            "El contenido textual sostiene casi todo el resultado y la similitud PCA aporta una mejora clara. "
            "Quitar popularidad, actividad, crecimiento o clúster sube un poco la precisión temática: esas señales "
            "son una decisión de producto (priorizar repositorios vigentes), no una mejora medida."
        )
        with st.expander("Ver tabla completa"):
            show_table(ablation.round(4))

    st.subheader("4. Impacto de las decisiones sobre los datos")
    quality = read_csv(str(c.DATA_QUALITY_PATH))
    if quality is not None:
        show_table(quality)
        st.caption(
            "Imputar los valores vacíos conserva casi 300 repositorios y mantiene una precisión algo mayor que descartarlos. "
            "Aplicar log a las variables con cola pesada mejora un poco la precisión y balancea los clústeres."
        )

    st.subheader("5. Casos fuertes y fallidos del híbrido")
    left, right = st.tabs(["Casos fuertes", "Casos fallidos"])
    with left:
        _cases(ctx, "Las 5 consultas mejor resueltas", c.STRONG_CASES_PATH, "strong_pick")
    with right:
        _cases(ctx, "Las 5 consultas peor resueltas", c.FAILURE_CASES_PATH, "failure_pick")

    old = read_csv(str(c.SYSTEMS_COMPARISON_PATH))
    if old is not None:
        with st.expander("Nota sobre la comparación de la semana 11"):
            st.write(
                "El reporte incluye una tabla (baseline 0.108, contenido 0.272, SVD 0.670) calculada con 50 consultas aleatorias, "
                "relevancia con todos los topics y TF-IDF de la descripción sola. No es comparable con las tablas de arriba, "
                "que usan un único protocolo para todos los sistemas."
            )
            show_table(old)

    with st.expander("Limitaciones"):
        st.markdown(
            "- La relevancia es un proxy basado en topics: no mide satisfacción real de usuarios.\n"
            "- Hay topics genéricos (como `python` o `deep-learning`) que pueden inflar coincidencias.\n"
            "- Algunos «fallos» son límites del proxy: recomendaciones razonables que comparten topics fuera del vocabulario evaluado.\n"
            "- Es una fotografía de datos de un solo día; las estrellas y la actividad cambian."
        )

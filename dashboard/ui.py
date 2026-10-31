"""Utilidades de interfaz compartidas por las secciones del dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# Paleta categórica apta para daltonismo (Okabe-Ito), un color por clúster / sistema.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#999999"]
PRIMARY = PALETTE[0]
MUTED = "#8a8f98"


def bordered():
    """Contenedor con borde (compatible con versiones antiguas de Streamlit)."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def show_table(frame: pd.DataFrame, **kwargs) -> None:
    """Tabla a ancho completo sin depender del nombre del parámetro (cambia entre versiones)."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.dataframe(frame, hide_index=True, **extra, **kwargs)
            return
        except TypeError:
            continue


def show_image(path: Path, caption: Optional[str] = None) -> bool:
    """Muestra una imagen de reports/ si existe; devuelve False si falta."""
    if not Path(path).exists():
        st.caption(f"(No se encontró {Path(path).name}; ejecuta el notebook correspondiente.)")
        return False
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(str(path), caption=caption, **extra)
            return True
        except TypeError:
            continue
    return False


def show_figure(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def new_figure(width: float = 7.5, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax

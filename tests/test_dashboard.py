from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pyvis")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
SECTIONS = ["Resumen", "Recomendador", "Clústeres y PCA", "Evaluación", "Grafo interactivo"]


def test_every_section_renders_without_errors():
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
    for section in SECTIONS:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception, section

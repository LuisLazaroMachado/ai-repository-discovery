"""Extracción de repositorios de IA desde la API de GitHub (PyGithub) y limpieza.

    python src/ingest.py                       # extrae una nueva fotografía en data/snapshots/<fecha>/
    python src/ingest.py --max-per-query 300   # menos repositorios por consulta (prueba rápida)
    python src/ingest.py --promote             # además reemplaza data/processed/repos_clean.csv

Por defecto NUNCA toca `data/processed/repos_clean.csv`: ese archivo es la fotografía oficial
(30-abr-2026, 3034 repositorios) sobre la que se calcularon todas las métricas del proyecto.
Una extracción nueva traerá otras estrellas y otros repositorios, así que sus resultados no
serán idénticos; para usarla en el dashboard ejecuta `--promote` y luego
`python -m src.build_artifacts` (y el notebook 05 para el grafo).

Requiere un token en la variable de entorno GITHUB_TOKEN (ver .env.example). La búsqueda de GitHub
devuelve como máximo 1000 resultados por consulta, por eso hay varias consultas por tema.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import CATALOG_PATH  # noqa: E402

MIN_STARS = 50

# Las 4 primeras son las consultas de la primera iteración del proyecto. La lista usada para la
# fotografía de 3034 repositorios no se conservó, así que las demás son una reconstrucción razonable
# (mismos temas de investigación en IA). Se pueden sustituir con --queries-file (una consulta por línea).
DEFAULT_QUERIES = [
    "topic:machine-learning topic:research",
    "topic:deep-learning topic:paper",
    "topic:nlp topic:research",
    "topic:computer-vision topic:research",
    "topic:large-language-models topic:research",
    "topic:llm topic:paper",
    "topic:reinforcement-learning topic:research",
    "topic:generative-ai topic:research",
    "topic:transformers topic:paper",
    "topic:diffusion-models topic:research",
    "topic:graph-neural-networks topic:paper",
    "topic:speech-recognition topic:research",
    "topic:recommender-system topic:research",
    "topic:pytorch topic:paper",
    "topic:tensorflow topic:research",
    "topic:awesome topic:deep-learning",
]

RAW_COLUMNS = [
    "repo_id", "nombre", "descripcion", "lenguaje", "stars", "forks",
    "topics", "fecha_creacion", "ultimo_commit", "url",
]
DERIVED_COLUMNS = ["dias_sin_commit", "antiguedad_dias", "num_topics", "ratio_forks_stars", "stars_por_dia"]


# --------------------------------------------------------------------------- #
# Limpieza e ingeniería de variables (función pura: se puede probar sin red)
# --------------------------------------------------------------------------- #
def clean_and_engineer_features(df: pd.DataFrame, snapshot: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Rellena vacíos, convierte fechas y crea las 5 variables derivadas del dataset limpio.

    - dias_sin_commit / antiguedad_dias: días enteros entre `snapshot` y el último commit / la creación.
    - num_topics: cantidad de topics del repositorio.
    - ratio_forks_stars: forks / estrellas (4 decimales).
    - stars_por_dia: estrellas / antigüedad en días (4 decimales).
    """
    df = df.copy()
    df["descripcion"] = df["descripcion"].fillna("No description")
    df["lenguaje"] = df["lenguaje"].fillna("Unknown")
    df["fecha_creacion"] = pd.to_datetime(df["fecha_creacion"], utc=True)
    df["ultimo_commit"] = pd.to_datetime(df["ultimo_commit"], utc=True)

    snapshot = pd.Timestamp.now(tz="UTC") if snapshot is None else pd.Timestamp(snapshot)
    df["dias_sin_commit"] = (snapshot - df["ultimo_commit"]).dt.days
    df["antiguedad_dias"] = (snapshot - df["fecha_creacion"]).dt.days

    from src.common import parse_topics

    df["num_topics"] = df["topics"].apply(lambda t: len(parse_topics(t)))
    df["ratio_forks_stars"] = (df["forks"] / df["stars"].replace(0, np.nan)).fillna(0).round(4)
    df["stars_por_dia"] = (df["stars"] / df["antiguedad_dias"].replace(0, 1)).round(4)
    return df


# --------------------------------------------------------------------------- #
# Extracción desde la API
# --------------------------------------------------------------------------- #
def connect_github(token: Optional[str] = None):
    from github import Auth, Github

    token = token or os.getenv("GITHUB_TOKEN")
    if not token:
        raise SystemExit(
            "Falta GITHUB_TOKEN. Crea un token en https://github.com/settings/tokens "
            "y guárdalo en un archivo .env (ver .env.example)."
        )
    return Github(auth=Auth.Token(token), per_page=100)


def repository_to_record(repo) -> dict:
    """Convierte un objeto Repository de PyGithub en una fila del dataset crudo."""
    topics = getattr(repo, "topics", None)
    if topics is None:  # versiones antiguas de PyGithub: una llamada extra por repositorio
        topics = repo.get_topics()
    return {
        "repo_id": repo.id,
        "nombre": repo.full_name,
        "descripcion": repo.description,
        "lenguaje": repo.language,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "topics": list(topics),
        "fecha_creacion": repo.created_at,
        "ultimo_commit": repo.pushed_at,
        "url": repo.html_url,
    }


def _wait_for_rate_limit(client) -> None:
    reset = client.get_rate_limit().resources.search.reset
    wait = max((reset - datetime.now(timezone.utc)).total_seconds(), 0) + 5
    print(f"  límite de búsqueda alcanzado; esperando {wait:.0f} s...")
    time.sleep(wait)


def fetch_repositories(client, queries: Iterable[str], max_per_query: int = 1000, min_stars: int = MIN_STARS) -> pd.DataFrame:
    from github import RateLimitExceededException

    records, seen = [], set()
    for query in queries:
        full_query = f"{query} stars:>{min_stars}"
        print(f"Consulta: {full_query}")
        added = 0
        while True:
            try:
                results = client.search_repositories(query=full_query, sort="stars", order="desc")
                for position, repo in enumerate(results):
                    if position >= max_per_query:
                        break
                    if repo.id in seen:
                        continue
                    seen.add(repo.id)
                    records.append(repository_to_record(repo))
                    added += 1
                break
            except RateLimitExceededException:
                _wait_for_rate_limit(client)
        print(f"  +{added} nuevos (total {len(records)})")
    return pd.DataFrame(records, columns=RAW_COLUMNS)


# --------------------------------------------------------------------------- #
def read_queries(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_QUERIES
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-per-query", type=int, default=1000, help="máximo por consulta (GitHub limita a 1000)")
    parser.add_argument("--queries-file", help="archivo de texto con una consulta por línea")
    parser.add_argument("--output-dir", help="carpeta de salida (por defecto data/snapshots/<fecha>)")
    parser.add_argument("--promote", action="store_true", help="copia el resultado a data/processed/repos_clean.csv")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    started = datetime.now(timezone.utc)
    queries = read_queries(args.queries_file)
    raw = fetch_repositories(connect_github(), queries, args.max_per_query)
    clean = clean_and_engineer_features(raw, snapshot=pd.Timestamp.now(tz="UTC"))

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "snapshots" / started.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "repos_raw.csv", index=False)
    clean.to_csv(out_dir / "repos_clean.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "extracted_at_utc": started.isoformat(),
                "source": "GitHub REST API v3 (search/repositories) via PyGithub",
                "queries": queries,
                "min_stars": MIN_STARS,
                "repositories": len(clean),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nFotografía guardada en {out_dir} ({len(clean)} repositorios).")

    if args.promote:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(CATALOG_PATH, index=False)
        print(f"Promovida a {CATALOG_PATH.relative_to(PROJECT_ROOT)}. Ahora ejecuta: python -m src.build_artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())

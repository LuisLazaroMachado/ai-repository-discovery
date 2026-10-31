import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import common as c
from src.ingest import RAW_COLUMNS, clean_and_engineer_features, repository_to_record

SNAPSHOT = pd.Timestamp("2026-04-30 05:10", tz="UTC")


def test_cleaning_fills_gaps_and_builds_derived_columns():
    raw = pd.DataFrame(
        {
            "repo_id": [1, 2], "nombre": ["a/x", "b/y"], "descripcion": [None, "algo"], "lenguaje": ["Python", None],
            "stars": [100, 0], "forks": [10, 0], "topics": [["ai", "nlp"], []],
            "fecha_creacion": ["2026-04-20T05:10:00Z", "2025-04-30T05:10:00Z"],
            "ultimo_commit": ["2026-04-29T05:10:00Z", "2026-04-30T05:10:00Z"], "url": ["u1", "u2"],
        }
    )
    clean = clean_and_engineer_features(raw, snapshot=SNAPSHOT)
    assert clean.loc[0, "descripcion"] == "No description" and clean.loc[1, "lenguaje"] == "Unknown"
    assert clean["dias_sin_commit"].tolist() == [1, 0]
    assert clean["antiguedad_dias"].tolist() == [10, 365]
    assert clean["num_topics"].tolist() == [2, 0]
    assert clean["ratio_forks_stars"].tolist() == [0.1, 0.0]  # sin división por cero
    assert clean["stars_por_dia"].tolist() == [10.0, 0.0]


def test_cleaning_reproduces_the_published_dataset():
    published = pd.read_csv(c.CATALOG_PATH)
    clean = clean_and_engineer_features(published[RAW_COLUMNS], snapshot=SNAPSHOT)
    assert (clean["num_topics"].values == published["num_topics"].values).all()
    assert np.allclose(clean["ratio_forks_stars"], published["ratio_forks_stars"], atol=1e-9)
    # la extracción duró unos minutos: unas pocas filas cambian de día justo en el borde
    assert (clean["dias_sin_commit"].values != published["dias_sin_commit"].values).mean() < 0.005
    assert (clean["antiguedad_dias"].values != published["antiguedad_dias"].values).mean() < 0.005


def _fake_repo(repo_id, topics=None):
    repo = types.SimpleNamespace(
        id=repo_id, full_name=f"owner/repo{repo_id}", description=None, language="Python", stargazers_count=60,
        forks_count=5, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        pushed_at=datetime(2026, 1, 1, tzinfo=timezone.utc), html_url=f"https://github.com/owner/repo{repo_id}",
    )
    repo.topics = topics if topics is not None else ["ai"]
    return repo


def test_repository_to_record_matches_the_raw_schema():
    record = repository_to_record(_fake_repo(7))
    assert list(record) == RAW_COLUMNS and record["topics"] == ["ai"]


def test_fetch_deduplicates_and_respects_the_limit():
    pytest.importorskip("github")
    from src.ingest import fetch_repositories

    class FakeClient:
        def search_repositories(self, query, sort=None, order=None):
            return [_fake_repo(i) for i in range(1, 6)] if "one" in query else [_fake_repo(i) for i in range(4, 9)]

    frame = fetch_repositories(FakeClient(), ["topic:one", "topic:two"], max_per_query=3)
    assert frame["repo_id"].tolist() == [1, 2, 3, 4, 5, 6]  # 4 y 5 se repiten solo una vez

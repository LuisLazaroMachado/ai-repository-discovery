import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src import common as c
from src.features import add_clusters, name_clusters, prepare_frame


def test_clustering_reproduces_the_published_partition():
    clean = pd.read_csv(c.CATALOG_PATH)
    published = pd.read_csv(c.CLUSTERED_CATALOG_PATH)
    rebuilt, stats = add_clusters(prepare_frame(clean))
    assert adjusted_rand_score(published["cluster_kmeans"], rebuilt["cluster_kmeans"]) > 0.999
    assert sorted(np.bincount(rebuilt["cluster_kmeans"])) == sorted(np.bincount(published["cluster_kmeans"]))
    assert 0 < stats["silhouette"] < 1


def test_published_names_apply_only_when_the_partition_matches():
    clean = pd.read_csv(c.CATALOG_PATH)
    published = pd.read_csv(c.CLUSTERED_CATALOG_PATH)
    rebuilt, _ = add_clusters(prepare_frame(clean))
    assert set(name_clusters(rebuilt, published)["cluster_nombre"]) == set(published["cluster_nombre"])
    assert name_clusters(rebuilt, None)["cluster_nombre"].str.startswith("Cluster ").all()

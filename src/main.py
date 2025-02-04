import random
from typing import Dict, List

import numpy as np
import sklearn.decomposition
from sklearn.cluster import DBSCAN
from supervisely import Application, logger
import umap
from fastapi import Request

from src.utils import timeit


app = Application()
server = app.get_server()


class ReductionMethod:
    PCA = "pca"
    UMAP = "umap"
    PCA_UMAP = "pca-umap"
    TSNE = "t-sne"
    PCA_TSNE = "pca-t-sne"


class ClusteringMethod:
    DBSCAN = "dbscan"


class SamplingMethod:
    RANDOM = "random"
    CENTROIDS = "centroids"


@timeit
def reduce_dimensions(
    vectors: np.ndarray, projection_method: str, dimensions: int = 2, settings: Dict = None
) -> np.ndarray:
    if settings is None:
        settings = {}
    umap_min_dist = settings.get("umap_min_dist", 0.05)
    metric = settings.get("metric", "euclidean")
    try:
        if projection_method == ReductionMethod.PCA:
            decomp = sklearn.decomposition.PCA(n_components=dimensions)
            projections = decomp.fit_transform(vectors)
        elif projection_method == ReductionMethod.UMAP:
            decomp = umap.UMAP(n_components=dimensions, min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(vectors)
        elif projection_method == ReductionMethod.PCA_UMAP:
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(vectors)
            decomp = umap.UMAP(n_components=dimensions, min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(projections)
        elif projection_method == ReductionMethod.TSNE:
            decomp = sklearn.manifold.TSNE(
                n_components=dimensions, perplexity=min(30, len(vectors) - 1), metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(vectors)
        elif projection_method == ReductionMethod.PCA_TSNE:
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(vectors)
            decomp = sklearn.manifold.TSNE(
                n_components=dimensions, perplexity=min(30, len(vectors) - 1), metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(projections)
        else:
            raise ValueError(f"Unexpected reduction method: {projection_method}")
    except Exception as e:
        logger.error("Error reducing vectors dimensions: %s", str(e), exc_info=True)
        raise
    return projections


def create_clusters(vectors: np.ndarray, method: str = ClusteringMethod.DBSCAN, settings: Dict = None) -> List[int]:
    """
    returns a list of cluster labels for each embedding. -1 means no cluster
    """
    if settings is None:
        settings = {}
    dbscan_eps = settings.get("dbscan_eps", 0.5)
    dbscan_min_samples = settings.get("dbscan_min_samples", 5)
    metric = settings.get("metric", "euclidean")
    try:
        if method == ClusteringMethod.DBSCAN:
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=metric)
            labels = clusterer.fit_predict(vectors)
        else:
            raise ValueError(f"Unexpected clustering method: {method}")

    except Exception as e:
        logger.error("Error creating clusters: %s", str(e), exc_info=True)
        raise

    return labels.tolist()


def create_samples(
    labels: List[int], vectors: np.ndarray, sample_size: int, method: str = "random", settings: Dict = None
) -> List[int]:
    if settings is None:
        settings = {}
    ignore_noise = settings.get("ignore_noise", False)
    label_to_indexes = {}
    for i, label in enumerate(labels):
        if ignore_noise and label == -1:
            continue
        label_to_indexes.setdefault(label, []).append(i)
    if method == SamplingMethod.CENTROIDS:
        for label in label_to_indexes.keys():
            indexes = label_to_indexes[label]
            cluster_vectors = [vectors[i] for i in indexes]
            centroid = np.mean(cluster_vectors, axis=0)
            distances = [np.linalg.norm(vector - centroid) for vector in cluster_vectors]
            sorted_items = sorted(list(zip(distances, indexes)), key=lambda i: i[0], reverse=True)
            label_to_indexes[label] = [p[1] for p in sorted_items]

    results = {}
    added = 0
    total_items = len(labels)
    while True:
        for label in label_to_indexes.keys():
            if added >= sample_size or added >= total_items:
                return results
            indexes: List = label_to_indexes[label]
            if len(indexes) == 0:
                continue
            if method == SamplingMethod.RANDOM:
                i = indexes.pop(random.choice(len(indexes)))
            elif method == SamplingMethod.CENTROIDS:
                i = indexes.pop()
            else:
                raise ValueError(f"Unexpected sampling method: {method}")
            results.setdefault(label, []).append(i)
    return results


@server.post("/projections")
async def projections_endpoint(request: Request):
    state = request.state.state
    vectors = state["vectors"]
    method = state.get("method", ReductionMethod.UMAP)
    dimensions = state.get("dimesions", 2)
    settings = state.get("settings", {})
    vectors = np.array(vectors)
    projections = reduce_dimensions(vectors, method, dimensions, settings)
    return projections.tolist()


@server.post("/clusters")
async def clusters_endpoint(request: Request):
    state = request.state.state
    method = state.get("method", ClusteringMethod.DBSCAN)
    reduce = state.get("reduce", False)
    settings = state.get("settings", {})
    reduction_method = settings.get("reduction_method", ReductionMethod.UMAP)
    reduction_dimensions = settings.get("reduction_dimensions", 20)
    vectors = state["vectors"]
    vectors = np.array(vectors)
    if reduce:
        vectors = reduce_dimensions(vectors, reduction_method, reduction_dimensions, settings)
    return create_clusters(vectors, method, settings)


@server.post("/diverse")
async def diverse_endpoint(request: Request):
    state = request.state.state
    method = state.get("method", SamplingMethod.RANDOM)
    sample_size = state.get("sample_size")
    settings = state.get("settings", {})
    reduce = settings.get("reduce", False)
    reduction_method = settings.get("reduction_method", ReductionMethod.UMAP)
    reduction_dimensions = settings.get("reduction_dimensions", 20)
    clustering_method = settings.get("clustring_methood", ClusteringMethod.DBSCAN)
    labels = state.get("labels")
    vectors = state["vectors"]
    vectors = np.array(vectors)
    if labels:
        labels = np.array(labels)
    else:
        if reduce:
            vectors = reduce_dimensions(vectors, reduction_method, reduction_dimensions, settings)
        labels = create_clusters(vectors, clustering_method, settings)
    samples = create_samples(labels, vectors, sample_size, method, settings)
    return samples

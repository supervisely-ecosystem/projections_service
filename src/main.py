import os
import random
from time import perf_counter
from typing import Callable, Dict, List, Set
from uuid import uuid4

import numpy as np
import sklearn.decomposition
import umap
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from fastapi import Request
from sklearn.cluster import DBSCAN, MiniBatchKMeans

import supervisely as sly
from src.utils import timeit, track_in_progress_tasks
from supervisely import Application, logger

if sly.is_development():
    # for debug
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    task_id = sly.app.development.enable_advanced_debug(team_id=1)
    if task_id:
        logger.info(f"Debug Task ID: {task_id}")
app = Application()
server = app.get_server()

# global flags to prevent stop of the app while processing requests
IN_PROGRESS_TASKS = set()  # type: Set[str]
STOP_SERVICE_INTERVAL = int(os.getenv("STOP_SERVICE_INTERVAL", 60 * 2))  # 2 minutes by default
CHECK_TASKS_INTERVAL = int(os.getenv("CHECK_TASKS_INTERVAL", 30))  # 30 seconds by default
LAST_NO_TASKS_CHECK = perf_counter()  # last time when no tasks were found


class ReductionMethod:
    PCA = "pca"
    UMAP = "umap"
    PCA_UMAP = "pca-umap"
    TSNE = "t-sne"
    PCA_TSNE = "pca-t-sne"


class ClusteringMethod:
    DBSCAN = "dbscan"
    KMEANS = "kmeans"


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
            logger.debug(
                "Reducing dimensions with PCA",
                extra={"n_components": dimensions, "vectors_shape": vectors.shape},
            )
            decomp = sklearn.decomposition.PCA(n_components=dimensions)
            projections = decomp.fit_transform(vectors)
        elif projection_method == ReductionMethod.UMAP:
            logger.debug(
                "Reducing dimensions with UMAP",
                extra={
                    "n_components": dimensions,
                    "min_dist": umap_min_dist,
                    "metric": metric,
                    "vectors_shape": vectors.shape,
                },
            )
            decomp = umap.UMAP(n_components=dimensions, min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(vectors)
        elif projection_method == ReductionMethod.PCA_UMAP:
            logger.debug(
                "Reducing dimensions with PCA and UMAP",
                extra={
                    "n_components": dimensions,
                    "min_dist": umap_min_dist,
                    "metric": metric,
                    "vectors_shape": vectors.shape,
                },
            )
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(vectors)
            decomp = umap.UMAP(n_components=dimensions, min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(projections)
        elif projection_method == ReductionMethod.TSNE:
            perplexity = min(30, len(vectors) - 1)
            logger.debug(
                "Reducing dimensions with t-SNE",
                extra={
                    "n_components": dimensions,
                    "metric": metric,
                    "perplexity": perplexity,
                    "vectors_shape": vectors.shape,
                },
            )
            decomp = sklearn.manifold.TSNE(
                n_components=dimensions, perplexity=perplexity, metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(vectors)
        elif projection_method == ReductionMethod.PCA_TSNE:
            perplexity = min(30, len(vectors) - 1)
            logger.debug(
                "Reducing dimensions with PCA and t-SNE",
                extra={
                    "n_components": dimensions,
                    "metric": metric,
                    "perplexity": perplexity,
                    "vectors_shape": vectors.shape,
                },
            )
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(vectors)
            decomp = sklearn.manifold.TSNE(
                n_components=dimensions, perplexity=perplexity, metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(projections)
        else:
            raise ValueError(f"Unexpected reduction method: {projection_method}")
    except Exception as e:
        logger.error("Error reducing vectors dimensions: %s", str(e), exc_info=True)
        raise
    return projections


@timeit
def create_clusters(
    vectors: np.ndarray, method: str = ClusteringMethod.KMEANS, settings: Dict = None
) -> List[int]:
    """
    Returns a list of cluster labels for each embedding. -1 means no cluster
    """
    if settings is None:
        settings = {}
    dbscan_eps = settings.get("dbscan_eps", 0.5)
    dbscan_min_samples = settings.get("dbscan_min_samples", 5)
    metric = settings.get("metric", "euclidean")
    num_clusters = settings.get("num_clusters", 8)  # for KMeans
    try:
        if method == ClusteringMethod.DBSCAN:
            logger.debug(
                "Creating clusters with DBSCAN",
                extra={
                    "eps": dbscan_eps,
                    "min_samples": dbscan_min_samples,
                    "metric": metric,
                    "vectors_shape": vectors.shape,
                },
            )
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=metric)
            labels = clusterer.fit_predict(vectors)
        elif method == ClusteringMethod.KMEANS:
            if len(vectors) < num_clusters:
                num_clusters = len(vectors)
            kmeans = MiniBatchKMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(vectors)
        else:
            raise ValueError(f"Unexpected clustering method: {method}")

    except Exception as e:
        logger.error("Error creating clusters: %s", str(e), exc_info=True)
        raise

    return labels.tolist()


@timeit
def create_samples(
    labels: List[int],
    vectors: np.ndarray,
    sample_size: int,
    method: str = "random",
    settings: Dict = None,
) -> List[int]:
    if len(labels) != len(vectors):
        raise ValueError("Labels and vectors must have the same length")
    if len(labels) == 0:
        raise ValueError("Labels and vectors must not be empty")
    if settings is None:
        settings = {}
    logger.debug(
        "Creating diverse samples",
        extra={"sample_size": sample_size, "method": method, "vectors_shape": vectors.shape},
    )
    ignore_noise = settings.get("ignore_noise", False)
    label_to_indexes = {}
    for i, label in enumerate(labels):
        if ignore_noise and label == -1:
            continue
        label_to_indexes.setdefault(label, []).append(i)
    if method == SamplingMethod.CENTROIDS:
        for label in label_to_indexes.keys():
            indexes = label_to_indexes[label]
            if not indexes:
                continue
            cluster_vectors = np.array([vectors[i] for i in indexes])
            centroid = np.mean(cluster_vectors, axis=0)
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            sorted_idx = np.argsort(-distances)
            label_to_indexes[label] = [indexes[i] for i in sorted_idx]

    results = {}
    added = 0
    total_items = len(labels)
    used_indexes = set()
    while True:
        added_this_round = False
        for label in label_to_indexes.keys():
            if added >= sample_size or added >= total_items:
                return results
            indexes: List = label_to_indexes[label]
            indexes = [idx for idx in indexes if idx not in used_indexes]
            if len(indexes) == 0:
                continue
            if method == SamplingMethod.RANDOM:
                i = indexes.pop(random.randint(0, len(indexes) - 1))
            elif method == SamplingMethod.CENTROIDS:
                i = indexes.pop()
            else:
                raise ValueError(f"Unexpected sampling method: {method}")
            results.setdefault(label, []).append(i)
            used_indexes.add(i)
            added += 1
            added_this_round = True
        if added_this_round is False:
            break  # no new samples were added in this round, break to avoid infinite loop
    return results


@server.post("/projections")
@track_in_progress_tasks(IN_PROGRESS_TASKS)
@timeit
async def projections_endpoint(request: Request):
    state = request.state.state
    vectors = state["vectors"]
    method = state.get("method", ReductionMethod.UMAP)
    dimensions = state.get("dimesions", 2)
    settings = state.get("settings", {})
    vectors = np.array(vectors)
    logger.info("Reducing dimensions with method %s", method)
    projections = reduce_dimensions(vectors, method, dimensions, settings)
    return projections.tolist()


@server.post("/clusters")
@track_in_progress_tasks(IN_PROGRESS_TASKS)
@timeit
async def clusters_endpoint(request: Request):
    state = request.state.state
    method = state.get("method", ClusteringMethod.DBSCAN)
    reduce = state.get("reduce", False)
    settings = state.get("settings", {})
    reduction_method = settings.get("reduction_method", ReductionMethod.UMAP)
    reduction_dimensions = settings.get("reduction_dimensions", 20)
    vectors = state["vectors"]
    vectors = np.array(vectors)
    extra = {
        "settings": settings,
        "method": method,
        "vector_count": len(vectors),
    }
    logger.info("Creating clusters with method %s", method, extra=extra)
    if reduce:
        vectors = reduce_dimensions(vectors, reduction_method, reduction_dimensions, settings)
    return create_clusters(vectors, method, settings)


@server.post("/diverse")
@track_in_progress_tasks(IN_PROGRESS_TASKS)
@timeit
async def diverse_endpoint(request: Request):
    state = request.state.state
    method = state.get("sampling_method", SamplingMethod.RANDOM)
    sample_size = state.get("sample_size")
    settings = state.get("settings", {})
    settings["num_clusters"] = sample_size
    reduce = settings.get("reduce", False)
    reduction_method = settings.get("reduction_method", ReductionMethod.UMAP)
    default_dimensions = 3 if reduction_method == ReductionMethod.UMAP else 20
    reduction_dimensions = settings.get("reduction_dimensions", default_dimensions)  #! UMAP=3
    clustering_method = settings.get("clustering_method", ClusteringMethod.KMEANS)
    labels = state.get("labels")
    vectors = state["vectors"]
    vectors = np.array(vectors)
    extra = {
        "settings": settings,
        "method": method,
        "sample_size": sample_size,
        "vector_count": len(vectors),
    }
    logger.info(
        "Diverse sampling with method %s: Sample size: %s",
        method,
        sample_size,
        extra=extra,
    )
    if labels:
        labels = np.array(labels)
    else:
        if reduce:
            vectors = reduce_dimensions(vectors, reduction_method, reduction_dimensions, settings)
        labels = create_clusters(vectors, clustering_method, settings)
    samples = create_samples(labels, vectors, sample_size, method, settings)
    return samples


def stop_if_no_tasks_for_a_while():
    """
    Check if there are no in-progress tasks for a while, and stop the service if so.
    This function is scheduled to run periodically.
    """
    global LAST_NO_TASKS_CHECK
    global IN_PROGRESS_TASKS

    if len(IN_PROGRESS_TASKS) == 0:
        current_time = perf_counter()
        if current_time - LAST_NO_TASKS_CHECK > STOP_SERVICE_INTERVAL:
            sly.logger.info("No in-progress tasks for a while, stopping the service...")
            app.stop()
        else:
            sly.logger.info(
                "No in-progress tasks, but not enough time has passed since the last check. Waiting for more tasks..."
            )
    else:
        LAST_NO_TASKS_CHECK = perf_counter()
        sly.logger.info(f"Found {len(IN_PROGRESS_TASKS)} in-progress tasks, resetting the timer.")


try:
    scheduler = AsyncIOScheduler(
        job_defaults={
            "misfire_grace_time": 30,  # Allow jobs to run up to 30 seconds late if they miss their scheduled time
        }
    )

    scheduler.add_job(
        stop_if_no_tasks_for_a_while,
        max_instances=1,
        trigger="interval",
        seconds=CHECK_TASKS_INTERVAL,
    )

    @server.on_event("startup")
    def on_startup():
        sly.logger.info("Starting scheduler...")
        scheduler.start()
        sly.logger.info("Scheduler started successfully")

    app.call_before_shutdown(scheduler.shutdown)

except Exception as e:
    sly.logger.error(f"Error during initialization of the scheduler: {str(e)}")
    raise

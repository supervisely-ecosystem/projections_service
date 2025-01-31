from typing import List

import numpy as np
import sklearn.decomposition
from supervisely import Application, logger
import umap
from fastapi import Request

from src.utils import timeit


app = Application()
server = app.get_server()


class ProjectionMethod:
    PCA = "PCA"
    UMAP = "UMAP"
    PCA_UMAP = "PCA-UMAP"
    TSNE = "t-SNE"
    PCA_TSNE = "PCA-t-SNE"


@timeit
def calculate_projections(
    embeddings, projection_method, dimensions=2, metric="euclidean", umap_min_dist=0.05
) -> List[List[float]]:
    try:
        if projection_method == ProjectionMethod.PCA:
            decomp = sklearn.decomposition.PCA(n_components=dimensions)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == ProjectionMethod.UMAP:
            decomp = umap.UMAP(n_components=dimensions, min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == ProjectionMethod.PCA_UMAP:
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(embeddings)
            decomp = umap.UMAP(n_components=dimensions, min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(projections)
        elif projection_method == ProjectionMethod.TSNE:
            decomp = sklearn.manifold.TSNE(
                n_components=dimensions, perplexity=min(30, len(embeddings) - 1), metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(embeddings)
        elif projection_method == ProjectionMethod.PCA_TSNE:
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(embeddings)
            decomp = sklearn.manifold.TSNE(
                n_components=dimensions, perplexity=min(30, len(embeddings) - 1), metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(projections)
        else:
            raise ValueError(f"unexpexted projection_method {projection_method}")
    except Exception as e:
        logger.error(f"Error creating projections: {str(e)}", exc_info=True)
        raise
    return projections


@server.post("/create_projections")
async def create_projections_endpoint(request: Request):
    state = request.state.state
    vectors = state["vectors"]
    method = state.get("method", ProjectionMethod.UMAP)
    dimensions = state.get("dimesions", 2)
    vectors = np.array(vectors)
    projections = calculate_projections(vectors, method, dimensions)
    return projections.tolist()

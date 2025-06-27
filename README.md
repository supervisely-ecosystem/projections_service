<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/projections_service/releases/download/v0.1.0/poster.jpg">

# Projections Service

**High-performance microservice for dimensionality reduction, clustering, and diverse sampling of high-dimensional vector embeddings**

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Features">Features</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/projections-service)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/projections-service)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/projections-service.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/projections-service.png)](https://supervisely.com)

</div>

## Overview

🧩 This application is a core component of the **AI Search** feature in Supervisely and works in conjunction with the **Embeddings Generator** and **CLIP as Service** applications.

**Projections Service** is a high-performance headless microservice that provides advanced dimensionality reduction, clustering, and diverse sampling capabilities for high-dimensional vector embeddings. The service enables efficient visualization and analysis of complex embedding spaces through state-of-the-art machine learning algorithms.

The service operates as a background microservice and integrates seamlessly with the Supervisely ecosystem, providing RESTful API endpoints for creating 2D/3D projections, identifying clusters in embedding space, and selecting diverse representative samples from large datasets.

> **Note**: This application is designed to be used in conjunction with the **Embeddings Generator** application, so you don't need to run it separately. It will be automatically started when you access the **AI Search** feature in Supervisely.

### Key Capabilities

- **Dimensionality Reduction**: Transform high-dimensional embeddings into 2D/3D space for visualization and analysis
- **Clustering Analysis**: Identify natural groupings and patterns in embedding spaces
- **Diverse Sampling**: Select representative subsets from large datasets using intelligent sampling strategies
- **Multiple Algorithms**: Support for PCA, UMAP, t-SNE, DBSCAN, K-means, and hybrid approaches


---

For technical support and questions, please join our [Supervisely Ecosystem Slack community](https://supervisely.com/slack).

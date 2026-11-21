#!/usr/bin/env python3
"""Recluster saved contextual embeddings without reloading transformer models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


MODELS = ("bert", "bamibert")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_model(base_dir: Path, model: str) -> tuple[np.ndarray, list[dict[str, str]], np.ndarray]:
    archive = np.load(base_dir / f"{model}_toi_embeddings.npz")
    vectors = np.asarray(archive["embeddings"], dtype=np.float32)
    original_labels = np.asarray(archive["cluster_labels"], dtype=np.int32)
    rows = load_rows(base_dir / f"{model}_toi_occurrences.csv")
    if len(vectors) != len(rows) or len(original_labels) != len(rows):
        raise ValueError(
            f"{model}: saved embeddings, labels, and occurrence rows do not align "
            f"({len(vectors)}, {len(original_labels)}, {len(rows)})."
        )
    return vectors, rows, original_labels


def cluster_metrics(vectors: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    unique, counts = np.unique(labels, return_counts=True)
    result: dict[str, object] = {
        "cluster_sizes": {str(int(k)): int(v) for k, v in zip(unique, counts)},
    }
    if len(unique) > 1 and len(unique) < len(labels):
        result["silhouette_euclidean"] = float(silhouette_score(vectors, labels))
        result["calinski_harabasz"] = float(calinski_harabasz_score(vectors, labels))
        result["davies_bouldin"] = float(davies_bouldin_score(vectors, labels))
    return result


def write_outputs(
    output_dir: Path,
    model: str,
    vectors: np.ndarray,
    coords: np.ndarray,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    run_name: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    centers = np.stack([vectors[labels == label].mean(axis=0) for label in np.unique(labels)])
    label_values = np.unique(labels)
    label_to_center = {int(label): i for i, label in enumerate(label_values)}
    distances = np.array([
        np.linalg.norm(vector - centers[label_to_center[int(label)]])
        for vector, label in zip(vectors, labels)
    ])

    occurrence_path = output_dir / f"{model}_toi_occurrences.csv"
    with occurrence_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["cluster", "distance_to_center", "source", "line_number", "context"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, label, distance in zip(rows, labels, distances):
            writer.writerow({
                "cluster": int(label),
                "distance_to_center": f"{distance:.6f}",
                "source": row["source"],
                "line_number": row["line_number"],
                "context": row["context"],
            })

    examples_path = output_dir / f"{model}_cluster_examples.csv"
    with examples_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["cluster", "rank", "distance_to_center", "source", "line_number", "context"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in label_values:
            members = np.flatnonzero(labels == label)
            closest = members[np.argsort(distances[members])[:5]]
            for rank, index in enumerate(closest, start=1):
                row = rows[index]
                writer.writerow({
                    "cluster": int(label),
                    "rank": rank,
                    "distance_to_center": f"{distances[index]:.6f}",
                    "source": row["source"],
                    "line_number": row["line_number"],
                    "context": row["context"],
                })

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=35, alpha=0.8)
    ax.set_title(f"Contextual embeddings for ‘tôi’ — {model} ({run_name})")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model}_toi_clusters.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output_dir / f"{model}_toi_embeddings.npz",
        embeddings=vectors,
        cluster_labels=labels,
        centers=centers,
    )
    return cluster_metrics(vectors, labels)


def coassociation_consensus(runs: list[np.ndarray], n_clusters: int) -> np.ndarray:
    """Cluster pairwise same-cluster votes; label IDs need not align across runs."""
    n = len(runs[0])
    coassociation = np.zeros((n, n), dtype=np.float32)
    for labels in runs:
        coassociation += labels[:, None] == labels[None, :]
    coassociation /= len(runs)
    distances = 1.0 - coassociation
    np.fill_diagonal(distances, 0.0)
    consensus = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distances)
    return consensus.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("toi_embedding_results"))
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 42])
    args = parser.parse_args()
    if args.clusters < 2 or not args.seeds:
        parser.error("--clusters must be at least 2, and provide one or more --seeds")

    models = {}
    for model in MODELS:
        vectors, rows, initial_labels = load_model(args.base_dir, model)
        coords = PCA(n_components=2, random_state=17).fit_transform(vectors)
        models[model] = (vectors, rows, initial_labels, coords)

    seed_labels: dict[str, dict[int, np.ndarray]] = {model: {} for model in MODELS}
    metrics: dict[str, dict[str, object]] = {}
    for model, (vectors, rows, initial_labels, coords) in models.items():
        seed_labels[model][17] = initial_labels
        metrics[f"{model}/seed_17"] = cluster_metrics(vectors, initial_labels)

    for seed in args.seeds:
        run_dir = args.base_dir / f"seed_{seed}"
        for model, (vectors, rows, _, coords) in models.items():
            labels = KMeans(n_clusters=args.clusters, random_state=seed, n_init=10).fit_predict(vectors)
            seed_labels[model][seed] = labels
            metrics[f"{model}/seed_{seed}"] = write_outputs(
                run_dir, model, vectors, coords, rows, labels, f"seed {seed}"
            )

    for model in MODELS:
        labels_by_seed = seed_labels[model]
        metrics[f"{model}/seed_agreement_ari"] = {
            f"seed_{left}_vs_{right}": float(
                adjusted_rand_score(labels_by_seed[left], labels_by_seed[right])
            )
            for left, right in ((17, 23), (17, 42), (23, 42))
        }

    consensus_dir = args.base_dir / "consensus_17_23_42"
    for model, (vectors, rows, _, coords) in models.items():
        consensus = coassociation_consensus(
            [seed_labels[model][seed] for seed in (17, 23, 42)], args.clusters
        )
        metrics[f"{model}/consensus_17_23_42"] = write_outputs(
            consensus_dir, model, vectors, coords, rows, consensus,
            "consensus of seeds 17, 23, and 42",
        )

    with (args.base_dir / "clustering_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved seed runs and consensus outputs under {args.base_dir}")


if __name__ == "__main__":
    main()

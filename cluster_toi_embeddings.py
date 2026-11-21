#!/usr/bin/env python3
"""Embed occurrences of Vietnamese “tôi”, cluster them, and save examples/plots.

Each vector is the contextual hidden state for one occurrence of “tôi” in its
surrounding line. The two encoders are clustered independently. Transformer
inference uses Apple MPS when available; K-means and PCA run on the CPU.
BamiBERT handles raw Vietnamese text directly, so no external segmenter is needed.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODELS = {
    "bert": "google-bert/bert-base-multilingual-cased",
    "bamibert": "Qualcomm-AI-Research/BamiBERT",
}
WORD_RE = re.compile(r"(?<!\w)tôi(?!\w)", re.IGNORECASE)


@dataclass
class Occurrence:
    source: str
    line_number: int
    context: str
    model_text: str = ""
    target_start: int = 0
    target_end: int = 0


def read_occurrences(paths: list[Path], radius: int) -> list[Occurrence]:
    records: list[Occurrence] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                matches = list(WORD_RE.finditer(text))
                for match in matches:
                    left = max(0, match.start() - radius)
                    right = min(len(text), match.end() + radius)
                    context = text[left:right]
                    records.append(
                        Occurrence(
                            source=str(path),
                            line_number=line_number,
                            context=context,
                            model_text=context,
                            target_start=match.start() - left,
                            target_end=match.end() - left,
                        )
                    )
    if not records:
        raise ValueError("No standalone occurrences of 'tôi' were found.")
    return records


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested but is unavailable. Use an Apple Silicon Mac "
                "with an MPS-enabled PyTorch build and supported macOS version."
            )
        return torch.device("mps")
    return torch.device("mps" if mps_available else "cpu")


def embed_target_contexts(tokenizer, model, records: list[Occurrence], device: torch.device,
                          batch_size: int, max_length: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        enc = tokenizer(
            [record.model_text for record in batch_records],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").cpu().numpy()
        target_masks = []
        for record, row_offsets, attention in zip(
            batch_records, offsets, enc["attention_mask"].numpy()
        ):
            target_mask = np.zeros(len(row_offsets), dtype=bool)
            for i, (left, right) in enumerate(row_offsets):
                if (right > left and left < record.target_end
                        and right > record.target_start):
                    target_mask[i] = True
            target_mask &= attention.astype(bool)
            target_masks.append(target_mask)

        model_inputs = {key: value.to(device) for key, value in enc.items()}
        with torch.inference_mode():
            hidden = model(**model_inputs).last_hidden_state.float().cpu().numpy()
        for row, mask in zip(hidden, target_masks):
            if not mask.any():
                raise ValueError("BERT tokenizer truncated or missed a target occurrence.")
            vectors.append(row[mask].mean(axis=0))
    return np.asarray(vectors, dtype=np.float32)


def save_model_outputs(name: str, vectors: np.ndarray, records: list[Occurrence],
                       output_dir: Path, clusters: int, examples_per_cluster: int,
                       seed: int) -> None:
    # L2 normalization makes K-means and center distances focus on direction.
    vectors = normalize(vectors, norm="l2").astype(np.float32)
    n_clusters = min(clusters, len(vectors))
    if n_clusters < 2:
        raise ValueError("At least two occurrences are needed for clustering.")
    fitted = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(vectors)
    labels = fitted.labels_
    distances = np.linalg.norm(vectors - fitted.cluster_centers_[labels], axis=1)

    csv_path = output_dir / f"{name}_toi_occurrences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cluster", "distance_to_center", "source", "line_number", "context"],
        )
        writer.writeheader()
        for record, label, distance in zip(records, labels, distances):
            writer.writerow({
                "cluster": int(label),
                "distance_to_center": f"{distance:.6f}",
                "source": record.source,
                "line_number": record.line_number,
                "context": record.context,
            })

    examples_path = output_dir / f"{name}_cluster_examples.csv"
    with examples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cluster", "rank", "distance_to_center", "source", "line_number", "context"],
        )
        writer.writeheader()
        for cluster_id in range(n_clusters):
            members = np.flatnonzero(labels == cluster_id)
            ordered = members[np.argsort(distances[members])][:examples_per_cluster]
            for rank, index in enumerate(ordered, start=1):
                record = records[index]
                writer.writerow({
                    "cluster": cluster_id,
                    "rank": rank,
                    "distance_to_center": f"{distances[index]:.6f}",
                    "source": record.source,
                    "line_number": record.line_number,
                    "context": record.context,
                })

    np.savez_compressed(
        output_dir / f"{name}_toi_embeddings.npz",
        embeddings=vectors,
        cluster_labels=labels,
        centers=fitted.cluster_centers_,
    )
    coords = PCA(n_components=2, random_state=seed).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=35, alpha=0.8)
    ax.set_title(f"Contextual embeddings for ‘tôi’ — {name} (PCA, {n_clusters} clusters)")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_toi_clusters.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path,
                        default=[Path("Historical.txt"), Path("ViSoBERT_sample.txt")])
    parser.add_argument("--output-dir", type=Path, default=Path("toi_embedding_results"))
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--examples-per-cluster", type=int, default=5)
    parser.add_argument("--context-radius", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    for path in args.inputs:
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    if args.clusters < 2 or args.examples_per_cluster < 1:
        parser.error("--clusters must be at least 2 and --examples-per-cluster at least 1")
    if args.max_length < 8 or args.context_radius < 1 or args.batch_size < 1:
        parser.error("max length, context radius, and batch size must be positive")

    records = read_occurrences(args.inputs, args.context_radius)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(records)} occurrences of standalone 'tôi'; inference device: {device}")

    for name, model_id in DEFAULT_MODELS.items():
        print(f"Loading {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModel.from_pretrained(model_id).eval().to(device)
        max_length = min(args.max_length, int(getattr(model.config, "max_position_embeddings", args.max_length)) - 2)
        for record in records:
            record.model_text = record.context
        vectors = embed_target_contexts(
            tokenizer, model, records, device, args.batch_size, max_length
        )

        save_model_outputs(
            name, vectors, records, args.output_dir,
            args.clusters, args.examples_per_cluster, args.seed,
        )
        del model, tokenizer
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"Saved embeddings, occurrence tables, cluster examples, and plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise

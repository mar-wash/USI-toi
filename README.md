# Contextual embeddings for “tôi”

`cluster_toi_embeddings.py` extracts one contextual vector for every standalone
occurrence of **tôi** in the input text files. It embeds the target token pieces
with the surrounding context, then clusters the vectors independently for
multilingual BERT and BamiBERT. It also writes a PCA plot and the nearest
examples to each cluster center.

The default inputs are `Historical.txt` and `ViSoBERT_sample.txt`. The default
device is MPS when PyTorch reports that Apple GPU support is available, and CPU
otherwise. An Apple Silicon Mac and an MPS-enabled PyTorch/macOS installation
are required for GPU inference.

## Install and run

From this folder, create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The script uses raw Vietnamese text for both models. BamiBERT does not need
PhoBERT's VnCoreNLP segmenter, so Java is not required. Then run:

```bash
python cluster_toi_embeddings.py --device mps
```

Use `--device auto` to select MPS when available and CPU otherwise.

Optional controls include:

```bash
python cluster_toi_embeddings.py \
  --inputs Historical.txt ViSoBERT_sample.txt \
  --clusters 4 --examples-per-cluster 5 --output-dir toi_embedding_results
```

## Outputs

For each model, the output directory contains:

- `*_toi_embeddings.npz`: normalized occurrence vectors, cluster labels, and centers.
- `*_toi_occurrences.csv`: each occurrence, its assigned cluster, source line, and distance to center.
- `*_cluster_examples.csv`: closest examples to each cluster center.
- `*_toi_clusters.png`: a 2D PCA view colored by cluster.

Clustering and PCA run on CPU; only transformer inference uses MPS. The models
are `google-bert/bert-base-multilingual-cased` and
`Qualcomm-AI-Research/BamiBERT`. Review their licenses before redistribution:
multilingual BERT is Apache-2.0, while BamiBERT is released under BSD 3-Clause
Clear plus Qualcomm's Responsible AI license and is intended for research and
educational use.

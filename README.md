# Contextual embeddings for “tôi”

The scripts in `scripts/` extract contextual embeddings for standalone
occurrences of **tôi**, cluster them separately for multilingual BERT and
BamiBERT, and save plots and examples nearest each cluster center. By default,
the embedding run samples text from the Hugging Face datasets
[`SEACrowd/visobert`](https://huggingface.co/datasets/SEACrowd/visobert) and
[`thaodd11/VieBookRead`](https://huggingface.co/datasets/thaodd11/VieBookRead).
Each context is drawn from up to 160 characters on either side of the target.

## Setup

From the repository root, activate your existing virtual environment and install
the requirements:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

VieBookRead requires gated access. Accept its terms on the dataset page, then
authenticate with Hugging Face in the environment where you run the script:

```bash
hf auth login
```

The ViSoBERT loader uses the dataset's SEACrowd loading script, so the
`seacrowd` dependency is included in `requirements.txt`. The VieBookRead reader
downloads only `final.zip`. It looks for text fields in
supported text, JSON, and table files inside that archive; if the archive uses
an unrecognized field name, provide it with `--viebookread-text-key`. You can
also download `final.zip` yourself and pass its path with
`--viebookread-archive /path/to/final.zip`.

## Run the embedding and clustering job

On an Apple Silicon Mac with MPS-enabled PyTorch:

```bash
python scripts/cluster_toi_embeddings.py --device mps
```

`--device auto` selects MPS when available and otherwise uses CPU. The defaults
sample up to 10,000 records and retain up to 10,000 occurrences of “tôi” from
each dataset. The record sample seed is 17. To use only one dataset, add either
`--datasets visobert` or `--datasets viebookread`. To use all available records,
set both limits to zero:

```bash
python scripts/cluster_toi_embeddings.py \
  --device mps --max-records-per-dataset 0 --max-occurrences-per-dataset 0
```

The full ViSoBERT split is very large, so the default bounded sample is intended
for an initial run. Both datasets are combined in each model's clustering by
default; the CSV files retain a `source` column to identify which dataset each
example came from.

## Recluster embeddings with additional seeds

After the embedding job has produced its output, this script uses the saved
vectors to run additional K-means seeds and a co-association consensus:

```bash
python scripts/cluster_saved_embeddings.py
```

By default it uses seeds 23 and 42 alongside the original seed 17, saves each
seed and the consensus in separate subfolders, and writes clustering metrics.
Use `--base-dir` if the embeddings are in a different output directory.

## Outputs

The default output directory is `toi_embedding_results/`. For each model it
contains normalized embeddings, occurrence and cluster-example CSV files, and
a 2D PCA plot. Clustering and PCA run on CPU; transformer inference uses the
selected device. Example CSVs include context excerpts from the source datasets.
VieBookRead's access and noncommercial research terms apply to the data and
derived excerpts; review its dataset card before sharing outputs.

"""Streaming and archive readers for the Vietnamese corpora used in this repo."""

from __future__ import annotations

import csv
import html
import io
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Iterable, Iterator


VISOBERT_REPO = "SEACrowd/visobert"
VIEBOOKREAD_REPO = "thaodd11/VieBookRead"
VIEBOOKREAD_ARCHIVE = "final.zip"
TEXT_FIELD_NAMES = {
    "text", "content", "body", "paragraph", "sentence", "page_text",
    "ocr_text", "raw_text", "reference_text", "ground_truth",
    "groundtruth", "pseudo_ground_truth", "corrected_text", "final_text",
}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".xml", ".html", ".htm", ".xhtml"}


def iter_visobert_records(max_records: int, seed: int) -> Iterator[tuple[str, int, str]]:
    """Read the public parquet mirror as a streaming HF dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package from requirements.txt.") from exc

    dataset = load_dataset(
        VISOBERT_REPO, split="train", streaming=True, trust_remote_code=True
    )
    if max_records > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=max(10_000, max_records)).take(max_records)
    for row_number, row in enumerate(dataset, start=1):
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield f"{VISOBERT_REPO}/train", row_number, text


def _normalized_field(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", field.casefold()).strip("_")


def _is_text_field(field: str, requested_key: str | None) -> bool:
    normalized = _normalized_field(field)
    if requested_key:
        return normalized == _normalized_field(requested_key)
    return normalized in TEXT_FIELD_NAMES or any(
        marker in normalized for marker in ("text", "content", "ground_truth", "ocr_text", "paragraph")
    )


def _json_text_values(value, requested_key: str | None, parent_key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_text_field(str(key), requested_key):
                if isinstance(child, str):
                    yield child
                else:
                    yield from _json_text_values(child, requested_key, str(key))
            elif isinstance(child, (dict, list)):
                yield from _json_text_values(child, requested_key, str(key))
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, str) and _is_text_field(parent_key, requested_key):
                yield child
            elif isinstance(child, (dict, list)):
                yield from _json_text_values(child, requested_key, parent_key)
    elif isinstance(value, str) and parent_key and _is_text_field(parent_key, requested_key):
        yield value


def _text_values_from_bytes(name: str, data: bytes, requested_key: str | None) -> Iterator[str]:
    suffix = Path(name).suffix.casefold()
    if suffix in {".txt", ".md"}:
        decoded = data.decode("utf-8-sig", errors="replace")
        # Preserve paragraph context while removing line-wrap artifacts.
        for paragraph in re.split(r"\n\s*\n", decoded):
            text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
            if text:
                yield text
        return

    if suffix in {".jsonl", ".ndjson"}:
        for line in data.decode("utf-8-sig", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield from _json_text_values(value, requested_key)
        return

    if suffix == ".json":
        try:
            value = json.loads(data.decode("utf-8-sig", errors="replace"))
        except json.JSONDecodeError:
            return
        yield from _json_text_values(value, requested_key)
        return

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        stream = io.StringIO(data.decode("utf-8-sig", errors="replace"))
        reader = csv.DictReader(stream, delimiter=delimiter)
        text_fields = [key for key in (reader.fieldnames or []) if _is_text_field(key, requested_key)]
        for row in reader:
            for field in text_fields:
                value = row.get(field)
                if value:
                    yield value
        return

    if suffix in {".xml", ".html", ".htm", ".xhtml"}:
        decoded = data.decode("utf-8-sig", errors="replace")
        decoded = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", decoded)
        decoded = re.sub(r"(?s)<[^>]+>", " ", decoded)
        text = re.sub(r"\s+", " ", html.unescape(decoded)).strip()
        if text:
            yield text


def _texts_in_zip(archive: zipfile.ZipFile, requested_key: str | None) -> Iterator[tuple[str, str]]:
    for member in archive.infolist():
        if member.is_dir():
            continue
        name = member.filename
        suffix = Path(name).suffix.casefold()
        try:
            payload = archive.read(member)
        except (OSError, RuntimeError, zipfile.BadZipFile):
            continue
        if suffix == ".zip" or suffix == ".epub":
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as nested:
                    yield from _texts_in_zip(nested, requested_key)
            except zipfile.BadZipFile:
                continue
        elif suffix in TEXT_SUFFIXES:
            for text_number, text in enumerate(
                _text_values_from_bytes(name, payload, requested_key), start=1
            ):
                if text.strip():
                    yield f"{name}#{text_number}", text


def _reservoir_sample(
    records: Iterable[tuple[str, str]], limit: int, seed: int
) -> Iterator[tuple[str, str]]:
    if limit <= 0:
        yield from records
        return
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        if index < limit:
            reservoir.append(record)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < limit:
                reservoir[replacement] = record
    yield from reservoir


def iter_viebookread_records(
    max_records: int,
    seed: int,
    local_archive: Path | None = None,
    text_key: str | None = None,
) -> Iterator[tuple[str, int, str]]:
    """Read the text-bearing final.zip archive from the gated dataset."""
    if local_archive:
        archive_path = local_archive.expanduser().resolve()
    else:
        try:
            from huggingface_hub import hf_hub_download
            archive_path = Path(hf_hub_download(
                repo_id=VIEBOOKREAD_REPO,
                filename=VIEBOOKREAD_ARCHIVE,
                repo_type="dataset",
                token=True,
            ))
        except Exception as exc:
            raise RuntimeError(
                "VieBookRead is gated. Accept its access terms on the dataset page, "
                "log in with `hf auth login`, then retry. Alternatively, provide "
                "the downloaded final.zip with --viebookread-archive."
            ) from exc

    try:
        with zipfile.ZipFile(archive_path) as archive:
            sampled = list(_reservoir_sample(_texts_in_zip(archive, text_key), max_records, seed))
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Could not read VieBookRead archive: {archive_path}") from exc

    if not sampled:
        suffix = f" using text key {text_key!r}" if text_key else " using recognized text fields"
        raise ValueError(
            "No text records were found in VieBookRead final.zip" + suffix + ". "
            "Inspect the archive fields and pass --viebookread-text-key with the text column."
        )
    for row_number, (member, text) in enumerate(sampled, start=1):
        yield f"{VIEBOOKREAD_REPO}/{VIEBOOKREAD_ARCHIVE}:{member}", row_number, text

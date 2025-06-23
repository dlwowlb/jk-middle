#!/usr/bin/env python3
"""Download audio from a metadata CSV and package it as WebDataset shards."""

import os
import tarfile
import json
import io
import csv
import urllib.request
from urllib.parse import urlparse


def load_rows(csv_path):
    """Read rows from a CSV file returning a list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("url")]


def download_audio(url):
    """Download audio data from a URL and infer the file extension."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    ext = os.path.splitext(urlparse(url).path)[1]
    if not ext:
        ext = ".wav"
    return data, ext


def add_sample(tar, idx, audio_bytes, ext, metadata):
    base = f"{idx:06d}"

    info = tarfile.TarInfo(f"{base}{ext}")
    info.size = len(audio_bytes)
    tar.addfile(info, io.BytesIO(audio_bytes))

    json_bytes = json.dumps(metadata).encode("utf-8")
    info = tarfile.TarInfo(f"{base}.json")
    info.size = len(json_bytes)
    tar.addfile(info, io.BytesIO(json_bytes))


def build_webdataset(csv_path, output_dir, shard_size=1000):
    rows = load_rows(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for shard_start in range(0, len(rows), shard_size):
        shard_rows = rows[shard_start:shard_start + shard_size]
        shard_index = shard_start // shard_size
        tar_path = os.path.join(output_dir, f"{shard_index:05d}.tar")

        with tarfile.open(tar_path, "w") as tar:
            for local_idx, row in enumerate(shard_rows):
                global_idx = shard_start + local_idx
                try:
                    audio_bytes, ext = download_audio(row["url"])
                except Exception as exc:
                    print(f"Failed to download {row['url']}: {exc}")
                    continue

                metadata = {
                    "text": row.get("title", ""),
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "license": row.get("license"),
                    "url": row.get("url"),
                }
                add_sample(tar, global_idx, audio_bytes, ext, metadata)
        print(f"Wrote {tar_path} with {len(shard_rows)} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create WebDataset shards from audio URLs listed in a CSV file"
    )
    parser.add_argument(
        "--csv_path", help="CSV file containing id,title,author,license,url columns"
    )
    parser.add_argument("--output_dir", help="Output directory for .tar shards")
    parser.add_argument(
        "--shard-size", type=int, default=1000, help="Number of samples per shard"
    )

    args = parser.parse_args()
    build_webdataset(args.csv_path, args.output_dir, args.shard_size)
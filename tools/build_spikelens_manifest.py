#!/usr/bin/env python3
"""
Build SpikeLens web manifest dynamically from exported trace folders.

Run from:
    ~/Downloads/spikelens_starter_package

Expected folders:
    spikelens_models/<model_id>/metadata.json
    spikelens_webapp/public/data/traces/<model_id>/*.json

Writes:
    spikelens_webapp/public/data/manifest.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Robust project-root detection.
# This script lives at:
#   <project_root>/tools/build_spikelens_manifest.py
# So project root is parent of the tools directory.
SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent.parent

MODELS_DIR = ROOT / "spikelens_models"
DATA_DIR = ROOT / "spikelens_webapp" / "public" / "data"
TRACES_DIR = DATA_DIR / "traces"
OUT_MANIFEST = DATA_DIR / "manifest.json"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def parse_model_id(model_id: str) -> dict:
    """
    Accepts names like:
      beta0.85_lat0.25_steps25
      beta0.9_lat0.3_steps30
      beta085_lat025_steps25
    """
    out = {}

    m = re.search(r"beta([0-9.]+)", model_id)
    if m:
        out["beta"] = safe_float(m.group(1))

    m = re.search(r"lat([0-9.]+)", model_id)
    if m:
        out["latency_threshold"] = safe_float(m.group(1))

    m = re.search(r"steps(\d+)", model_id)
    if m:
        out["num_steps"] = safe_int(m.group(1))

    return out


def parse_sample_filename(path: Path) -> dict:
    """
    Accepts names like:
      sample_0000_label_7_pred_7.json
      sample_0018_label_3_pred_2.json
    """
    name = path.name
    out = {}

    m = re.search(r"sample[_-]?(\d+)", name)
    if m:
        out["sample_idx"] = int(m.group(1))

    m = re.search(r"label[_-]?(\d+)", name)
    if m:
        out["true_label"] = int(m.group(1))

    m = re.search(r"pred[_-]?(\d+)", name)
    if m:
        out["pred_label"] = int(m.group(1))

    if "true_label" in out and "pred_label" in out:
        out["correct"] = out["true_label"] == out["pred_label"]

    return out


def normalize_summary_samples(model_id: str, summary: Any) -> list[dict]:
    """
    Supports summary shapes:
      {"samples": [...]}
      {"traces": [...]}
      [...]
    """
    if isinstance(summary, dict) and isinstance(summary.get("samples"), list):
        raw_samples = summary["samples"]
    elif isinstance(summary, dict) and isinstance(summary.get("traces"), list):
        raw_samples = summary["traces"]
    elif isinstance(summary, list):
        raw_samples = summary
    else:
        raw_samples = []

    samples = []
    for s in raw_samples:
        if not isinstance(s, dict):
            continue

        filename = s.get("file") or s.get("filename")
        path = s.get("path")

        if not path and filename:
            path = f"data/traces/{model_id}/{filename}"

        if not path:
            continue

        item = dict(s)
        item["path"] = path.replace("\\", "/")
        samples.append(item)

    return samples


def scan_samples_from_files(model_id: str, model_dir: Path) -> list[dict]:
    samples = []

    for f in sorted(model_dir.glob("*.json")):
        if f.name == "summary.json":
            continue

        sample = parse_sample_filename(f)

        # Try reading only metadata from the trace itself.
        obj = read_json(f, default={})
        if isinstance(obj, dict):
            sample["sample_idx"] = obj.get("sample_idx", sample.get("sample_idx"))
            sample["true_label"] = obj.get("true_label", obj.get("target", sample.get("true_label")))
            sample["pred_label"] = obj.get("pred_label", sample.get("pred_label"))
            sample["correct"] = obj.get("correct", sample.get("correct"))

        if "correct" not in sample and "true_label" in sample and "pred_label" in sample:
            sample["correct"] = sample["true_label"] == sample["pred_label"]

        sample.setdefault("sample_idx", len(samples))
        sample.setdefault("true_label", None)
        sample.setdefault("pred_label", None)
        sample.setdefault("correct", False)

        sample["filename"] = f.name
        sample["path"] = f"data/traces/{model_id}/{f.name}"

        samples.append(sample)

    return samples


def build_manifest() -> dict:
    if not TRACES_DIR.exists():
        raise FileNotFoundError(f"Missing traces directory: {TRACES_DIR}")

    variants = []

    for trace_model_dir in sorted([p for p in TRACES_DIR.iterdir() if p.is_dir()]):
        model_id = trace_model_dir.name

        if model_id == "demo":
            # Keep demo only if you want; by default skip it once real models exist.
            continue

        model_metadata = read_json(MODELS_DIR / model_id / "metadata.json", default={})
        summary = read_json(trace_model_dir / "summary.json", default=None)

        if summary is not None:
            samples = normalize_summary_samples(model_id, summary)
        else:
            samples = []

        if not samples:
            samples = scan_samples_from_files(model_id, trace_model_dir)

        parsed = parse_model_id(model_id)

        beta = model_metadata.get("beta", parsed.get("beta"))
        latency_threshold = model_metadata.get("latency_threshold", parsed.get("latency_threshold"))
        num_steps = (
            model_metadata.get("num_steps")
            or model_metadata.get("num_steps_inference")
            or model_metadata.get("num_steps_train")
            or parsed.get("num_steps")
        )

        variant = {
            "model_id": model_id,
            "beta": beta,
            "latency_threshold": latency_threshold,
            "num_steps": num_steps,
            "neuron_threshold": model_metadata.get("neuron_threshold", 1.0),
            "learning_rate": model_metadata.get("learning_rate", model_metadata.get("lr")),
            "summary_path": f"data/traces/{model_id}/summary.json",
            "sample_count": len(samples),
            "correct_count": sum(1 for s in samples if s.get("correct") is True),
            "wrong_count": sum(1 for s in samples if s.get("correct") is False),
            "samples": samples,
        }

        variants.append(variant)

    manifest = {
        "created_by": "tools/build_spikelens_manifest.py",
        "architecture": {
            "input": 784,
            "fc1": 1024,
            "fc2": 256,
            "out": 10,
            "grid_shapes": {
                "input": [28, 28],
                "fc1": [32, 32],
                "fc2": [16, 16],
                "out": [1, 10],
            },
        },
        "variants": variants,
        "total_variants": len(variants),
        "total_samples": sum(v["sample_count"] for v in variants),
    }

    return manifest


def main():
    print(f"[SpikeLens manifest] ROOT       = {ROOT}")
    print(f"[SpikeLens manifest] MODELS_DIR = {MODELS_DIR}")
    print(f"[SpikeLens manifest] TRACES_DIR = {TRACES_DIR}")
    print(f"[SpikeLens manifest] OUT        = {OUT_MANIFEST}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()

    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote: {OUT_MANIFEST}")
    print(f"Models: {manifest['total_variants']}")
    print(f"Total samples: {manifest['total_samples']}")

    if manifest["total_variants"] == 0:
        print("\nWARNING: manifest has 0 models.")
        print("This means traces were not exported to:")
        print(f"  {TRACES_DIR}")
        print("Run the export notebook/script first.")

    for v in manifest["variants"]:
        print(
            f"  {v['model_id']}: "
            f"{v['sample_count']} samples, "
            f"{v['correct_count']} correct, "
            f"{v['wrong_count']} wrong"
        )

if __name__ == "__main__":
    main()

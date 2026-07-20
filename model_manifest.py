"""Write an auditable manifest for every production run."""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from config import BOOTSTRAP_MODELS, EDGE_RULE, FOCUS, MODEL_VERSION


def sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(path="model_manifest.json"):
    files = ["fights_v2.csv", "raw/ufc-master.csv", "odds_upcoming.csv",
             "odds_log.csv", "method_model.pkl"]
    hashes = {name: sha256(name) for name in files if Path(name).exists()}
    rows = {}
    if Path("fights_v2.csv").exists():
        f = pd.read_csv("fights_v2.csv", usecols=["date"])
        rows["fights"] = int(len(f))
        rows["fight_date_max"] = str(pd.to_datetime(f["date"]).max().date())
    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": MODEL_VERSION,
        "focus": FOCUS,
        "edge_rule": EDGE_RULE,
        "bootstrap_models": BOOTSTRAP_MODELS,
        "python": sys.version,
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "inputs": hashes,
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    print(json.dumps(write_manifest(), indent=2))

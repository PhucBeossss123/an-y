"""Fetch official iSarcasmEval English Subtask A files and write checksums."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = "https://api.github.com/repos/iabufarha/iSarcasmEval"
FILES = {
    "train.En.csv": "train/train.En.csv",
    "task_A_En_test.csv": "test/task_A_En_test.csv",
    "README.md": "README.md",
    "LICENSE": "LICENSE",
}


def main():
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{ROOT}/commits/main", timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    records = {}
    for local_name, source_path in FILES.items():
        url = f"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/{revision}/{source_path}"
        download = requests.get(url, timeout=120)
        download.raise_for_status()
        content = download.content
        target = out / local_name
        target.write_bytes(content)
        records[local_name] = {
            "source_path": source_path,
            "url": url,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        print(f"Saved {target} ({len(content)} bytes)")
    manifest = {
        "repository": "https://github.com/iabufarha/iSarcasmEval",
        "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "MIT (see LICENSE)",
        "files": records,
        "task": "English Subtask A; text and binary sarcastic label only",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Source revision: {revision}")


if __name__ == "__main__":
    main()

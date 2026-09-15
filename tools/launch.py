"""从统一配置启动一个流水线阶段。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/pipeline.json"


def run_stage(stage: str, force: bool = False) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if stage not in config:
        raise ValueError(f"未知阶段: {stage}")
    (ROOT / "artifacts/cache").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts/reports").mkdir(parents=True, exist_ok=True)
    for command in config[stage]:
        skip_path = command.get("skip_if_exists")
        if skip_path and not force and (ROOT / skip_path).is_file():
            print(f"复用已有产物: {skip_path}", flush=True)
            continue
        label = command.get("name", command["script"])
        argv = [sys.executable, "-u", command["script"], *command["args"]]
        print(f"\n[{stage}] {label}", flush=True)
        print(" ".join(argv), flush=True)
        subprocess.run(argv, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--force", action="store_true", help="强制重建可复用缓存")
    args = parser.parse_args()
    run_stage(args.stage, force=args.force)


if __name__ == "__main__":
    main()


"""为宽同分子式候选池构建可复用的分子图嵌入缓存。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physchemrag.config import OUTPUT_REPORTS_DIR  # noqa: E402
from physchemrag.module4_cascade.learned_cascade import SpectrumGraphContrastiveModel  # noqa: E402
from physchemrag.shared.molecular_formula import canonical_smiles  # noqa: E402
from physchemrag.shared.chemical_domain import (  # noqa: E402
    DOMAIN_NAME,
    domain_metadata,
    domain_metadata_matches,
    inspect_smiles_domain,
)
from physchemrag.shared.graph import graph_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-pool-cache",
        type=Path,
        default=OUTPUT_REPORTS_DIR / "wide_formula_training_pools_domain_hac13.json",
    )
    parser.add_argument("--contrastive-weights", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_REPORTS_DIR / "wide_formula_graph_embeddings_domain_hac13.pt",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--disable-domain-filter",
        action="store_true",
        help="关闭闭域门控，仅用于复现旧版缓存",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_candidates < 0:
        parser.error("batch-size 必须大于 0，max-candidates 不能为负")
    return args


def domain_result(result: object) -> tuple[bool, list[str]]:
    """兼容共享域检查器返回字典或对象。"""
    if isinstance(result, bool):
        return result, [] if result else ["outside_domain"]
    getter = result.get if isinstance(result, dict) else lambda key, default=None: getattr(result, key, default)
    accepted = next(
        (bool(value) for key in ("valid", "in_domain", "accepted", "eligible") if (value := getter(key, None)) is not None),
        False,
    )
    raw = getter("reasons", getter("reason", getter("code", None)))
    reasons = [str(value) for value in raw] if isinstance(raw, (list, tuple, set)) else ([str(raw)] if raw else [])
    return accepted, reasons or ([] if accepted else ["outside_domain"])


def load_candidates(
    path: Path,
    max_candidates: int,
    smoke: bool,
    apply_domain_filter: bool,
) -> tuple[list[dict], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if apply_domain_filter and not domain_metadata_matches(payload.get("chemical_domain")):
        raise ValueError(f"宽候选池缺少一致的 {DOMAIN_NAME} 元数据: {path}")
    unique: dict[str, str] = {}
    count = 0
    audit: Counter[str] = Counter()
    for formula, pool in dict(payload.get("formula_pools", {})).items():
        for item in list(pool):
            smiles = str(item.get("smiles") or item.get("canonical") or "")
            audit["input_candidates"] += 1
            if apply_domain_filter:
                accepted, reasons = domain_result(
                    inspect_smiles_domain(
                        smiles,
                        formula=formula,
                        require_single_component=True,
                        require_neutral_closed_shell=True,
                    )
                )
                if not accepted:
                    for reason in reasons:
                        audit[f"candidate_structure:{reason}"] += 1
                    continue
            key = canonical_smiles(smiles, isomeric=False) or ""
            if key:
                unique.setdefault(key, smiles)
                count += 1
            else:
                audit["invalid_canonical_smiles"] += 1
            if smoke and len(unique) >= 1024:
                break
        if smoke and len(unique) >= 1024:
            break
    items = [{"canonical": key, "smiles": smiles} for key, smiles in unique.items()]
    if max_candidates > 0:
        items = items[:max_candidates]
    if len(items) < 2:
        raise RuntimeError("宽候选池可编码结构不足")
    audit["retained_unique_candidates"] = len(items)
    print(f"宽池候选 unique={len(items)}，原始候选记录={count}", flush=True)
    return items, dict(audit)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    apply_domain_filter = not args.disable_domain_filter
    items, domain_audit = load_candidates(
        args.wide_pool_cache,
        args.max_candidates,
        args.smoke,
        apply_domain_filter,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.contrastive_weights, map_location=device, weights_only=False)
    if apply_domain_filter and not domain_metadata_matches(checkpoint.get("chemical_domain")):
        raise ValueError(
            f"双塔权重缺少一致的 {DOMAIN_NAME} 元数据: {args.contrastive_weights}"
        )
    model = SpectrumGraphContrastiveModel(
        dare_weights=str(checkpoint.get("dare_weights")),
        node_in_dim=int(checkpoint.get("node_in_dim", 9)),
        edge_in_dim=int(checkpoint.get("edge_in_dim", 3)),
        embed_dim=int(checkpoint.get("embed_dim", 512)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    vectors = []
    kept_items = []
    failed = []
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        graph_list = []
        valid_items = []
        for item in chunk:
            data = graph_data(item["smiles"])
            if data is None:
                failed.append(item["canonical"])
                continue
            graph_list.append(data)
            valid_items.append(item)
        if graph_list:
            batch = Batch.from_data_list(graph_list).to(device, non_blocking=True)
            with torch.inference_mode():
                output = model.encode_graph(batch).cpu().float()
            vectors.append(output)
            kept_items.extend(valid_items)
        processed = min(start + args.batch_size, len(items))
        if processed == args.batch_size or processed % (args.batch_size * 20) == 0 or processed == len(items):
            print(f"分子图嵌入 {processed}/{len(items)}", flush=True)
    matrix = torch.cat(vectors, dim=0) if vectors else torch.empty((0, 512), dtype=torch.float32)
    index = {item["canonical"]: position for position, item in enumerate(kept_items)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "method": "wide_formula_graph_embedding_cache",
            "chemical_domain": domain_metadata() if apply_domain_filter else None,
            "domain_filter_enabled": apply_domain_filter,
            "domain_audit": domain_audit,
            "wide_pool_cache": str(args.wide_pool_cache.resolve()),
            "embeddings": matrix,
            "index": index,
            "failed": failed,
            "contrastive_weights": str(args.contrastive_weights.resolve()),
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "rows": list(matrix.shape),
                "failed": len(failed),
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""在完整同分子式候选池上训练分子式条件化双塔。

该模型把分子式元素计数作为条件输入，直接以真实结构的显式索引为
Listwise 正样本。候选数组会在每一步随机置换，不读取 PubChem 顺序、CID
数值或旧 teacher 分数。训练完成后可用于对完整同分子式池进行第二路粗召回。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physchemrag.config import DATA_DIR, OUTPUT_REPORTS_DIR, WEIGHTS_DIR  # noqa: E402
from physchemrag.module2_crossmodal.formula_conditioned_retriever import (  # noqa: E402
    FORMULA_DIM,
    FormulaConditionedDualAdapter,
    formula_vector,
)
from physchemrag.module4_cascade.learned_cascade import SpectrumGraphContrastiveModel  # noqa: E402
from physchemrag.shared.chemical_domain import (  # noqa: E402
    DOMAIN_NAME,
    FORMULA_ELEMENTS as DOMAIN_FORMULA_ELEMENTS,
    domain_metadata,
    domain_metadata_matches,
    inspect_formula_domain,
    inspect_smiles_domain,
)
from listwise import (  # noqa: E402
    EMBEDDING_DIM,
    encode_query,
    load_embedding_cache,
    load_records,
    load_wide_groups,
    split_by_formula,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-pool-cache",
        type=Path,
        default=OUTPUT_REPORTS_DIR / "wide_formula_training_pools_domain_hac13.json",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=DATA_DIR / "level1_experimental_alignment_final_anchors",
    )
    parser.add_argument("--contrastive-weights", type=Path, required=True)
    parser.add_argument("--graph-embedding-cache", type=Path, required=True)
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=WEIGHTS_DIR / "module4_wide_formula_adapter_domain_hac13_best.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_REPORTS_DIR / "module4_wide_formula_adapter_domain_hac13_training.json",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hard-negative-k", type=int, default=64)
    parser.add_argument("--hard-negative-weight", type=float, default=0.20)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="按分子式切分使用的独立种子；未指定时复用 --seed",
    )
    parser.add_argument(
        "--initial-adapter-weights",
        type=Path,
        default=None,
        help="可选的模拟同分子式预训练 Adapter；加载后在真实谱组上微调",
    )
    parser.add_argument(
        "--disable-domain-filter",
        action="store_true",
        help="关闭闭域门控，仅用于复现旧版训练",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden_dim < 32 or args.hard_negative_k < 1:
        parser.error("epochs、hidden-dim、hard-negative-k 参数不合法")
    if not 0.05 <= args.val_fraction < 0.5:
        parser.error("val-fraction 必须位于 [0.05, 0.5)")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def load_contrastive(path: Path) -> SpectrumGraphContrastiveModel:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if not domain_metadata_matches(checkpoint.get("chemical_domain")):
        raise ValueError(f"双塔权重缺少一致的 {DOMAIN_NAME} 元数据: {path}")
    model = SpectrumGraphContrastiveModel(
        dare_weights=str(checkpoint.get("dare_weights")),
        node_in_dim=int(checkpoint.get("node_in_dim", 9)),
        edge_in_dim=int(checkpoint.get("edge_in_dim", 3)),
        embed_dim=int(checkpoint.get("embed_dim", EMBEDDING_DIM)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


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


def filter_groups_by_domain(groups: list[dict], apply_filter: bool) -> tuple[list[dict], dict[str, int]]:
    """在训练入口再次门控公式、目标和候选结构，并记录排除原因。"""
    if not apply_filter:
        return groups, {"disabled": len(groups)}
    audit: Counter[str] = Counter()
    kept: list[dict] = []
    for group in groups:
        formula_ok, formula_reasons = domain_result(inspect_formula_domain(group["formula"]))
        if not formula_ok:
            for reason in formula_reasons:
                audit[f"formula:{reason}"] += 1
            continue
        target_ok, target_reasons = domain_result(
            inspect_smiles_domain(
                group["positive_canonical"], formula=group["formula"],
                require_single_component=True, require_neutral_closed_shell=True,
            )
        )
        if not target_ok:
            for reason in target_reasons:
                audit[f"target:{reason}"] += 1
            continue
        candidates = []
        for item in group["candidates"]:
            accepted, reasons = domain_result(
                inspect_smiles_domain(
                    item["smiles"], formula=group["formula"],
                    require_single_component=True, require_neutral_closed_shell=True,
                )
            )
            if accepted:
                candidates.append(item)
            else:
                for reason in reasons:
                    audit[f"candidate:{reason}"] += 1
        target = group["positive_canonical"]
        if len(candidates) < 2 or not any(item["canonical"] == target for item in candidates):
            audit["target_removed_or_candidate_pool_too_small"] += 1
            continue
        copied = dict(group)
        copied["candidates"] = candidates
        copied["candidate_count_full"] = len(candidates)
        kept.append(copied)
    audit["input_groups"] = len(groups)
    audit["retained_groups"] = len(kept)
    return kept, dict(audit)


def validate_formula_checkpoint(checkpoint: dict, path: Path) -> None:
    """拒绝缺失或不匹配当前闭域元素词表的旧 Adapter。"""
    if not domain_metadata_matches(checkpoint.get("chemical_domain")):
        raise ValueError(
            f"Adapter 权重域不匹配: {path}，期望 {DOMAIN_NAME!r}，实际 {checkpoint.get('chemical_domain')!r}"
        )
    if tuple(checkpoint.get("formula_elements", ())) != tuple(DOMAIN_FORMULA_ELEMENTS):
        raise ValueError(f"Adapter 权重 formula_elements 不匹配: {path}")
    if int(checkpoint.get("formula_dim", -1)) != len(DOMAIN_FORMULA_ELEMENTS):
        raise ValueError(f"Adapter 权重 formula_dim 不匹配: {path}")
    if int(FORMULA_DIM) != len(DOMAIN_FORMULA_ELEMENTS):
        raise RuntimeError(
            "当前 FormulaConditionedDualAdapter 仍使用旧元素维度；"
            f"需要与 {DOMAIN_NAME} 的 {len(DOMAIN_FORMULA_ELEMENTS)} 维词表一致"
        )


def _resolved_path_matches(value: object, expected: Path) -> bool:
    """比较缓存记录的来源路径，拒绝静默混用旧产物。"""
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve() == expected.resolve()


def validate_training_artifacts(
    pool_metadata: dict,
    embedding_metadata: dict,
    *,
    wide_pool_cache: Path,
    graph_embedding_cache: Path,
    contrastive_weights: Path,
    apply_domain_filter: bool,
) -> None:
    """核验候选池、图缓存与双塔权重属于同一条正式流水线。"""
    if apply_domain_filter:
        if not domain_metadata_matches(pool_metadata.get("chemical_domain")):
            raise ValueError(f"宽候选池缺少一致的 {DOMAIN_NAME} 元数据: {wide_pool_cache}")
        if not domain_metadata_matches(embedding_metadata.get("chemical_domain")):
            raise ValueError(f"图嵌入缓存缺少一致的 {DOMAIN_NAME} 元数据: {graph_embedding_cache}")
    if not _resolved_path_matches(embedding_metadata.get("wide_pool_cache"), wide_pool_cache):
        raise ValueError("图嵌入缓存与当前 --wide-pool-cache 来源不一致，请重新构建图缓存")
    if not _resolved_path_matches(embedding_metadata.get("contrastive_weights"), contrastive_weights):
        raise ValueError("图嵌入缓存与当前 --contrastive-weights 来源不一致，请重新构建图缓存")


def load_initial_adapter(model: FormulaConditionedDualAdapter, path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {"path": None}
    if not path.is_file():
        raise FileNotFoundError(f"初始 Adapter 权重不存在: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"初始 Adapter 权重格式无效: {path}")
    validate_formula_checkpoint(checkpoint, path)
    if int(checkpoint.get("embedding_dim", -1)) != int(model.embedding_dim):
        raise ValueError(f"初始 Adapter embedding_dim 不匹配: {path}")
    if int(checkpoint.get("hidden_dim", -1)) != int(model.query_adapter[0].out_features):
        raise ValueError(f"初始 Adapter hidden_dim 不匹配: {path}")
    if abs(float(checkpoint.get("residual_scale", float("nan"))) - float(model.residual_scale)) > 1e-12:
        raise ValueError(f"初始 Adapter residual_scale 不匹配: {path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "path": str(path.resolve()),
        "method": checkpoint.get("method"),
        "chemical_domain": checkpoint.get("chemical_domain"),
        "formula_elements": list(checkpoint.get("formula_elements", [])),
        "best_validation": checkpoint.get("best_validation"),
    }


def prepare_examples(
    groups: list[dict],
    records: dict,
    contrastive,
    embedding_cache: dict[str, torch.Tensor],
    *,
    label: str,
) -> list[dict]:
    """把候选池整理为训练样本；同一查询只做一次光谱编码。"""
    examples = []
    query_cache: dict[str, torch.Tensor] = {}
    total = len(groups)
    for position, group in enumerate(groups, start=1):
        items = [item for item in group["candidates"] if item["canonical"] in embedding_cache]
        target = next((index for index, item in enumerate(items) if item["canonical"] == group["positive_canonical"]), None)
        if target is None or len(items) < 2:
            continue
        query_record = records[group["cas"]]
        cas = str(group["cas"])
        query = query_cache.get(cas)
        if query is None:
            query = encode_query(contrastive, query_record["spectrum"], query_record.get("valid_mask"))
            query_cache[cas] = query
        molecules = torch.stack([embedding_cache[item["canonical"]] for item in items]).float()
        formula = formula_vector(group["formula"])
        examples.append(
            {
                "cas": group["cas"],
                "formula": group["formula"],
                "query": query,
                "molecules": molecules,
                "formula_features": formula,
                "positive_index": int(target),
            }
        )
        if position == 1 or position % 25 == 0 or position == total:
            print(f"准备{label}样本 {position}/{total}，有效={len(examples)}", flush=True)
    return examples


def score_example(model: FormulaConditionedDualAdapter, example: dict) -> torch.Tensor:
    query = example["query"].to(DEVICE).unsqueeze(0)
    molecules = example["molecules"].to(DEVICE).unsqueeze(0)
    formula = example["formula_features"].to(DEVICE).unsqueeze(0)
    return model(query, molecules, formula).squeeze(0)


def metrics(model: FormulaConditionedDualAdapter, examples: list[dict]) -> dict[str, float]:
    ranks = []
    with torch.inference_mode():
        for example in examples:
            scores = score_example(model, example)
            order = torch.argsort(scores, descending=True).tolist()
            ranks.append(order.index(example["positive_index"]) + 1)
    values = np.asarray(ranks, dtype=np.int64)
    return {
        "count": int(len(values)),
        "top1": float(np.mean(values <= 1)) if len(values) else 0.0,
        "top5": float(np.mean(values <= 5)) if len(values) else 0.0,
        "top10": float(np.mean(values <= 10)) if len(values) else 0.0,
        "top24": float(np.mean(values <= 24)) if len(values) else 0.0,
        "mrr": float(np.mean(1.0 / values)) if len(values) else 0.0,
        "median_rank": float(np.median(values)) if len(values) else None,
    }


def main() -> None:
    args = parse_args()
    if int(FORMULA_DIM) != len(DOMAIN_FORMULA_ELEMENTS):
        raise RuntimeError(
            f"Adapter 当前分子式维度 {FORMULA_DIM} 与闭域词表 {len(DOMAIN_FORMULA_ELEMENTS)} 不一致"
        )
    if args.smoke:
        args.max_groups = args.max_groups or 8
        args.max_groups = min(args.max_groups, 8)
        args.epochs = min(args.epochs, 1)
    set_seed(args.seed)
    started = time.perf_counter()
    records = load_records(args.query_dir)
    apply_domain_filter = not args.disable_domain_filter
    groups, pool_metadata = load_wide_groups(
        args.wide_pool_cache,
        records,
        args.max_groups,
        0,
        return_metadata=True,
    )
    groups, domain_audit = filter_groups_by_domain(groups, apply_domain_filter)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_groups, val_groups = split_by_formula(groups, args.val_fraction, split_seed)
    contrastive = load_contrastive(args.contrastive_weights)
    embedding_cache, embedding_metadata = load_embedding_cache(
        args.graph_embedding_cache,
        return_metadata=True,
    )
    validate_training_artifacts(
        pool_metadata,
        embedding_metadata,
        wide_pool_cache=args.wide_pool_cache,
        graph_embedding_cache=args.graph_embedding_cache,
        contrastive_weights=args.contrastive_weights,
        apply_domain_filter=apply_domain_filter,
    )
    print(f"开始准备训练样本 groups={len(train_groups)}", flush=True)
    train_examples = prepare_examples(train_groups, records, contrastive, embedding_cache, label="训练")
    print(f"开始准备验证样本 groups={len(val_groups)}", flush=True)
    val_examples = prepare_examples(val_groups, records, contrastive, embedding_cache, label="验证")
    if len(train_examples) < 2 or len(val_examples) < 2:
        raise RuntimeError(f"有效训练/验证组过少: {len(train_examples)}/{len(val_examples)}")
    print(
        f"宽池分子式条件双塔 train/val={len(train_examples)}/{len(val_examples)} "
        f"device={DEVICE}；完整同式池监督，不使用 PubChem rank/teacher",
        flush=True,
    )
    model = FormulaConditionedDualAdapter(
        embedding_dim=int(contrastive.embed_dim),
        formula_dim=FORMULA_DIM,
        hidden_dim=args.hidden_dim,
    ).to(DEVICE)
    initialization = load_initial_adapter(model, args.initial_adapter_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    baseline = metrics(model, val_examples)
    print(f"冻结身份初始化验证: {baseline}", flush=True)
    best_key = (-1.0, -1.0, -1.0)
    best_validation = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for position in order:
            example = train_examples[position]
            scores = score_example(model, example)
            target = example["positive_index"]
            label_loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=DEVICE))
            negative_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=DEVICE)
            negative_mask[target] = False
            negative_indices = torch.where(negative_mask)[0]
            hard_count = min(args.hard_negative_k, int(negative_indices.numel()))
            hard_order = torch.argsort(scores.detach()[negative_indices], descending=True)[:hard_count]
            hard_indices = negative_indices[hard_order]
            positive = scores[target]
            hard_loss = F.softplus(args.hard_negative_margin - positive + scores[hard_indices]).mean()
            loss = label_loss + args.hard_negative_weight * hard_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        validation = metrics(model, val_examples)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
            f"val Top1/5/10/24={validation['top1']:.3f}/{validation['top5']:.3f}/"
            f"{validation['top10']:.3f}/{validation['top24']:.3f} MRR={validation['mrr']:.4f}",
            flush=True,
        )
        key = (validation["top1"], validation["top5"], validation["top10"])
        if key > best_key:
            best_key = key
            best_validation = validation
            args.output_weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "method": "wide_formula_conditioned_dual_adapter_no_pubchem_order",
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": int(contrastive.embed_dim),
                    "formula_dim": FORMULA_DIM,
                    "hidden_dim": args.hidden_dim,
                    "residual_scale": model.residual_scale,
                    "contrastive_weights": str(args.contrastive_weights.resolve()),
                    "graph_embedding_cache": str(args.graph_embedding_cache.resolve()),
                    "split_seed": int(split_seed),
                    "train_groups": [group["cas"] for group in train_groups],
                    "validation_groups": [group["cas"] for group in val_groups],
                    # 同时保存公式白名单，供后续精排器进行公式级监督重叠审计。
                    "train_formulas": sorted({str(group["formula"]) for group in train_groups}),
                    "validation_formulas": sorted({str(group["formula"]) for group in val_groups}),
                    "best_validation": validation,
                    "chemical_domain": domain_metadata() if apply_domain_filter else None,
                    "formula_elements": list(DOMAIN_FORMULA_ELEMENTS),
                    "initial_adapter": initialization,
                },
                args.output_weights,
            )
    report = {
        "schema_version": 1,
        "method": "wide_formula_conditioned_dual_adapter_no_pubchem_order",
        "status": "completed",
        "groups": len(groups),
        "train_groups": len(train_examples),
        "validation_groups": len(val_examples),
        "split_seed": int(split_seed),
        "train_formulas": sorted({str(group["formula"]) for group in train_groups}),
        "validation_formulas": sorted({str(group["formula"]) for group in val_groups}),
        "baseline_validation": baseline,
        "best_validation": best_validation,
        "weights": str(args.output_weights.resolve()),
        "candidate_order_policy": "每步随机置换；不读取 PubChem CID/API rank、缓存位置或 teacher 分数",
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "chemical_domain": domain_metadata() if apply_domain_filter else None,
        "domain_filter_enabled": apply_domain_filter,
        "domain_audit": domain_audit,
        "initial_adapter": initialization,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"权重: {args.output_weights.resolve()}\n报告: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""训练已知分子式条件下的宽候选池 Listwise 精排器。

候选池先经过严格分子式筛选，再由冻结的光谱-分子图双塔进行粗召回。
精排器只读取双塔嵌入、结构描述符和分子式计数向量，不读取 PubChem
CID/API 返回顺序、候选原始位置或旧 teacher 分数。验证时不保护正样本，
并同时报告粗召回率和条件精排 Top-k，避免把召回失败误报成排序失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch
from torch_geometric.utils.smiles import from_smiles

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
from physchemrag.module4_cascade.learned_cascade import (  # noqa: E402
    CandidateIndependentFormulaRanker,
    FormulaConditionedSetTransformerRanker,
    SpectrumGraphContrastiveModel,
    listwise_set_loss,
)
from physchemrag.module4_cascade.forward_spectral_features import compare_forward_spectra  # noqa: E402
from physchemrag.module4_cascade.spectral_views import prepare_masked_spectrum  # noqa: E402
from physchemrag.shared.chemical_domain import (  # noqa: E402
    DOMAIN_NAME,
    FORMULA_ELEMENTS as DOMAIN_FORMULA_ELEMENTS,
    domain_metadata,
    domain_metadata_matches,
    inspect_smiles_domain,
)
from physchemrag.shared.molecular_formula import (  # noqa: E402
    canonical_smiles,
    formula_from_smiles,
    normalize_formula,
)

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 512
SET_STATS_DIM = 5
FEATURE_DIM = 6  # 双塔粗分数 + 五个不依赖标签的结构描述符
FUSED_FEATURE_DIM = 8  # 双塔/分子式双路粗分数 + 五个结构描述符
FORWARD_FEATURE_DIM = 15  # 双路粗分数 + 7 个候选谱一致性特征 + 五个结构描述符


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-pool-cache",
        type=Path,
        default=OUTPUT_REPORTS_DIR / "wide_formula_training_pools_offline_v2.json",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=DATA_DIR / "level1_experimental_alignment_final_anchors",
    )
    parser.add_argument("--contrastive-weights", type=Path, required=True)
    parser.add_argument(
        "--formula-adapter-weights",
        type=Path,
        default=None,
        help="可选：完整同分子式训练得到的条件化双塔，用于第二路粗召回",
    )
    parser.add_argument(
        "--graph-embedding-cache",
        type=Path,
        default=None,
        help="可选的全池分子图嵌入缓存；避免每次训练重复编码候选",
    )
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=None,
        help="可选的 SpecGNN 候选前向谱缓存；raw_cosine_forward 策略必须提供",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="可选的确定性 Listwise 特征缓存；可跨训练 seed 复用",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="忽略并覆盖已有 Listwise 特征缓存",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=WEIGHTS_DIR / "module4_formula_wide_listwise_ranker_best.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_REPORTS_DIR / "module4_formula_wide_listwise_ranker_training.json",
    )
    parser.add_argument("--candidate-limit", type=int, default=512)
    parser.add_argument(
        "--union-width",
        type=int,
        default=0,
        help="启用分子式适配器时每一路粗召回宽度；0 表示使用 candidate-limit",
    )
    parser.add_argument(
        "--shortlist-policy",
        choices=("auto", "formula_adapter", "fused_union", "base"),
        default="auto",
        help="候选集合来源；auto 在有适配器时使用分子式适配器主召回",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--hard-negative-k", type=int, default=32)
    parser.add_argument(
        "--ranker-architecture",
        choices=("candidate_independent", "set_transformer"),
        default="candidate_independent",
        help="精排器架构；candidate_independent 不让候选分数依赖同池组成",
    )
    parser.add_argument(
        "--candidate-feature-policy",
        choices=("raw_cosine", "raw_cosine_forward", "rank_normalized"),
        default="raw_cosine",
        help="候选特征策略；raw_cosine_forward 额外使用候选预测谱与实验谱的物理一致性",
    )
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-candidates-per-formula", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="按分子式切分使用的独立种子；未指定时复用 --seed",
    )
    parser.add_argument(
        "--allow-adapter-split-overlap",
        action="store_true",
        help="允许分子式适配器监督训练组与当前验证组重叠；正式评测不应启用",
    )
    parser.add_argument(
        "--no-target-protect-training",
        action="store_true",
        help="训练 shortlist 不强制加入正样本；用于严格测量粗召回失败",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.candidate_limit < 2 or args.union_width < 0 or args.epochs < 1 or args.encode_batch_size < 1:
        parser.error("candidate-limit、epochs、encode-batch-size 必须大于 1/0")
    if not 0.05 <= args.val_fraction < 0.5:
        parser.error("val-fraction 必须位于 [0.05, 0.5)")
    if args.ranker_architecture == "candidate_independent" and args.formula_adapter_weights is None:
        parser.error("candidate_independent 精排器需要 --formula-adapter-weights，以保持分子式条件输入一致")
    if args.candidate_feature_policy == "raw_cosine_forward" and args.prediction_cache is None:
        parser.error("raw_cosine_forward 必须提供 --prediction-cache")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def path_signature(path: Path | None) -> dict | None:
    """返回输入文件或 PT 目录的快速失效签名。"""
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    digest = hashlib.sha256()
    files = sorted(resolved.glob("*.pt"))
    for item in files:
        stat = item.stat()
        digest.update(f"{item.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {
        "path": str(resolved),
        "pt_files": len(files),
        "manifest_sha256": digest.hexdigest(),
    }


def feature_cache_metadata(
    args: argparse.Namespace,
    split_seed: int,
    shortlist_policy: str,
) -> dict:
    """描述所有会改变确定性 Listwise 输入特征的配置。"""
    return {
        "schema_version": 1,
        "chemical_domain": domain_metadata(),
        "wide_pool_cache": path_signature(args.wide_pool_cache),
        "query_dir": path_signature(args.query_dir),
        "contrastive_weights": path_signature(args.contrastive_weights),
        "formula_adapter_weights": path_signature(args.formula_adapter_weights),
        "graph_embedding_cache": path_signature(args.graph_embedding_cache),
        "prediction_cache": path_signature(args.prediction_cache),
        "candidate_limit": int(args.candidate_limit),
        "union_width": int(args.union_width),
        "shortlist_policy": shortlist_policy,
        "candidate_feature_policy": args.candidate_feature_policy,
        "val_fraction": float(args.val_fraction),
        "split_seed": int(split_seed),
        "max_groups": int(args.max_groups),
        "max_candidates_per_formula": int(args.max_candidates_per_formula),
        "target_protect_training": not args.no_target_protect_training,
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    """先写临时文件再原子替换，避免中断后留下不完整缓存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_feature_cache(args: argparse.Namespace, expected_metadata: dict) -> dict | None:
    """加载与当前输入、切分及候选策略完全匹配的特征缓存。"""
    if args.feature_cache is None or args.rebuild_feature_cache or not args.feature_cache.exists():
        return None
    payload = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    if payload.get("cache_metadata") != expected_metadata:
        raise RuntimeError(
            f"Listwise 特征缓存与当前输入或参数不一致: {args.feature_cache}；"
            "请更换缓存路径或加入 --rebuild-feature-cache"
        )
    if not domain_metadata_matches(payload.get("chemical_domain")):
        raise ValueError(f"Listwise 特征缓存缺少一致的 {DOMAIN_NAME} 元数据")
    print(f"复用 Listwise 特征缓存: {args.feature_cache.resolve()}", flush=True)
    return payload


def canonical(value: str) -> str:
    return canonical_smiles(str(value or ""), isomeric=False) or ""


def load_records(directory: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cas = str(payload.get("cas", path.stem))
        smiles = str(payload.get("connectivity_smiles") or payload.get("smiles") or "")
        formula = normalize_formula(
            payload.get("molecular_formula") or formula_from_smiles(smiles)
        )
        spectrum = torch.as_tensor(payload.get("spectrum"), dtype=torch.float32).flatten()
        valid_mask = torch.as_tensor(
            payload.get("valid_mask", torch.ones_like(spectrum, dtype=torch.bool)),
            dtype=torch.bool,
        ).flatten()
        key = canonical(smiles)
        domain = inspect_smiles_domain(smiles, formula)
        if (
            key
            and domain.valid
            and spectrum.numel() == 1800
            and valid_mask.numel() == 1800
            and bool(valid_mask.any())
            and torch.isfinite(spectrum).all()
        ):
            records[cas] = {
                "spectrum": spectrum,
                "valid_mask": valid_mask,
                "smiles": smiles,
                "canonical": key,
                "formula": formula,
            }
    return records


def load_wide_groups(
    path: Path,
    records: dict[str, dict],
    max_groups: int,
    max_candidates_per_formula: int,
    *,
    return_metadata: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pools = dict(payload.get("formula_pools", {}))
    raw_groups = payload.get("groups", payload)
    trusted_domain_cache = bool(payload.get("domain_filter_enabled")) and domain_metadata_matches(
        payload.get("chemical_domain")
    )
    groups: list[dict] = []
    # 同一分子式可能对应多条实验谱；候选池只规范化和校验一次。
    # 候选列表在下游按只读对象使用，共享引用不会改变候选内容或排序。
    candidates_by_formula: dict[str, list[dict]] = {}
    for raw in raw_groups:
        cas = str(raw.get("cas", ""))
        if cas not in records:
            continue
        formula = normalize_formula(raw.get("formula") or formula_from_smiles(records[cas]["smiles"]))
        if formula != records[cas].get("formula"):
            continue
        positive = canonical(str(raw.get("positive_canonical") or records[cas]["smiles"]))
        if formula not in candidates_by_formula:
            raw_pool = list(pools.get(formula, []))
            prepared_candidates: list[dict] = []
            seen: set[str] = set()
            for item in raw_pool:
                smiles = str(item.get("smiles") or item.get("canonical") or "")
                if trusted_domain_cache:
                    # 该文件在构建阶段已经执行同一 domain_hac13 过滤并保存 canonical。
                    # 只有域元数据明确匹配时才跳过昂贵的逐候选 RDKit 重复复核。
                    key = str(item.get("canonical") or "")
                    if not key or key in seen:
                        continue
                else:
                    key = canonical(smiles)
                    domain = inspect_smiles_domain(smiles, formula)
                    if not key or key in seen or not domain.valid:
                        continue
                    if normalize_formula(formula_from_smiles(smiles)) != formula:
                        continue
                seen.add(key)
                prepared_candidates.append({"smiles": smiles, "canonical": key})
            candidates_by_formula[formula] = prepared_candidates
        candidates = candidates_by_formula[formula]
        if max_candidates_per_formula > 0:
            candidates = candidates[:max_candidates_per_formula]
        positive_index = next(
            (index for index, item in enumerate(candidates) if item["canonical"] == positive),
            None,
        )
        if positive_index is None or len(candidates) < 2:
            continue
        groups.append(
            {
                "cas": cas,
                "formula": formula,
                "positive_canonical": positive,
                "positive_index_full": int(positive_index),
                "candidate_count_full": len(candidates),
                "target_in_pool": True,
                "formula_exact_match": True,
                "candidates": candidates,
            }
        )
        if max_groups > 0 and len(groups) >= max_groups:
            break
    if not groups:
        raise RuntimeError("宽同分子式池没有可用训练组")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"formula_pools", "groups"}
    }
    return (groups, metadata) if return_metadata else groups


def split_by_formula(groups: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        buckets[group["formula"]].append(group)
    keys = list(buckets)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(groups) * fraction))
    validation: list[dict] = []
    train: list[dict] = []
    for key in keys:
        (validation if len(validation) < target else train).extend(buckets[key])
    if not train or not validation:
        raise RuntimeError(f"按分子式切分后 train/validation 为空: {len(train)}/{len(validation)}")
    return train, validation


def graph_data(smiles: str):
    data = from_smiles(smiles)
    if data is None or data.x is None:
        return None
    data.x = data.x.float()
    data.edge_attr = (
        data.edge_attr.float()
        if data.edge_attr is not None
        else torch.zeros((0, 3), dtype=torch.float32)
    )
    return data


def structure_features(smiles: str) -> list[float]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return [0.0] * 5
    heavy = molecule.GetNumHeavyAtoms()
    rotatable = sum(
        int(b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic() and not b.IsInRing())
        for b in molecule.GetBonds()
    )
    rings = molecule.GetRingInfo().NumRings()
    aromatic = sum(int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()) / max(heavy, 1)
    hetero = sum(int(atom.GetAtomicNum() not in (1, 6)) for atom in molecule.GetAtoms()) / max(heavy, 1)
    return [heavy / 50.0, rotatable / 20.0, rings / 10.0, aromatic, hetero]


def set_stats(coarse_scores: torch.Tensor) -> torch.Tensor:
    """从粗召回分数构造集合统计；不读取候选原始位置。"""
    values = coarse_scores.float()
    probabilities = torch.softmax(values / 0.20, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum()
    entropy = entropy / max(float(np.log(max(values.numel(), 2))), 1.0)
    top = torch.topk(values, min(5, values.numel())).values
    gap = top[0] - top[1] if top.numel() > 1 else values.new_zeros(())
    return torch.stack(
        [
            values.new_tensor(values.numel() / 512.0),
            values.mean(),
            values.std(unbiased=False),
            gap,
            entropy,
        ]
    ).float()


def encode_query(
    contrastive: SpectrumGraphContrastiveModel,
    spectrum: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # spectrum: [L]；valid_mask: [L] -> filled: [1,1,L]；输出查询向量 [D]。
    filled, _ = prepare_masked_spectrum(spectrum, valid_mask)
    batch = filled.to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        return contrastive.encode_spectrum(batch).squeeze(0).cpu()


def encode_graphs(
    contrastive: SpectrumGraphContrastiveModel,
    items: list[dict],
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """批量编码图；返回 CPU 嵌入，避免把全池驻留在显存。"""
    embeddings: dict[str, torch.Tensor] = {}
    failed: list[str] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        valid_items = []
        graph_list = []
        for item in chunk:
            data = graph_data(item["smiles"])
            if data is None:
                failed.append(item["canonical"])
                continue
            graph_list.append(data)
            valid_items.append(item)
        if not graph_list:
            continue
        batch = Batch.from_data_list(graph_list).to(DEVICE, non_blocking=True)
        with torch.inference_mode():
            output = contrastive.encode_graph(batch).cpu()
        for item, vector in zip(valid_items, output):
            embeddings[item["canonical"]] = vector.float()
    return embeddings, failed


def load_embedding_cache(
    path: Path | None,
    *,
    return_metadata: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict]:
    if path is None:
        return ({}, {}) if return_metadata else {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vectors = torch.as_tensor(payload.get("embeddings"), dtype=torch.float32)
    index = {str(key): int(value) for key, value in dict(payload.get("index", {})).items()}
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"图嵌入缓存形状异常: {tuple(vectors.shape)}")
    output = {}
    for key, position in index.items():
        if not 0 <= position < vectors.shape[0]:
            raise ValueError(f"图嵌入缓存索引越界: {key} -> {position}")
        output[key] = vectors[position].contiguous()
    if len(output) != len(index):
        raise ValueError("图嵌入缓存存在重复索引键")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"embeddings", "index"}
    }
    return (output, metadata) if return_metadata else output


def load_prediction_map(path: Path | None) -> dict[str, torch.Tensor]:
    """加载候选前向谱，并按 canonical SMILES 建立无序结构索引。"""
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    predictions = torch.as_tensor(payload.get("predictions"), dtype=torch.float32)
    if predictions.ndim != 2 or predictions.shape[1] != 1800:
        raise ValueError(f"前向谱缓存形状异常: {tuple(predictions.shape)}")
    output: dict[str, torch.Tensor] = {}
    for key, position in dict(payload.get("index", {})).items():
        raw_key = str(key)
        normalized = canonical(raw_key)
        vector = predictions[int(position)].contiguous()
        if raw_key:
            output[raw_key] = vector
        if normalized:
            output[normalized] = vector
    if not output:
        raise ValueError(f"前向谱缓存没有有效 canonical 索引: {path}")
    return output


def load_formula_adapter(path: Path | None) -> FormulaConditionedDualAdapter | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if not domain_metadata_matches(checkpoint.get("chemical_domain")):
        raise ValueError(
            f"分子式适配器缺少一致的 {DOMAIN_NAME} 元数据: {path}"
        )
    if tuple(checkpoint.get("formula_elements", ())) != tuple(DOMAIN_FORMULA_ELEMENTS):
        raise ValueError(
            f"分子式适配器元素词表不匹配: {path}; "
            f"实际={checkpoint.get('formula_elements')!r}，期望={list(DOMAIN_FORMULA_ELEMENTS)!r}"
        )
    if int(checkpoint.get("formula_dim", -1)) != len(DOMAIN_FORMULA_ELEMENTS):
        raise ValueError(
            f"分子式适配器维度不匹配: {path}; "
            f"实际={checkpoint.get('formula_dim')!r}，期望={len(DOMAIN_FORMULA_ELEMENTS)}"
        )
    model = FormulaConditionedDualAdapter(
        embedding_dim=int(checkpoint.get("embedding_dim", EMBEDDING_DIM)),
        formula_dim=int(checkpoint.get("formula_dim", FORMULA_DIM)),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        residual_scale=float(checkpoint.get("residual_scale", 0.35)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def rank_normalized(values: torch.Tensor) -> torch.Tensor:
    """将分数转换为 [0,1] 的集合内相对名次；不依赖输入位置。"""
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return 1.0 - ranks / max(float(values.numel() - 1), 1.0)


def prepare_group(
    group: dict,
    records: dict[str, dict],
    contrastive: SpectrumGraphContrastiveModel,
    formula_adapter: FormulaConditionedDualAdapter | None,
    embedding_cache: dict[str, torch.Tensor],
    candidate_limit: int,
    union_width: int,
    shortlist_policy: str,
    protect_positive: bool,
    encode_batch_size: int,
    candidate_feature_policy: str,
    prediction_map: dict[str, torch.Tensor],
) -> tuple[dict, dict[str, float]]:
    """粗召回并构造训练/验证集合；验证不调用目标保护。"""
    query_record = records[group["cas"]]
    query = encode_query(
        contrastive,
        query_record["spectrum"],
        query_record.get("valid_mask"),
    )
    missing = [item for item in group["candidates"] if item["canonical"] not in embedding_cache]
    if missing:
        fresh, failed = encode_graphs(contrastive, missing, encode_batch_size)
        embedding_cache.update(fresh)
        if failed:
            group = dict(group)
            group["candidates"] = [item for item in group["candidates"] if item["canonical"] not in set(failed)]
    candidates = [item for item in group["candidates"] if item["canonical"] in embedding_cache]
    if len(candidates) < 2:
        raise ValueError(f"有效图候选不足: {group['cas']}")
    positive_key = group["positive_canonical"]
    full_positive = next((i for i, item in enumerate(candidates) if item["canonical"] == positive_key), None)
    if full_positive is None:
        raise ValueError(f"正样本图解析失败: {group['cas']}")
    candidate_matrix = torch.stack([embedding_cache[item["canonical"]] for item in candidates])
    coarse_full = (candidate_matrix @ query.view(-1, 1)).squeeze(1)
    formula_features = formula_vector(group["formula"])
    formula_full = None
    if formula_adapter is not None:
        with torch.inference_mode():
            formula_full = formula_adapter(
                query.to(DEVICE).unsqueeze(0),
                candidate_matrix.to(DEVICE).unsqueeze(0),
                formula_features.to(DEVICE).unsqueeze(0),
            ).squeeze(0).cpu()
    base_rank = rank_normalized(coarse_full)
    formula_rank = rank_normalized(formula_full) if formula_full is not None else None
    if candidate_feature_policy not in {"raw_cosine", "raw_cosine_forward", "rank_normalized"}:
        raise ValueError(f"未知 candidate_feature_policy: {candidate_feature_policy}")
    # FormulaConditionedDualAdapter 输出带温度缩放的 logits；raw_cosine
    # 策略将其还原为余弦分数，与评测端的适配向量内积保持一致。
    adapter_scale = None
    if formula_full is not None:
        adapter_scale = formula_adapter.logit_scale.detach().exp().clamp(1.0, 100.0).cpu()
    keep_count = min(candidate_limit, len(candidates))
    if formula_adapter is not None:
        branch_width = min(len(candidates), int(union_width or keep_count))
        base_order = torch.argsort(coarse_full, descending=True)[:branch_width]
        formula_order = torch.argsort(formula_full, descending=True)[:branch_width]
        if shortlist_policy == "formula_adapter":
            selected_indices = formula_order[:keep_count].tolist()
        elif shortlist_policy == "base":
            selected_indices = base_order[:keep_count].tolist()
        else:
            union_indices = list(dict.fromkeys(base_order.tolist() + formula_order.tolist()))
            combined_full = 0.5 * (base_rank + formula_rank)
            if len(union_indices) > keep_count:
                union_tensor = torch.tensor(union_indices, dtype=torch.long)
                union_order = torch.argsort(combined_full[union_tensor], descending=True)[:keep_count]
                selected_indices = union_tensor[union_order].tolist()
            else:
                selected_indices = union_indices
    else:
        if shortlist_policy == "formula_adapter":
            raise ValueError("shortlist-policy=formula_adapter 需要 --formula-adapter-weights")
        coarse_order = torch.argsort(coarse_full, descending=True)
        selected_indices = coarse_order[:keep_count].tolist()
    coarse_recall = float(full_positive in set(selected_indices))
    if protect_positive and full_positive not in selected_indices:
        selected_indices[-1] = full_positive
    selected_indices = list(dict.fromkeys(selected_indices))
    selected = [candidates[index] for index in selected_indices]
    positive_index = next(
        (i for i, item in enumerate(selected) if item["canonical"] == positive_key),
        -1,
    )
    selected_embeddings = torch.stack([embedding_cache[item["canonical"]] for item in selected])
    selected_base_rank = base_rank[selected_indices].float()
    selected_formula_rank = formula_rank[selected_indices].float() if formula_rank is not None else None
    if selected_formula_rank is None:
        selected_coarse = coarse_full[selected_indices].float()
    elif candidate_feature_policy == "raw_cosine":
        # 原始余弦是候选独立基准；不会因同池候选增删而改变。
        selected_base_cosine = coarse_full[selected_indices].float()
        selected_formula_cosine = (formula_full[selected_indices] / adapter_scale).float()
        if shortlist_policy == "formula_adapter":
            selected_coarse = selected_formula_cosine
        elif shortlist_policy == "base":
            selected_coarse = selected_base_cosine
        else:
            selected_coarse = 0.5 * (selected_base_cosine + selected_formula_cosine)
    elif shortlist_policy == "formula_adapter":
        selected_coarse = selected_formula_rank
    elif shortlist_policy == "base":
        selected_coarse = selected_base_rank
    else:
        selected_coarse = 0.5 * (selected_base_rank + selected_formula_rank)
    if selected_formula_rank is None:
        feature_rows = [
            [float(selected_coarse[i]), *structure_features(item["smiles"])]
            for i, item in enumerate(selected)
        ]
    elif candidate_feature_policy in {"raw_cosine", "raw_cosine_forward"}:
        selected_formula_cosine = (formula_full[selected_indices] / adapter_scale).float()
        selected_base_cosine = coarse_full[selected_indices].float()
        forward_rows: list[list[float]] | None = None
        if candidate_feature_policy == "raw_cosine_forward":
            missing_prediction = [
                item["canonical"] for item in selected if item["canonical"] not in prediction_map
            ]
            if missing_prediction:
                raise ValueError(
                    f"raw_cosine_forward 缺少 {len(missing_prediction)} 个候选的前向谱；"
                    "请先用同一宽池构建 --prediction-cache"
                )
            # predicted/target: [N,1800]；仅保留前 7 个候选相关特征，
            # 第 8 个 target entropy 对同一查询恒定，不能提供候选区分信息。
            predicted = torch.stack([prediction_map[item["canonical"]] for item in selected])
            target, _ = prepare_masked_spectrum(
                query_record["spectrum"], query_record.get("valid_mask")
            )
            forward = compare_forward_spectra(
                predicted.float(), target.flatten().unsqueeze(0).expand(predicted.shape[0], -1)
            )[:, :7]
            forward_rows = forward.float().tolist()
        feature_rows = [
            [
                float(selected_formula_cosine[i]),
                float(selected_base_cosine[i]),
                float(selected_formula_cosine[i] - selected_base_cosine[i]),
                *([] if forward_rows is None else forward_rows[i]),
                *structure_features(item["smiles"]),
            ]
            for i, item in enumerate(selected)
        ]
    else:
        feature_rows = [
            [
                float(selected_coarse[i]),
                float(selected_base_rank[i]),
                float(selected_formula_rank[i]),
                *structure_features(item["smiles"]),
            ]
            for i, item in enumerate(selected)
        ]
    features = torch.tensor(feature_rows, dtype=torch.float32)
    return {
        "cas": group["cas"],
        "formula": group["formula"],
        "formula_features": formula_features,
        "embeddings": selected_embeddings,
        "features": features,
        "query": query,
        "set_stats": set_stats(selected_coarse),
        "coarse_scores": selected_coarse,
        "positive_index": int(positive_index),
        "candidate_count_full": len(candidates),
        "shortlist_count": len(selected),
        "coarse_recall": coarse_recall,
        "target_protected": bool(protect_positive and not bool(coarse_recall)),
        "candidate_feature_policy": candidate_feature_policy,
    }, {"coarse_recall": coarse_recall, "candidate_count": float(len(candidates))}


def rank_metrics(ranks: list[int], total: int) -> dict[str, float]:
    values = np.asarray(ranks, dtype=np.int64)
    if not len(values):
        return {"evaluated": 0, "total": int(total), "top1": 0.0, "top5": 0.0, "top10": 0.0, "top24": 0.0, "mrr": 0.0}
    return {
        "evaluated": int(len(values)),
        "total": int(total),
        "top1": float(np.sum(values <= 1) / total),
        "top5": float(np.sum(values <= 5) / total),
        "top10": float(np.sum(values <= 10) / total),
        "top24": float(np.sum(values <= 24) / total),
        "mrr": float(np.sum(1.0 / values) / total),
        "conditional_top1": float(np.mean(values <= 1)),
        "conditional_top5": float(np.mean(values <= 5)),
        "conditional_top10": float(np.mean(values <= 10)),
        "conditional_top24": float(np.mean(values <= 24)),
    }


def evaluate(
    model: FormulaConditionedSetTransformerRanker,
    prepared: list[dict],
    device: torch.device,
    include_queries: bool = False,
) -> dict:
    model.eval()
    ranks: list[int] = []
    coarse_ranks: list[int] = []
    recalls: list[float] = []
    query_rows: list[dict] = []
    with torch.inference_mode():
        for item in prepared:
            recalls.append(float(item["coarse_recall"]))
            coarse_order = torch.argsort(item["coarse_scores"], descending=True).tolist()
            coarse_rank = None
            if item["positive_index"] in coarse_order:
                coarse_rank = coarse_order.index(item["positive_index"]) + 1
                coarse_ranks.append(coarse_rank)
            ranker_rank = None
            if not item["target_protected"] and item["coarse_recall"] < 0.5:
                if include_queries:
                    query_rows.append(
                        {
                            "cas": item["cas"],
                            "formula": item["formula"],
                            "coarse_recall": float(item["coarse_recall"]),
                            "candidate_count_full": int(item["candidate_count_full"]),
                            "shortlist_count": int(item["shortlist_count"]),
                            "coarse_rank": coarse_rank,
                            "ranker_rank": None,
                        }
                    )
                continue
            scores, _, _ = model(
                item["embeddings"].to(device),
                item["features"].to(device),
                item["query"].to(device),
                item["formula_features"].to(device),
                item["set_stats"].to(device),
                item["coarse_scores"].to(device),
            )
            order = torch.argsort(scores, descending=True).tolist()
            ranker_rank = order.index(item["positive_index"]) + 1
            ranks.append(ranker_rank)
            if include_queries:
                query_rows.append(
                    {
                        "cas": item["cas"],
                        "formula": item["formula"],
                        "coarse_recall": float(item["coarse_recall"]),
                        "candidate_count_full": int(item["candidate_count_full"]),
                        "shortlist_count": int(item["shortlist_count"]),
                        "coarse_rank": coarse_rank,
                        "ranker_rank": ranker_rank,
                    }
                )
    result = {
        "shortlist_recall": float(np.mean(recalls)) if recalls else 0.0,
        "ranker": rank_metrics(ranks, len(prepared)),
        "coarse": rank_metrics(coarse_ranks, len(prepared)),
    }
    if include_queries:
        result["queries"] = query_rows
    return result


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_groups = args.max_groups or 8
        args.max_groups = min(args.max_groups, 8)
        args.epochs = min(args.epochs, 1)
        args.candidate_limit = min(args.candidate_limit, 32)
    set_seed(args.seed)
    started = time.perf_counter()
    records = load_records(args.query_dir)
    groups = load_wide_groups(
        args.wide_pool_cache,
        records,
        args.max_groups,
        args.max_candidates_per_formula,
    )
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_groups, val_groups = split_by_formula(groups, args.val_fraction, split_seed)
    checkpoint = torch.load(args.contrastive_weights, map_location=DEVICE, weights_only=False)
    contrastive = SpectrumGraphContrastiveModel(
        dare_weights=str(checkpoint.get("dare_weights")),
        node_in_dim=int(checkpoint.get("node_in_dim", 9)),
        edge_in_dim=int(checkpoint.get("edge_in_dim", 3)),
        embed_dim=int(checkpoint.get("embed_dim", EMBEDDING_DIM)),
    ).to(DEVICE)
    contrastive.load_state_dict(checkpoint["model_state_dict"], strict=True)
    contrastive.eval()
    for parameter in contrastive.parameters():
        parameter.requires_grad_(False)
    formula_adapter = load_formula_adapter(args.formula_adapter_weights)
    adapter_overlap = None
    adapter_formula_overlap = None
    adapter_formula_overlap_audit = "checkpoint_train_formulas"
    if formula_adapter is not None and args.formula_adapter_weights is not None:
        adapter_checkpoint = torch.load(
            args.formula_adapter_weights, map_location="cpu", weights_only=False
        )
        adapter_train_groups = set(map(str, adapter_checkpoint.get("train_groups", [])))
        current_validation_groups = set(group["cas"] for group in val_groups)
        adapter_overlap = sorted(adapter_train_groups & current_validation_groups)
        adapter_train_formulas = {
            str(value) for value in adapter_checkpoint.get("train_formulas", []) if str(value)
        }
        current_validation_formulas = {str(group["formula"]) for group in val_groups}
        adapter_unknown_train_groups: list[str] = []
        if not adapter_train_formulas:
            # 兼容旧 Adapter：仅在 CAS 属于当前池时回推公式；其余 CAS 无法审计，必须显式放宽。
            formula_by_cas = {str(group["cas"]): str(group["formula"]) for group in groups}
            adapter_unknown_train_groups = sorted(adapter_train_groups - set(formula_by_cas))
            adapter_train_formulas = {
                formula_by_cas[cas] for cas in adapter_train_groups if cas in formula_by_cas
            }
            adapter_formula_overlap_audit = "checkpoint_train_groups_backfilled_from_current_pool"
        adapter_formula_overlap = sorted(adapter_train_formulas & current_validation_formulas)
        if (adapter_overlap or adapter_formula_overlap or adapter_unknown_train_groups) and not args.allow_adapter_split_overlap:
            raise RuntimeError(
                "分子式适配器与当前验证集存在监督重叠: "
                f"CAS={len(adapter_overlap)}，formula={len(adapter_formula_overlap)}，"
                f"无法核验旧 CAS={len(adapter_unknown_train_groups)}。"
                "请用相同 --split-seed 重新训练适配器，"
                "或仅在诊断时显式加入 --allow-adapter-split-overlap。"
            )
    if formula_adapter is not None:
        print(
            f"复用分子式条件适配器: {args.formula_adapter_weights.resolve()}；"
            f"候选召回策略={args.shortlist_policy}",
            flush=True,
        )
    shortlist_policy = args.shortlist_policy
    if shortlist_policy == "auto":
        shortlist_policy = "formula_adapter" if formula_adapter is not None else "base"
    print(
        f"宽同分子式 Listwise 训练 train/val={len(train_groups)}/{len(val_groups)} "
        f"device={DEVICE} candidate_limit={args.candidate_limit} shortlist_policy={shortlist_policy}；"
        "不使用 PubChem rank/teacher",
        flush=True,
    )

    expected_cache_metadata = feature_cache_metadata(args, split_seed, shortlist_policy)
    cached_features = load_feature_cache(args, expected_cache_metadata)
    if cached_features is not None:
        prepared_train = list(cached_features["prepared_train"])
        prepared_val = list(cached_features["prepared_val"])
    else:
        embedding_cache = load_embedding_cache(args.graph_embedding_cache)
        if embedding_cache:
            print(f"复用图嵌入缓存: {len(embedding_cache)} 个候选", flush=True)
        prediction_map = load_prediction_map(args.prediction_cache)
        if args.candidate_feature_policy == "raw_cosine_forward":
            print(f"复用候选前向谱缓存: {len(prediction_map)} 个候选", flush=True)
        prepared_train = []
        prepared_val = []
        for position, group in enumerate(train_groups, start=1):
            prepared, _ = prepare_group(
                group,
                records,
                contrastive,
                formula_adapter,
                embedding_cache,
                args.candidate_limit,
                args.union_width,
                shortlist_policy,
                not args.no_target_protect_training,
                args.encode_batch_size,
                args.candidate_feature_policy,
                prediction_map,
            )
            prepared_train.append(prepared)
            if position == 1 or position % 25 == 0 or position == len(train_groups):
                print(
                    f"训练候选组 {position}/{len(train_groups)}，图缓存={len(embedding_cache)}",
                    flush=True,
                )
        for position, group in enumerate(val_groups, start=1):
            prepared, _ = prepare_group(
                group,
                records,
                contrastive,
                formula_adapter,
                embedding_cache,
                args.candidate_limit,
                args.union_width,
                shortlist_policy,
                False,
                args.encode_batch_size,
                args.candidate_feature_policy,
                prediction_map,
            )
            prepared_val.append(prepared)
            if position == 1 or position % 25 == 0 or position == len(val_groups):
                print(
                    f"验证候选组 {position}/{len(val_groups)}，图缓存={len(embedding_cache)}",
                    flush=True,
                )
        if args.feature_cache is not None:
            atomic_torch_save(
                {
                    "cache_metadata": expected_cache_metadata,
                    "chemical_domain": domain_metadata(),
                    "prepared_train": prepared_train,
                    "prepared_val": prepared_val,
                },
                args.feature_cache,
            )
            print(f"已写入 Listwise 特征缓存: {args.feature_cache.resolve()}", flush=True)

    # raw_cosine: formula cosine、base cosine、二者差值 + 五个结构特征，共 8 维。
    # rank_normalized 保留旧版三种集合内名次 + 五个结构特征，共 8 维。
    candidate_feature_dim = (
        FORWARD_FEATURE_DIM
        if args.candidate_feature_policy == "raw_cosine_forward"
        else FUSED_FEATURE_DIM
        if formula_adapter is not None
        else FEATURE_DIM
    )
    ranker_class = (
        CandidateIndependentFormulaRanker
        if args.ranker_architecture == "candidate_independent"
        else FormulaConditionedSetTransformerRanker
    )
    model = ranker_class(
        embedding_dim=int(checkpoint.get("embed_dim", EMBEDDING_DIM)),
        candidate_feature_dim=candidate_feature_dim,
        formula_dim=FORMULA_DIM,
        set_stats_dim=SET_STATS_DIM,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_key = (-1.0, -1.0, -1.0)
    best_epoch = -1
    best_validation = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(prepared_train)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for position in order:
            item = prepared_train[position]
            if item["positive_index"] < 0:
                # 目标未被粗召回时不能计算监督损失，保留为召回失败统计。
                continue
            permutation = torch.randperm(item["embeddings"].shape[0])
            embeddings = item["embeddings"][permutation]
            features = item["features"][permutation]
            coarse = item["coarse_scores"][permutation]
            positive = int((permutation == item["positive_index"]).nonzero(as_tuple=False)[0].item())
            scores, gate, residual = model(
                embeddings.to(DEVICE),
                features.to(DEVICE),
                item["query"].to(DEVICE),
                item["formula_features"].to(DEVICE),
                item["set_stats"].to(DEVICE),
                coarse.to(DEVICE),
            )
            loss = listwise_set_loss(
                scores,
                positive,
                coarse.to(DEVICE),
                residual,
                gate,
                hard_negative_k=args.hard_negative_k,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        validation = evaluate(model, prepared_val, DEVICE)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "validation": validation,
        }
        history.append(row)
        loss_text = "n/a" if row["train_loss"] is None else f"{row['train_loss']:.4f}"
        print(
            f"Epoch {epoch}/{args.epochs} loss={loss_text} "
            f"shortlist_recall={validation['shortlist_recall']:.3f} "
            f"ranker Top1/5/10/24={validation['ranker']['top1']:.3f}/"
            f"{validation['ranker']['top5']:.3f}/{validation['ranker']['top10']:.3f}/"
            f"{validation['ranker']['top24']:.3f} coarse={validation['coarse']['top1']:.3f}/"
            f"{validation['coarse']['top5']:.3f}/{validation['coarse']['top10']:.3f}/"
            f"{validation['coarse']['top24']:.3f}",
            flush=True,
        )
        key = (
            validation["ranker"]["top1"],
            validation["ranker"]["top5"],
            validation["ranker"]["top10"],
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_validation = evaluate(model, prepared_val, DEVICE, include_queries=True)
            args.output_weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "method": (
                        "formula_conditioned_wide_listwise_candidate_independent_no_pubchem_order"
                        if args.ranker_architecture == "candidate_independent"
                        else "formula_conditioned_wide_listwise_no_pubchem_order"
                    ),
                    "ranker_architecture": args.ranker_architecture,
                    "candidate_feature_policy": args.candidate_feature_policy,
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": int(checkpoint.get("embed_dim", EMBEDDING_DIM)),
                    "candidate_feature_dim": candidate_feature_dim,
                    "formula_dim": FORMULA_DIM,
                    "formula_elements": list(DOMAIN_FORMULA_ELEMENTS),
                    "chemical_domain": domain_metadata(),
                    "set_stats_dim": SET_STATS_DIM,
                    "candidate_limit": args.candidate_limit,
                    "seed": int(args.seed),
                    "split_seed": int(split_seed),
                    "contrastive_weights": str(args.contrastive_weights.resolve()),
                    "formula_adapter_weights": str(args.formula_adapter_weights.resolve()) if args.formula_adapter_weights else None,
                    "union_width": args.union_width,
                    "shortlist_policy": shortlist_policy,
                    "train_groups": [item["cas"] for item in train_groups],
                    "validation_groups": [item["cas"] for item in val_groups],
                    "best_validation": validation,
                },
                args.output_weights,
            )

    report = {
        "schema_version": 1,
        "method": (
            "formula_conditioned_wide_listwise_candidate_independent_no_pubchem_order"
            if args.ranker_architecture == "candidate_independent"
            else "formula_conditioned_wide_listwise_no_pubchem_order"
        ),
        "status": "completed",
        "wide_pool_cache": str(args.wide_pool_cache.resolve()),
        "contrastive_weights": str(args.contrastive_weights.resolve()),
        "formula_adapter_weights": str(args.formula_adapter_weights.resolve()) if args.formula_adapter_weights else None,
        "candidate_feature_policy": args.candidate_feature_policy,
        "formula_elements": list(DOMAIN_FORMULA_ELEMENTS),
        "chemical_domain": domain_metadata(),
        "groups": len(groups),
        "train_groups": len(train_groups),
        "validation_groups": len(val_groups),
        "candidate_limit": args.candidate_limit,
        "seed": int(args.seed),
        "ranker_architecture": args.ranker_architecture,
        "union_width": args.union_width,
        "shortlist_policy": shortlist_policy,
        "split_seed": int(split_seed),
        "adapter_validation_overlap_count": len(adapter_overlap or []),
        "adapter_formula_validation_overlap_count": len(adapter_formula_overlap or []),
        "adapter_formula_validation_overlap": adapter_formula_overlap or [],
        "adapter_formula_overlap_audit": adapter_formula_overlap_audit,
        "adapter_unknown_train_group_count": len(adapter_unknown_train_groups),
        "best_epoch": best_epoch,
        "target_protection_training": not args.no_target_protect_training,
        "feature_cache": str(args.feature_cache.resolve()) if args.feature_cache else None,
        "feature_cache_metadata": expected_cache_metadata,
        "best_validation": best_validation,
        "weights": str(args.output_weights.resolve()),
        "candidate_order_policy": "候选随机置换；不读取 PubChem CID/API rank、缓存位置或 teacher 分数",
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"权重: {args.output_weights.resolve()}\n报告: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()

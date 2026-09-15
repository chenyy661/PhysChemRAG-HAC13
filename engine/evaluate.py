"""评估无 PubChem 顺序依赖的学习型级联。

流程：分子式候选全集 -> 双塔余弦粗检索 -> Set Transformer/Listwise 精排 ->
前向光谱软验证特征 -> Top-24。候选截取采用规范 SMILES 哈希，仅为避免大于
上限时的输入位置偏置；PubChem API 返回名次从不参与排序。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physchemrag.config import DATA_DIR, LEVEL4_EXPERIMENTAL_DIR, OUTPUT_REPORTS_DIR, WAVE_LEN  # noqa: E402
from physchemrag.module2_crossmodal.formula_conditioned_retriever import (  # noqa: E402
    FORMULA_DIM,
    FormulaConditionedDualAdapter,
    formula_vector,
)
from physchemrag.module4_cascade.forward_spectral_features import compare_forward_spectra, fixed_forward_score  # noqa: E402
from physchemrag.module4_cascade.learned_cascade import (  # noqa: E402
    CandidateIndependentFormulaRanker,
    FormulaConditionedSetTransformerRanker,
    SetTransformerListwiseRanker,
    SpectrumGraphContrastiveModel,
)
from physchemrag.module4_cascade.spectral_views import prepare_masked_spectrum  # noqa: E402
from physchemrag.shared.chemical_domain import (  # noqa: E402
    FORMULA_ELEMENTS as DOMAIN_FORMULA_ELEMENTS,
    domain_metadata,
    domain_metadata_matches,
    inspect_smiles_domain,
)
from physchemrag.shared.molecular_formula import canonical_smiles, formula_from_smiles, normalize_formula  # noqa: E402
from physchemrag.shared.graph import graph_data  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help="统一外部评估数据集清单；提供后由清单解析 query-dir 与 candidate-cache",
    )
    parser.add_argument("--candidate-cache", type=Path, default=OUTPUT_REPORTS_DIR / "final_holdout_formula_only_union_candidates.json")
    parser.add_argument("--candidate-field", type=str, default="formula_only_results", choices=["formula_only_results", "results", "ir_formula_results", "candidates"])
    parser.add_argument("--query-dir", type=Path, default=LEVEL4_EXPERIMENTAL_DIR / "final_holdout")
    parser.add_argument("--contrastive-weights", type=Path, required=True)
    parser.add_argument(
        "--formula-adapter-weights",
        type=Path,
        default=None,
        help="可选：动态硬负样本训练的分子式条件双塔适配器",
    )
    parser.add_argument("--ranker-weights", type=Path, default=None)
    parser.add_argument("--prediction-cache", type=Path, default=None)
    parser.add_argument(
        "--full-pool-ranker",
        action="store_true",
        help="候选独立精排器在完整同分子式候选池上评分；集合型精排器禁止使用",
    )
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--coarse-topk", type=int, default=512)
    parser.add_argument(
        "--coarse-topk-policy",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="粗排 shortlist 规模策略；默认 fixed 保持历史结果，adaptive 按候选池规模分配",
    )
    parser.add_argument(
        "--adaptive-shortlist-fraction",
        type=float,
        default=0.25,
        help="adaptive 策略保留的候选比例",
    )
    parser.add_argument(
        "--adaptive-shortlist-min",
        type=int,
        default=256,
        help="adaptive 策略的 shortlist 下限",
    )
    parser.add_argument(
        "--adaptive-shortlist-max",
        type=int,
        default=5000,
        help="adaptive 策略的 shortlist 上限",
    )
    parser.add_argument(
        "--candidate-quality-policy",
        choices=("none", "neutral_single_component"),
        default="neutral_single_component",
        help="候选质量门控；默认 none，neutral_single_component 统一排除带电、自由基和多组分结构",
    )
    parser.add_argument(
        "--enforce-formula",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="重新由 canonical SMILES 计算分子式并严格匹配查询分子式",
    )
    parser.add_argument(
        "--use-valid-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将查询 payload 的 valid_mask 应用于模型输入；缺失时使用全有效掩码",
    )
    parser.add_argument("--output-topk", type=int, default=24, help="报告中保留多少个集合精排候选；物理软验证建议设置为 64")
    parser.add_argument(
        "--forward-ablation",
        action="store_true",
        help="在同一 Top-K 粗检索候选上报告双塔、SpecGNN 固定分数及二者 RRF 消融",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0, help="双塔与前向谱模型间 RRF 的平滑常数")
    parser.add_argument(
        "--final-score",
        choices=("auto", "coarse", "ranker", "coarse-forward-rrf"),
        default="auto",
        help="写入最终名次与候选列表的排序轨道；auto 在有 ranker 时选 ranker，否则选 coarse",
    )
    parser.add_argument("--graph-batch-size", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--shortlist-output",
        type=Path,
        default=None,
        help="可选：写出双塔 Top-K 候选，供生成完整 SpecGNN 前向缓存",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_manifest_path(value: object, manifest_path: Path, label: str) -> Path:
    """解析数据集清单路径，并要求目标已经存在。"""
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"数据集清单中的 {label} 不存在: {path}")
    return path


def apply_dataset_manifest(args: argparse.Namespace) -> dict | None:
    """用单一数据集清单绑定查询目录、候选缓存和默认输出。"""
    if args.dataset_manifest is None:
        if args.output is None:
            args.output = OUTPUT_REPORTS_DIR / "learned_cascade_evaluation.json"
        return None
    manifest_path = args.dataset_manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"数据集清单顶层必须为对象: {manifest_path}")
    if not domain_metadata_matches(manifest.get("chemical_domain")):
        raise ValueError("数据集清单缺少一致的 domain_hac13 元数据")
    if manifest.get("usage_policy") != "frozen_external_evaluation_only_not_for_training_or_model_selection":
        raise ValueError("数据集清单未锁定为冻结外部评估用途")
    args.query_dir = resolve_manifest_path(manifest.get("query_dir"), manifest_path, "query_dir")
    args.candidate_cache = resolve_manifest_path(
        manifest.get("candidate_cache"), manifest_path, "candidate_cache"
    )
    if args.output is None:
        default_output = manifest.get("default_evaluation_output")
        args.output = (
            Path(str(default_output)).resolve()
            if default_output
            else OUTPUT_REPORTS_DIR / f"learned_cascade_{manifest.get('dataset_name', 'combined')}.json"
        )
    args.dataset_manifest = manifest_path
    return manifest


def canonical(value: str) -> str:
    return canonical_smiles(str(value or ""), isomeric=False) or ""


def load_queries(directory: Path, domain_audit: dict[str, dict] | None = None) -> dict[str, dict]:
    records = {}
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        spectrum = torch.as_tensor(payload.get("spectrum"), dtype=torch.float32).flatten()
        valid_mask = torch.as_tensor(
            payload.get("valid_mask", torch.ones_like(spectrum, dtype=torch.bool)),
            dtype=torch.bool,
        ).flatten()
        smiles = str(payload.get("connectivity_smiles") or payload.get("smiles") or "")
        formula = normalize_formula(payload.get("molecular_formula") or formula_from_smiles(smiles))
        domain = inspect_smiles_domain(smiles, formula)
        cas = str(payload.get("cas", path.stem))
        if domain_audit is not None:
            domain_audit[cas] = {
                "cas": cas,
                "path": str(path.resolve()),
                **domain.as_dict(),
            }
        if not domain.valid:
            continue
        if (
            spectrum.numel() == WAVE_LEN
            and valid_mask.numel() == WAVE_LEN
            and torch.isfinite(spectrum).all()
            and bool(valid_mask.any())
        ):
            records[cas] = {
                "spectrum": spectrum,
                "valid_mask": valid_mask,
                "smiles": smiles,
                "formula": formula,
            }
    return records


def load_prediction_map(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    predictions = torch.as_tensor(payload.get("predictions"), dtype=torch.float32)
    if predictions.ndim != 2 or predictions.shape[1] != WAVE_LEN:
        raise ValueError(f"前向预测缓存形状异常: {tuple(predictions.shape)}")
    output = {}
    for key, index in dict(payload.get("index", {})).items():
        raw_key = str(key)
        normalized = canonical(raw_key)
        if raw_key:
            output[raw_key] = predictions[int(index)]
        if normalized:
            output[normalized] = predictions[int(index)]
    return output


def structure_features(smiles: str) -> list[float]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return [0.0] * 5
    heavy = molecule.GetNumHeavyAtoms()
    rotatable = sum(int(b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic() and not b.IsInRing()) for b in molecule.GetBonds())
    rings = molecule.GetRingInfo().NumRings()
    aromatic = sum(int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()) / max(heavy, 1)
    hetero = sum(int(atom.GetAtomicNum() not in (1, 6)) for atom in molecule.GetAtoms()) / max(heavy, 1)
    return [heavy / 50.0, rotatable / 20.0, rings / 10.0, aromatic, hetero]


def candidate_quality(smiles: str) -> tuple[bool, str]:
    """返回候选是否满足中性单组分质量门控及其拒绝原因。"""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False, "invalid_smiles"
    if len(Chem.GetMolFrags(molecule, asMols=False, sanitizeFrags=False)) != 1:
        return False, "multiple_components"
    if Chem.GetFormalCharge(molecule) != 0:
        return False, "formal_charge"
    if any(atom.GetNumRadicalElectrons() > 0 for atom in molecule.GetAtoms()):
        return False, "radical_electrons"
    return True, ""


def select_candidates(
    raw_items: list[dict],
    max_candidates: int,
    quality_policy: str = "none",
    molecular_formula: str = "",
    enforce_formula: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """规范化、去重并可选执行化学质量门控；超限时按结构哈希截取。"""
    seen = set()
    candidates = []
    rejected: dict[str, int] = {}
    for item in raw_items:
        smiles = str(item.get("smiles") or item.get("canonical_smiles") or "")
        key = canonical(smiles)
        if not key or key in seen:
            if not key:
                rejected["invalid_smiles"] = rejected.get("invalid_smiles", 0) + 1
            continue
        if quality_policy == "neutral_single_component":
            accepted, reason = candidate_quality(key)
            if not accepted:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
        if enforce_formula and molecular_formula:
            candidate_formula = normalize_formula(formula_from_smiles(key))
            if candidate_formula != normalize_formula(molecular_formula):
                rejected["formula_mismatch"] = rejected.get("formula_mismatch", 0) + 1
                continue
        domain = inspect_smiles_domain(
            key,
            molecular_formula if enforce_formula else None,
        )
        if not domain.valid:
            rejected[domain.reason] = rejected.get(domain.reason, 0) + 1
            continue
        seen.add(key)
        candidates.append({"smiles": smiles, "canonical": key})
    # 始终按规范结构哈希固定序列化顺序；该顺序不作为模型特征。
    candidates = sorted(candidates, key=lambda row: hashlib.sha256(row["canonical"].encode()).hexdigest())
    if max_candidates > 0 and len(candidates) > max_candidates:
        rejected["hash_cap"] = len(candidates) - max_candidates
        candidates = candidates[:max_candidates]
    return candidates, rejected


def resolve_shortlist_size(candidate_count: int, args: argparse.Namespace) -> int:
    """根据候选数量决定粗排 shortlist 大小，所有边界均显式截断。"""
    if candidate_count < 1:
        return 0
    if args.coarse_topk_policy == "fixed":
        return min(int(args.coarse_topk), candidate_count)
    fraction = min(max(float(args.adaptive_shortlist_fraction), 0.0), 1.0)
    lower = max(1, int(args.adaptive_shortlist_min))
    upper = max(lower, int(args.adaptive_shortlist_max))
    desired = max(lower, int(np.ceil(candidate_count * fraction)))
    return min(candidate_count, upper, max(1, desired))


def encode_graphs(model: SpectrumGraphContrastiveModel, candidates: list[dict], batch_size: int, device: torch.device) -> tuple[list[dict], torch.Tensor]:
    valid, vectors = [], []
    with torch.inference_mode():
        for start in range(0, len(candidates), max(1, batch_size)):
            chunk = candidates[start : start + max(1, batch_size)]
            data_list, kept = [], []
            for item in chunk:
                data = graph_data(item["smiles"])
                if data is not None:
                    data_list.append(data)
                    kept.append(item)
            if not data_list:
                continue
            batch = Batch.from_data_list(data_list).to(device, non_blocking=True)
            encoded = model.encode_graph(batch).cpu()
            vectors.append(encoded)
            valid.extend(kept)
    if not vectors:
        raise RuntimeError("没有候选能够完成分子图编码")
    return valid, torch.cat(vectors, dim=0)


def faiss_order(embeddings: torch.Tensor, query_embedding: torch.Tensor) -> torch.Tensor:
    """用 FAISS 对当前分子式候选做内积检索；不可用时退回 PyTorch 精确检索。"""
    if embeddings.ndim != 2 or query_embedding.ndim != 1:
        raise ValueError("FAISS 输入维度异常")
    try:
        import faiss  # 延迟导入，便于无 FAISS 环境执行 smoke

        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings.numpy().astype("float32", copy=False))
        _, indices = index.search(query_embedding.view(1, -1).numpy().astype("float32", copy=False), embeddings.shape[0])
        return torch.from_numpy(indices[0].astype("int64"))
    except (ImportError, RuntimeError, ValueError):
        return torch.argsort(embeddings @ query_embedding, descending=True)


def load_formula_adapter(
    path: Path | None,
    embedding_dim: int,
    device: torch.device,
) -> FormulaConditionedDualAdapter | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    formula_dim = int(checkpoint.get("formula_dim", -1))
    if formula_dim != FORMULA_DIM:
        raise ValueError(
            f"分子式适配器维度={formula_dim}，当前闭域要求 {FORMULA_DIM}；旧 13 维权重不可复用"
        )
    if tuple(checkpoint.get("formula_elements", ())) != tuple(DOMAIN_FORMULA_ELEMENTS):
        raise ValueError("分子式适配器元素词表与当前 10 维闭域定义不一致")
    if not domain_metadata_matches(checkpoint.get("chemical_domain")):
        raise ValueError("分子式适配器缺少一致的 domain_hac13 元数据")
    model = FormulaConditionedDualAdapter(
        embedding_dim=int(checkpoint.get("embedding_dim", embedding_dim)),
        formula_dim=formula_dim,
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        residual_scale=float(checkpoint.get("residual_scale", 0.15)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def apply_formula_adapter(
    adapter: FormulaConditionedDualAdapter,
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    formula: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对查询 [D] 和候选 [N,D] 应用同一分子式条件，返回归一化适配向量。"""
    if query_embedding.ndim != 1 or candidate_embeddings.ndim != 2:
        raise ValueError("分子式适配输入必须为查询 [D] 和候选 [N,D]")
    features = formula_vector(formula).to(device, non_blocking=True)
    query = adapter.encode_query(
        query_embedding.to(device, non_blocking=True).unsqueeze(0),
        features.unsqueeze(0),
    ).squeeze(0)
    chunks = []
    for start in range(0, candidate_embeddings.shape[0], batch_size):
        # candidates: [N_i,D]；formula_batch: [N_i,F]
        candidates = candidate_embeddings[start : start + batch_size].to(
            device, non_blocking=True
        )
        formula_batch = features.unsqueeze(0).expand(candidates.shape[0], -1)
        chunks.append(adapter.encode_molecule(candidates, formula_batch).cpu())
    return query.cpu(), torch.cat(chunks, dim=0)


def build_features(valid: list[dict], coarse: torch.Tensor, target_spectrum: torch.Tensor, prediction_map: dict[str, torch.Tensor]) -> torch.Tensor:
    target = target_spectrum.view(1, WAVE_LEN)
    forward_rows = []
    for item in valid:
        predicted = prediction_map.get(item["canonical"])
        if predicted is None:
            forward_rows.append(torch.zeros(8))
        else:
            forward_rows.append(compare_forward_spectra(predicted.view(1, WAVE_LEN), target).squeeze(0).cpu())
    forward = torch.stack(forward_rows).float()
    structure = torch.tensor([structure_features(item["smiles"]) for item in valid], dtype=torch.float32)
    features = torch.cat([coarse.cpu().unsqueeze(1), forward, structure], dim=1)
    if features.ndim != 2 or features.shape[1] != 14:
        raise ValueError(f"候选特征形状异常: {tuple(features.shape)}")
    return features


def stats(values: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(values / 0.20, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum() / max(float(np.log(max(values.numel(), 2))), 1.0)
    top = torch.topk(values, min(5, values.numel())).values
    gap = top[0] - top[1] if top.numel() > 1 else values.new_zeros(())
    count_value = values.new_tensor(values.numel() / 512.0)
    return torch.stack([count_value, values.mean(), values.std(unbiased=False), gap, entropy]).float()


def rank_normalized(values: torch.Tensor) -> torch.Tensor:
    """将候选分数转换为集合内的归一化名次。"""
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return 1.0 - ranks / max(float(values.numel() - 1), 1.0)


def metric(ranks: list[int | None], total: int) -> dict[str, float | int | None]:
    present = np.asarray([int(rank) for rank in ranks if rank is not None], dtype=np.int64)
    return {
        "query_count": total,
        "input_recall": float(len(present) / max(total, 1)),
        "top1": float(np.sum(present <= 1) / max(total, 1)),
        "top5": float(np.sum(present <= 5) / max(total, 1)),
        "top10": float(np.sum(present <= 10) / max(total, 1)),
        "top24": float(np.sum(present <= 24) / max(total, 1)),
        "mrr": float(np.sum(1.0 / present) / max(total, 1)) if len(present) else 0.0,
        "median_rank_when_present": float(np.median(present)) if len(present) else None,
    }


def main() -> None:
    args = parse_args()
    dataset_manifest = apply_dataset_manifest(args)
    if args.smoke:
        args.query_count = args.query_count or 3
        args.max_candidates = min(args.max_candidates, 64)
        args.coarse_topk = min(args.coarse_topk, 16)
        args.graph_batch_size = min(args.graph_batch_size, 32)
    if args.forward_ablation and args.prediction_cache is None:
        raise ValueError("--forward-ablation 需要 --prediction-cache")
    if args.final_score == "coarse-forward-rrf" and not args.forward_ablation:
        raise ValueError("--final-score coarse-forward-rrf 需要 --forward-ablation")
    if args.rrf_k <= 0.0:
        raise ValueError("--rrf-k 必须大于 0")
    started = time.perf_counter()
    with args.candidate_cache.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    all_source_queries = list(payload.get("queries", payload.get("groups", [])))
    if dataset_manifest is not None:
        expected_cas = [str(row.get("cas", "")) for row in dataset_manifest.get("rows", [])]
        actual_cas = [str(query.get("cas", "")) for query in all_source_queries]
        if len(expected_cas) != int(dataset_manifest.get("query_count", -1)):
            raise ValueError("数据集清单的 query_count 与 rows 数量不一致")
        if len(expected_cas) != len(set(expected_cas)):
            raise ValueError("数据集清单包含重复 CAS")
        if len(actual_cas) != len(set(actual_cas)):
            raise ValueError("统一候选缓存包含重复 CAS")
        if set(actual_cas) != set(expected_cas):
            raise ValueError("统一候选缓存与数据集清单的 CAS 集合不一致")
    query_domain_audit: dict[str, dict] = {}
    queries = load_queries(args.query_dir, query_domain_audit)
    if dataset_manifest is not None:
        expected_cas_set = {str(row.get("cas", "")) for row in dataset_manifest.get("rows", [])}
        missing_queries = sorted(expected_cas_set - set(queries))
        unexpected_queries = sorted(set(queries) - expected_cas_set)
        if missing_queries or unexpected_queries:
            raise ValueError(
                "统一查询目录与数据集清单不一致: "
                f"missing={missing_queries[:5]}, unexpected={unexpected_queries[:5]}"
            )
    prediction_map = load_prediction_map(args.prediction_cache)
    contrastive_checkpoint = torch.load(args.contrastive_weights, map_location=DEVICE, weights_only=False)
    if not domain_metadata_matches(contrastive_checkpoint.get("chemical_domain")):
        raise ValueError("双塔 checkpoint 缺少一致的 domain_hac13 元数据")
    contrastive = SpectrumGraphContrastiveModel(
        dare_weights=str(contrastive_checkpoint.get("dare_weights")),
        node_in_dim=int(contrastive_checkpoint.get("node_in_dim", 9)),
        edge_in_dim=int(contrastive_checkpoint.get("edge_in_dim", 3)),
        embed_dim=int(contrastive_checkpoint.get("embed_dim", 512)),
    ).to(DEVICE)
    contrastive.load_state_dict(contrastive_checkpoint["model_state_dict"], strict=True)
    contrastive.eval()
    formula_adapter = load_formula_adapter(
        args.formula_adapter_weights, contrastive.embed_dim, DEVICE
    )
    ranker = None
    formula_conditioned_ranker = False
    candidate_independent_ranker = False
    candidate_feature_policy = "rank_normalized"
    if args.ranker_weights is not None:
        ranker_checkpoint = torch.load(args.ranker_weights, map_location=DEVICE, weights_only=False)
        if not domain_metadata_matches(ranker_checkpoint.get("chemical_domain")):
            raise ValueError("精排 checkpoint 缺少一致的 domain_hac13 元数据")
        formula_conditioned_ranker = str(ranker_checkpoint.get("method", "")).startswith(
            "formula_conditioned_wide_listwise"
        ) or "formula_dim" in ranker_checkpoint
        if formula_conditioned_ranker:
            # 旧 checkpoint 没有该字段，按其历史名次特征回退；新 candidate-independent
            # 权重默认使用 raw_cosine，避免候选池组成改变单候选分数。
            candidate_feature_policy = str(
                ranker_checkpoint.get("candidate_feature_policy", "rank_normalized")
            )
            if candidate_feature_policy not in {"raw_cosine", "rank_normalized"}:
                raise ValueError(
                    f"精排 checkpoint 的 candidate_feature_policy 无效: {candidate_feature_policy}"
                )
        if formula_conditioned_ranker:
            if formula_adapter is None:
                raise ValueError("新的分子式条件精排器需要 --formula-adapter-weights")
            if int(ranker_checkpoint.get("formula_dim", -1)) != FORMULA_DIM:
                raise ValueError("分子式条件精排器维度与当前 10 维闭域定义不一致")
            if tuple(ranker_checkpoint.get("formula_elements", ())) != tuple(DOMAIN_FORMULA_ELEMENTS):
                raise ValueError("分子式条件精排器元素词表与当前 10 维闭域定义不一致")
            ranker_class = (
                CandidateIndependentFormulaRanker
                if str(ranker_checkpoint.get("ranker_architecture", "")) == "candidate_independent"
                or "candidate_independent" in str(ranker_checkpoint.get("method", ""))
                else FormulaConditionedSetTransformerRanker
            )
            candidate_independent_ranker = ranker_class is CandidateIndependentFormulaRanker
            ranker = ranker_class(
                embedding_dim=int(ranker_checkpoint.get("embedding_dim", contrastive.embed_dim)),
                candidate_feature_dim=int(ranker_checkpoint.get("candidate_feature_dim", 8)),
                formula_dim=int(ranker_checkpoint.get("formula_dim", FORMULA_DIM)),
                set_stats_dim=int(ranker_checkpoint.get("set_stats_dim", 5)),
            ).to(DEVICE)
        else:
            if args.formula_adapter_weights is not None:
                raise ValueError("旧版 Set Transformer 不能与分子式适配器同时使用")
            ranker = SetTransformerListwiseRanker(
                embedding_dim=int(ranker_checkpoint.get("embedding_dim", contrastive.embed_dim)),
                candidate_feature_dim=int(ranker_checkpoint.get("candidate_feature_dim", 14)),
                set_stats_dim=int(ranker_checkpoint.get("set_stats_dim", 5)),
            ).to(DEVICE)
        ranker.load_state_dict(ranker_checkpoint["model_state_dict"], strict=True)
        ranker.eval()
    if args.full_pool_ranker and ranker is None:
        raise ValueError("--full-pool-ranker 需要 --ranker-weights")
    if args.full_pool_ranker and not candidate_independent_ranker:
        raise ValueError("--full-pool-ranker 只适用于候选独立精排器；Set Transformer 依赖集合上下文")
    if formula_conditioned_ranker and args.forward_ablation:
        raise ValueError("新的分子式条件精排器不支持旧版前向谱消融特征")
    final_score_method = args.final_score
    if final_score_method == "auto":
        final_score_method = "ranker" if ranker is not None else "coarse"
    if final_score_method == "ranker" and ranker is None:
        raise ValueError("--final-score ranker 需要 --ranker-weights")
    source_queries = all_source_queries
    source_queries = source_queries[args.query_offset : args.query_offset + args.query_count] if args.query_count > 0 else source_queries[args.query_offset :]
    raw_source_query_count = len(source_queries)
    domain_exclusions = [
        query_domain_audit[str(query.get("cas", ""))]
        for query in source_queries
        if str(query.get("cas", "")) in query_domain_audit
        and not query_domain_audit[str(query.get("cas", ""))]["valid"]
    ]
    source_queries = [
        query
        for query in source_queries
        if str(query.get("cas", "")) not in query_domain_audit
        or query_domain_audit[str(query.get("cas", ""))]["valid"]
    ]
    if not source_queries:
        raise RuntimeError("候选缓存中没有待评估查询")
    coarse_ranks, final_ranks, rows = [], [], []
    shortlist_groups = []
    shortlist_coarse_ranks: list[int | None] = []
    forward_ranks: list[int | None] = []
    rrf_ranks: list[int | None] = []
    target_prediction_count = 0
    predicted_shortlist_candidates = 0
    total_shortlist_candidates = 0
    total_quality_rejections: dict[str, int] = {}
    target_removed_by_selection = 0
    target_present_after_selection = 0
    target_graph_encoding_failed = 0
    valid_mask_coverages: list[float] = []
    for query_index, query in enumerate(source_queries):
        cas = str(query.get("cas", ""))
        record = queries.get(cas)
        if record is None:
            coarse_ranks.append(None); final_ranks.append(None)
            if args.forward_ablation:
                shortlist_coarse_ranks.append(None); forward_ranks.append(None); rrf_ranks.append(None)
            continue
        target = canonical(record["smiles"] or query.get("smiles", ""))
        raw_items = list(query.get(args.candidate_field, []))
        if args.candidate_field == "candidates" and not raw_items:
            raw_items = list(query.get("results", []))
        candidates, candidate_rejections = select_candidates(
            raw_items,
            args.max_candidates,
            quality_policy=args.candidate_quality_policy,
            molecular_formula=str(query.get("molecular_formula") or record["formula"]),
            enforce_formula=bool(args.enforce_formula),
        )
        for reason, count in candidate_rejections.items():
            total_quality_rejections[reason] = total_quality_rejections.get(reason, 0) + int(count)
        target_in_selected = any(item["canonical"] == target for item in candidates)
        if target_in_selected:
            target_present_after_selection += 1
        elif any(canonical(str(item.get("smiles") or item.get("canonical_smiles") or "")) == target for item in raw_items):
            target_removed_by_selection += 1
        if len(candidates) < 1:
            coarse_ranks.append(None); final_ranks.append(None)
            if args.forward_ablation:
                shortlist_coarse_ranks.append(None); forward_ranks.append(None); rrf_ranks.append(None)
            continue
        valid, embeddings = encode_graphs(contrastive, candidates, args.graph_batch_size, DEVICE)
        target_in_input = any(item["canonical"] == target for item in valid)
        if not target_in_input and target_in_selected:
            target_graph_encoding_failed += 1
        model_mask = record["valid_mask"] if args.use_valid_mask else None
        spectrum, effective_mask = prepare_masked_spectrum(
            record["spectrum"],
            model_mask,
        )
        spectrum = spectrum.to(DEVICE, non_blocking=True)
        # spectrum: [B,1,1800]；启用掩码时，缺测区由相邻有效点线性填补。
        raw_mask_fraction = float(record["valid_mask"].float().mean().item())
        effective_mask_fraction = float(effective_mask.float().mean().item())
        valid_mask_coverages.append(raw_mask_fraction)
        with torch.inference_mode():
            query_embedding = contrastive.encode_spectrum(spectrum).squeeze(0).cpu()
        base_query_embedding = query_embedding
        base_embeddings = embeddings
        base_scores = base_embeddings @ base_query_embedding.unsqueeze(1)
        base_scores = base_scores.squeeze(1)
        formula = normalize_formula(
            query.get("molecular_formula") or record["formula"] or formula_from_smiles(record["smiles"])
        )
        if formula_adapter is not None:
            query_embedding, embeddings = apply_formula_adapter(
                formula_adapter,
                query_embedding,
                embeddings,
                formula,
                args.graph_batch_size,
                DEVICE,
            )
        coarse_scores = embeddings @ query_embedding.unsqueeze(1)
        coarse_scores = coarse_scores.squeeze(1)
        # FAISS 仅用于学习到的图/光谱向量粗检索，不读取 PubChem 返回名次。
        coarse_order = faiss_order(embeddings, query_embedding)
        coarse_rank = next((position for position, index in enumerate(coarse_order.tolist(), start=1) if valid[index]["canonical"] == target), None)
        shortlist_size = len(valid) if args.full_pool_ranker else resolve_shortlist_size(len(valid), args)
        shortlist_order = coarse_order[:shortlist_size]
        shortlist = [valid[index] for index in shortlist_order.tolist()]
        shortlist_embeddings = embeddings[shortlist_order]
        shortlist_coarse = coarse_scores[shortlist_order]
        if formula_conditioned_ranker:
            formula_features = formula_vector(formula).float()
            base_rank = rank_normalized(base_scores)[shortlist_order]
            formula_rank = rank_normalized(coarse_scores)[shortlist_order]
            structure = torch.tensor(
                [structure_features(item["smiles"]) for item in shortlist],
                dtype=torch.float32,
            )
            if candidate_feature_policy == "raw_cosine":
                # [N,8] = formula cosine、base cosine、二者差值、5 个结构特征。
                base_cosine = base_scores[shortlist_order]
                formula_cosine = coarse_scores[shortlist_order]
                shortlist_coarse = formula_cosine
                features = torch.cat(
                    [
                        formula_cosine.unsqueeze(1),
                        base_cosine.unsqueeze(1),
                        (formula_cosine - base_cosine).unsqueeze(1),
                        structure,
                    ],
                    dim=1,
                )
            else:
                # 旧权重兼容：三列集合内名次 + 五个结构特征。
                shortlist_coarse = formula_rank
                features = torch.cat(
                    [shortlist_coarse.unsqueeze(1), base_rank.unsqueeze(1), formula_rank.unsqueeze(1), structure],
                    dim=1,
                )
        else:
            formula_features = None
            features = build_features(shortlist, shortlist_coarse, spectrum.cpu(), prediction_map)
        if args.shortlist_output is not None:
            shortlist_groups.append(
                {
                    "cas": cas,
                    "formula": formula,
                    "candidates": [
                        {
                            "smiles": item["smiles"],
                            "canonical": item["canonical"],
                            "coarse_score": float(shortlist_coarse[index]),
                        }
                        for index, item in enumerate(shortlist)
                    ],
                }
            )
        if ranker is None:
            final_scores = shortlist_coarse
            gate_value = 0.0
        else:
            with torch.inference_mode():
                if formula_conditioned_ranker:
                    final_scores, gate, _ = ranker(
                        shortlist_embeddings.to(DEVICE),
                        features.to(DEVICE),
                        query_embedding.to(DEVICE),
                        formula_features.to(DEVICE),
                        stats(shortlist_coarse).to(DEVICE),
                        shortlist_coarse.to(DEVICE),
                    )
                else:
                    final_scores, gate, _ = ranker(
                        shortlist_embeddings.to(DEVICE),
                        features.to(DEVICE),
                        query_embedding.to(DEVICE),
                        stats(shortlist_coarse).to(DEVICE),
                        shortlist_coarse.to(DEVICE),
                    )
            final_scores = final_scores.cpu()
            gate_value = float(gate.cpu())
        final_order = torch.argsort(final_scores, descending=True).tolist()
        final_rank = next((position for position, index in enumerate(final_order, start=1) if shortlist[index]["canonical"] == target), None)
        ablation_row = None
        if args.forward_ablation:
            target_shortlist_index = next(
                (index for index, item in enumerate(shortlist) if item["canonical"] == target),
                None,
            )
            shortlist_coarse_rank = None if target_shortlist_index is None else target_shortlist_index + 1
            shortlist_coarse_ranks.append(shortlist_coarse_rank)

            # forward_features: [N,8]；缺失预测不产生伪分数，统一放在前向排序尾部。
            forward_features = features[:, 1:9]
            forward_scores = fixed_forward_score(forward_features)
            available = torch.tensor(
                [item["canonical"] in prediction_map for item in shortlist],
                dtype=torch.bool,
            )
            if final_score_method == "coarse-forward-rrf" and not bool(available.all()):
                missing_count = int((~available).sum().item())
                raise RuntimeError(
                    f"最终 RRF 禁止缺失前向预测: {cas} 缺少 {missing_count}/{available.numel()}"
                )
            predicted_shortlist_candidates += int(available.sum().item())
            total_shortlist_candidates += int(available.numel())
            if target_shortlist_index is not None and bool(available[target_shortlist_index]):
                target_prediction_count += 1

            coarse_positions = torch.empty(len(shortlist), dtype=torch.long)
            coarse_positions[torch.arange(len(shortlist))] = torch.arange(1, len(shortlist) + 1)
            forward_order = sorted(
                range(len(shortlist)),
                key=lambda index: (
                    not bool(available[index]),
                    -float(forward_scores[index]) if bool(available[index]) else 0.0,
                    int(coarse_positions[index]),
                ),
            )
            forward_positions = torch.empty(len(shortlist), dtype=torch.long)
            for position, candidate_index in enumerate(forward_order, start=1):
                forward_positions[candidate_index] = position
            # 两个名次都来自可解释模型输出；不包含 PubChem/CID/API 顺序。
            rrf_scores = (
                1.0 / (float(args.rrf_k) + coarse_positions.float())
                + 1.0 / (float(args.rrf_k) + forward_positions.float())
            )
            rrf_order = torch.argsort(rrf_scores, descending=True).tolist()
            forward_rank = (
                None
                if target_shortlist_index is None or not bool(available[target_shortlist_index])
                else int(forward_positions[target_shortlist_index])
            )
            rrf_rank = (
                None
                if target_shortlist_index is None
                else rrf_order.index(target_shortlist_index) + 1
            )
            forward_ranks.append(forward_rank)
            rrf_ranks.append(rrf_rank)
            ablation_row = {
                "shortlist_coarse": shortlist_coarse_rank,
                "forward_fixed": forward_rank,
                "coarse_forward_rrf": rrf_rank,
                "target_forward_prediction_available": bool(
                    target_shortlist_index is not None and available[target_shortlist_index]
                ),
            }
            if final_score_method == "coarse-forward-rrf":
                final_scores = rrf_scores
                final_order = rrf_order
                final_rank = rrf_rank
        if final_score_method == "coarse":
            final_scores = shortlist_coarse
            final_order = list(range(len(shortlist)))
            final_rank = next(
                (
                    position
                    for position, item in enumerate(shortlist, start=1)
                    if item["canonical"] == target
                ),
                None,
            )
        coarse_ranks.append(coarse_rank if target_in_input else None)
        final_ranks.append(final_rank if final_rank is not None else None)
        output_topk = max(1, min(int(args.output_topk), len(final_order)))
        top_rows = []
        for position in final_order[:output_topk]:
            item = dict(shortlist[position])
            item["coarse_score"] = float(shortlist_coarse[position])
            item["final_score"] = float(final_scores[position])
            forward_available = item["canonical"] in prediction_map
            item["forward_prediction_available"] = forward_available
            item["forward_fixed_score"] = (
                float(fixed_forward_score(features[position : position + 1, 1:9]).item())
                if forward_available
                else None
            )
            top_rows.append(item)
        rows.append({
            "cas": cas,
            "target": target,
            "formula": formula,
            "candidate_field": args.candidate_field,
            "candidate_count_after_hash_cap": len(candidates),
            "raw_candidate_count": len(raw_items),
            "candidate_quality_policy": args.candidate_quality_policy,
            "enforce_formula": bool(args.enforce_formula),
            "candidate_rejections": candidate_rejections,
            "graph_valid_count": len(valid),
            "coarse_shortlist_count": len(shortlist),
            "full_pool_ranker": bool(args.full_pool_ranker and candidate_independent_ranker),
            "raw_valid_mask_fraction": raw_mask_fraction,
            "effective_valid_mask_fraction": effective_mask_fraction,
            "use_valid_mask": bool(args.use_valid_mask),
            "output_topk": output_topk,
            "target_in_input": target_in_input,
            "ranks": {
                "coarse_full": coarse_rank if target_in_input else None,
                "final_shortlist": final_rank,
                **({"ablation": ablation_row} if ablation_row is not None else {}),
            },
            "gate": gate_value,
            "top24": top_rows,
        })
        if (
            query_index == 0
            or (query_index + 1) % max(args.log_interval, 1) == 0
            or query_index + 1 == len(source_queries)
        ):
            print(
                f"[{query_index + 1}/{len(source_queries)}] {cas} candidates={len(valid)} "
                f"coarse_rank={coarse_rank} final_rank={final_rank}",
                flush=True,
            )

    result = {
        "schema_version": 2,
        "method": "learned_cascade_no_pubchem_rank",
        "dataset_manifest": (
            str(args.dataset_manifest.resolve()) if args.dataset_manifest is not None else None
        ),
        "dataset_name": dataset_manifest.get("dataset_name") if dataset_manifest else None,
        "dataset_usage_policy": dataset_manifest.get("usage_policy") if dataset_manifest else None,
        "source_datasets": dataset_manifest.get("source_datasets") if dataset_manifest else None,
        "candidate_order_policy": "PubChem 只用于候选生成；候选输入位置和 formula_cid_rank 不参与模型或分数；超限时按规范结构哈希截取",
        "candidate_cache": str(args.candidate_cache.resolve()),
        "chemical_domain": domain_metadata(),
        "raw_query_count": raw_source_query_count,
        "eligible_query_count": len(source_queries),
        "domain_excluded_count": len(domain_exclusions),
        "domain_exclusions": domain_exclusions,
        "candidate_field": args.candidate_field,
        "contrastive_weights": str(args.contrastive_weights.resolve()),
        "formula_adapter_weights": (
            str(args.formula_adapter_weights.resolve())
            if args.formula_adapter_weights
            else None
        ),
        "ranker_weights": str(args.ranker_weights.resolve()) if args.ranker_weights else None,
        "candidate_feature_policy": candidate_feature_policy,
        "prediction_cache": str(args.prediction_cache.resolve()) if args.prediction_cache else None,
        "max_candidates": args.max_candidates,
        "coarse_topk": args.coarse_topk,
        "full_pool_ranker": bool(args.full_pool_ranker),
        "coarse_topk_policy": args.coarse_topk_policy,
        "adaptive_shortlist_fraction": args.adaptive_shortlist_fraction,
        "adaptive_shortlist_min": args.adaptive_shortlist_min,
        "adaptive_shortlist_max": args.adaptive_shortlist_max,
        "candidate_quality_policy": args.candidate_quality_policy,
        "enforce_formula": bool(args.enforce_formula),
        "use_valid_mask": bool(args.use_valid_mask),
        "candidate_rejections_total": total_quality_rejections,
        "target_present_after_selection": target_present_after_selection,
        "target_removed_by_selection": target_removed_by_selection,
        "target_graph_encoding_failed": target_graph_encoding_failed,
        "raw_valid_mask_fraction_mean": float(np.mean(valid_mask_coverages)) if valid_mask_coverages else None,
        "model_mask_policy": (
            "使用 valid_mask，并对缺测区按相邻有效点填补"
            if args.use_valid_mask
            else "忽略 valid_mask，全部波数点直接输入"
        ),
        "final_score_method": final_score_method,
        "summary": {
            "coarse": metric(coarse_ranks, len(source_queries)),
            "final": metric(final_ranks, len(source_queries)),
            **(
                {
                    "ablation": {
                        "shortlist_coarse": metric(shortlist_coarse_ranks, len(source_queries)),
                        "forward_fixed": metric(forward_ranks, len(source_queries)),
                        "coarse_forward_rrf": metric(rrf_ranks, len(source_queries)),
                        "rrf_k": float(args.rrf_k),
                        "target_forward_prediction_recall": float(
                            target_prediction_count / max(len(source_queries), 1)
                        ),
                        "shortlist_candidate_prediction_coverage": float(
                            predicted_shortlist_candidates / max(total_shortlist_candidates, 1)
                        ),
                    }
                }
                if args.forward_ablation
                else {}
            ),
            "evaluated_rows": len(rows),
        },
        "queries": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.shortlist_output is not None:
        args.shortlist_output.parent.mkdir(parents=True, exist_ok=True)
        args.shortlist_output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "method": "spectrum_graph_formula_pool_topk_no_pubchem_rank",
                    "source_candidate_cache": str(args.candidate_cache.resolve()),
                    "contrastive_weights": str(args.contrastive_weights.resolve()),
                    "formula_adapter_weights": (
                        str(args.formula_adapter_weights.resolve())
                        if args.formula_adapter_weights
                        else None
                    ),
                    "candidate_order_policy": "候选由学习到的双塔分数选择；不读取 PubChem/CID/API 顺序",
                    "groups": shortlist_groups,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"双塔 shortlist: {args.shortlist_output.resolve()}", flush=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"报告: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

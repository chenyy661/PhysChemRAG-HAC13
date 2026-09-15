"""在闭域模拟数据上预训练同分子式候选级 Formula Adapter。

脚本只使用模拟索引内的结构，不查询或展开 PubChem 候选池。每条模拟光谱
以自身分子图为正样本，负样本在训练时仅从相同规范分子式的其他 canonical
结构中在线采样。训练和验证按完整分子式组隔离，避免结构或分子式泄漏。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physchemrag.config import DATA_DIR, OUTPUT_REPORTS_DIR, WEIGHTS_DIR  # noqa: E402
from physchemrag.module2_crossmodal.formula_conditioned_retriever import (  # noqa: E402
    FORMULA_DIM,
    FORMULA_ELEMENTS as ADAPTER_FORMULA_ELEMENTS,
    FormulaConditionedDualAdapter,
    formula_vector,
)
from physchemrag.module4_cascade.learned_cascade import (  # noqa: E402
    SpectrumGraphContrastiveModel,
)
from physchemrag.shared.chemical_domain import (  # noqa: E402
    DOMAIN_NAME,
    FORMULA_ELEMENTS,
    domain_metadata,
    domain_metadata_matches,
    inspect_smiles_domain,
)
from physchemrag.shared.dataset import PhysChemRADataset  # noqa: E402


@dataclass(frozen=True)
class SimulationRecord:
    """参与候选级预训练的一条唯一连接结构记录。"""

    dataset_idx: int
    formula: str
    canonical: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--structure-index",
        type=Path,
        default=DATA_DIR
        / "conformers_v2"
        / "training_index_decontaminated_all_external_domain_hac13.jsonl",
        help="去污染且经过 domain_hac13 过滤的模拟结构索引",
    )
    parser.add_argument(
        "--contrastive-weights",
        type=Path,
        default=WEIGHTS_DIR / "module4_spectrum_graph_contrastive_domain_hac13_best.pth",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=OUTPUT_REPORTS_DIR
        / "simulated_formula_adapter_base_embeddings_domain_hac13.pt",
        help="冻结双塔的模拟光谱/自身图嵌入缓存",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=WEIGHTS_DIR / "module4_formula_adapter_simulated_domain_hac13_best.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_REPORTS_DIR
        / "module4_formula_adapter_simulated_domain_hac13_training.json",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--candidates-per-query",
        "--max-candidates-per-query",
        dest="candidates_per_query",
        type=int,
        default=64,
        help="训练时每条查询的候选上限，含一个自身正样本",
    )
    parser.add_argument("--hard-negative-k", type=int, default=16)
    parser.add_argument(
        "--base-hard-negative-k",
        type=int,
        default=32,
        help="从完整同分子式池按冻结双塔余弦选取的困难负样本数；其余候选由随机同式负样本补齐",
    )
    parser.add_argument(
        "--negative-sampling",
        choices=("base_topk_mixed", "random"),
        default="base_topk_mixed",
        help="同式负样本策略；base_topk_mixed 使用完整池双塔 hard negative + 随机负样本",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=0.20)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--split-seed", type=int, default=20260913)
    parser.add_argument(
        "--max-index-records",
        type=int,
        default=0,
        help="最多扫描多少条索引记录；0 表示全部，仅用于诊断",
    )
    parser.add_argument(
        "--max-formulas",
        type=int,
        default=0,
        help="最多保留多少个可训练分子式组；0 表示全部",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--rebuild-embedding-cache", action="store_true")
    parser.add_argument(
        "--allow-legacy-contrastive",
        action="store_true",
        help="允许缺少闭域元数据的旧双塔 checkpoint；仅用于迁移诊断",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    positive = (
        "epochs",
        "batch_size",
        "embedding_batch_size",
        "candidates_per_query",
        "hard_negative_k",
        "hidden_dim",
        "evaluation_batch_size",
        "patience",
        "log_interval",
    )
    if any(int(getattr(args, name)) < 1 for name in positive):
        parser.error("训练、批次、候选、维度、patience 和日志间隔参数必须为正")
    if args.candidates_per_query < 2:
        parser.error("candidates-per-query 至少为 2")
    if args.base_hard_negative_k < 0:
        parser.error("base-hard-negative-k 不能为负")
    if not 0.05 <= args.val_fraction < 0.5:
        parser.error("val-fraction 必须位于 [0.05, 0.5)")
    if min(args.max_index_records, args.max_formulas, args.max_train_samples, args.max_val_samples) < 0:
        parser.error("所有 max-* 限制必须为非负整数")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate 必须为正，weight-decay 不能为负")
    return args


def smoke_path(path: Path) -> Path:
    """为 smoke 产物增加独立后缀，避免覆盖正式缓存和权重。"""
    return path.with_name(f"{path.stem}_smoke{path.suffix}")


def configure_smoke(args: argparse.Namespace) -> None:
    """将 smoke 限制为足以覆盖完整数据通路的短任务。"""
    if not args.smoke:
        return
    args.epochs = min(args.epochs, 1)
    args.batch_size = min(args.batch_size, 16)
    args.embedding_batch_size = min(args.embedding_batch_size, 16)
    args.candidates_per_query = min(args.candidates_per_query, 8)
    args.hard_negative_k = min(args.hard_negative_k, 4)
    args.base_hard_negative_k = min(args.base_hard_negative_k, args.candidates_per_query - 1)
    args.evaluation_batch_size = min(args.evaluation_batch_size, 32)
    args.max_index_records = args.max_index_records or 10_000
    args.max_formulas = args.max_formulas or 32
    args.max_train_samples = args.max_train_samples or 96
    args.max_val_samples = args.max_val_samples or 64
    args.log_interval = 1
    args.embedding_cache = smoke_path(args.embedding_cache)
    args.output_weights = smoke_path(args.output_weights)
    args.report = smoke_path(args.report)


def seed_everything(seed: int) -> None:
    """固定所有本地随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def sha256_file(path: Path) -> str:
    """流式计算文件哈希，供嵌入缓存一致性校验。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records_signature(records: Iterable[SimulationRecord]) -> str:
    """计算有序结构记录签名。"""
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record.dataset_idx}\t{record.formula}\t{record.canonical}\n".encode("utf-8")
        )
    return digest.hexdigest()


def read_structure_index(
    path: Path,
    dataset_length: int,
    max_records: int,
    max_formulas: int,
    seed: int,
) -> tuple[list[SimulationRecord], dict[str, Any], dict[str, list[int]]]:
    """读取索引、再次执行闭域检查，并建立严格同分子式候选组。"""
    if not path.is_file():
        raise FileNotFoundError(f"模拟结构索引不存在: {path}")
    audit: Counter[str] = Counter()
    seen_canonical: set[str] = set()
    records: list[SimulationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line.strip():
            raise ValueError(f"模拟结构索引缺少头信息: {path}")
        header = json.loads(header_line)
        header_domain = header.get("chemical_domain")
        if not domain_metadata_matches(header_domain):
            raise ValueError(f"模拟结构索引缺少一致的 {DOMAIN_NAME} 元数据")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            if max_records > 0 and audit["input_rows"] >= max_records:
                break
            audit["input_rows"] += 1
            row = json.loads(line)
            if row.get("dataset_idx") is None:
                audit["missing_dataset_idx"] += 1
                continue
            dataset_idx = int(row["dataset_idx"])
            if not 0 <= dataset_idx < dataset_length:
                raise ValueError(
                    f"模拟索引第 {line_number} 行 dataset_idx 越界: "
                    f"{dataset_idx}/{dataset_length}"
                )
            inspection = inspect_smiles_domain(
                str(row.get("smiles") or ""),
                str(row.get("formula") or "") or None,
            )
            if not inspection.valid:
                for reason in inspection.reasons:
                    audit[f"excluded:{reason}"] += 1
                continue
            if inspection.canonical_smiles in seen_canonical:
                audit["duplicate_canonical"] += 1
                continue
            seen_canonical.add(inspection.canonical_smiles)
            records.append(
                SimulationRecord(
                    dataset_idx=dataset_idx,
                    formula=inspection.normalized_formula,
                    canonical=inspection.canonical_smiles,
                )
            )
            audit["domain_valid_unique"] += 1

    formula_to_records: dict[str, list[SimulationRecord]] = defaultdict(list)
    for record in records:
        formula_to_records[record.formula].append(record)
    viable_formulas = sorted(
        formula for formula, members in formula_to_records.items() if len(members) >= 2
    )
    audit["singleton_formula_records"] = sum(
        len(members) for members in formula_to_records.values() if len(members) < 2
    )
    audit["candidate_formulas_before_limit"] = len(viable_formulas)
    if max_formulas > 0 and len(viable_formulas) > max_formulas:
        random.Random(seed).shuffle(viable_formulas)
        viable_formulas = sorted(viable_formulas[:max_formulas])
    selected = set(viable_formulas)
    records = [record for record in records if record.formula in selected]
    records.sort(key=lambda item: item.dataset_idx)
    position_by_canonical = {record.canonical: index for index, record in enumerate(records)}
    formula_to_rows: dict[str, list[int]] = defaultdict(list)
    for record in records:
        formula_to_rows[record.formula].append(position_by_canonical[record.canonical])
    if len(formula_to_rows) < 2 or len(records) < 4:
        raise RuntimeError(
            "可训练同分子式组不足；需要至少两个各含两个 canonical 结构的分子式"
        )
    audit["retained_records"] = len(records)
    audit["retained_formulas"] = len(formula_to_rows)
    audit["min_candidates_per_formula"] = min(map(len, formula_to_rows.values()))
    audit["max_candidates_per_formula"] = max(map(len, formula_to_rows.values()))
    return records, {"header": header, "counts": dict(audit)}, dict(formula_to_rows)


def split_formulas(
    formula_to_rows: dict[str, list[int]], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """按完整分子式组切分，并尽量逼近目标验证结构数。"""
    formulas = sorted(formula_to_rows)
    random.Random(seed).shuffle(formulas)
    target_rows = max(2, round(sum(map(len, formula_to_rows.values())) * val_fraction))
    validation: list[str] = []
    validation_rows = 0
    for formula in formulas[:-1]:
        if validation_rows >= target_rows and validation:
            break
        validation.append(formula)
        validation_rows += len(formula_to_rows[formula])
    validation_set = set(validation)
    train = [formula for formula in formulas if formula not in validation_set]
    if not train or not validation:
        raise RuntimeError("按分子式隔离切分后 train/validation 为空")
    return sorted(train), sorted(validation)


def limit_rows(rows: list[int], maximum: int, seed: int) -> list[int]:
    """在不改变候选池的情况下限制充当查询的结构数。"""
    if maximum <= 0 or len(rows) <= maximum:
        return sorted(rows)
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return sorted(selected[:maximum])


def contrastive_domain_name(checkpoint: dict[str, Any]) -> str:
    """读取不同版本 checkpoint 中的闭域名称。"""
    payload = checkpoint.get("chemical_domain", checkpoint.get("domain"))
    if isinstance(payload, dict):
        return str(payload.get("name") or "")
    return str(payload or "")


def load_contrastive_model(
    path: Path,
    device: torch.device,
    allow_legacy: bool,
) -> tuple[SpectrumGraphContrastiveModel, dict[str, Any], str]:
    """加载并冻结光谱/图双塔，同时验证闭域元数据。"""
    if not path.is_file():
        raise FileNotFoundError(f"对比学习权重不存在: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"对比学习 checkpoint 格式不合法: {path}")
    checkpoint_domain = contrastive_domain_name(checkpoint)
    checkpoint_domain_metadata = checkpoint.get("chemical_domain")
    if not domain_metadata_matches(checkpoint_domain_metadata):
        # 迁移开关仅接受完全缺少域字段的历史权重，绝不接受声明了错误域的文件。
        if not (allow_legacy and checkpoint_domain_metadata is None):
            raise ValueError(
                f"对比学习 checkpoint 域不匹配: {checkpoint_domain or 'missing'}; "
                f"正式预训练必须使用完整 {DOMAIN_NAME} 元数据"
            )
    dare_weights = str(checkpoint.get("dare_weights") or "")
    if not dare_weights:
        raise ValueError("对比学习 checkpoint 缺少 dare_weights")
    model = SpectrumGraphContrastiveModel(
        dare_weights=dare_weights,
        node_in_dim=int(checkpoint.get("node_in_dim", 9)),
        edge_in_dim=int(checkpoint.get("edge_in_dim", 3)),
        embed_dim=int(checkpoint.get("embed_dim", 512)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, sha256_file(path)


def autocast_context(device: torch.device, enabled: bool):
    """Ampere 及更新 GPU 使用 BF16；其他设备保持 FP32。"""
    use_bfloat16 = (
        enabled
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    if use_bfloat16:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_or_build_embedding_cache(
    *,
    path: Path,
    records: list[SimulationRecord],
    dataset: PhysChemRADataset,
    contrastive: SpectrumGraphContrastiveModel,
    contrastive_sha256: str,
    structure_index: Path,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    rebuild: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """缓存冻结双塔嵌入；候选仍在每个训练步骤在线采样。"""
    signature = records_signature(records)
    structure_index_sha256 = sha256_file(structure_index)
    if path.is_file() and not rebuild:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(
                f"模拟嵌入缓存根节点不是对象: {path}；请添加 --rebuild-embedding-cache"
            )
        raw_metadata = payload.get("metadata")
        if not isinstance(raw_metadata, dict):
            raise ValueError(
                f"模拟嵌入缓存 metadata 缺失或不是对象: {path}；"
                "请添加 --rebuild-embedding-cache"
            )
        metadata = dict(raw_metadata)
        expected = {
            "schema_version": 2,
            "domain_name": DOMAIN_NAME,
            "chemical_domain": domain_metadata(),
            "formula_elements": list(FORMULA_ELEMENTS),
            "records_signature": signature,
            "contrastive_sha256": contrastive_sha256,
            "structure_index_sha256": structure_index_sha256,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        root_expected = {
            "schema_version": 2,
            "method": "simulated_formula_adapter_frozen_base_embedding_cache",
            "chemical_domain": domain_metadata(),
            "formula_elements": list(FORMULA_ELEMENTS),
        }
        mismatches.update(
            {
                f"root.{key}": (payload.get(key), value)
                for key, value in root_expected.items()
                if payload.get(key) != value
            }
        )
        spectrum = torch.as_tensor(payload.get("spectrum_embeddings"), dtype=torch.float32)
        graph = torch.as_tensor(payload.get("graph_embeddings"), dtype=torch.float32)
        expected_shape = (len(records), int(contrastive.embed_dim))
        if mismatches or tuple(spectrum.shape) != expected_shape or tuple(graph.shape) != expected_shape:
            raise ValueError(
                f"模拟嵌入缓存与当前输入不一致: metadata={mismatches}, "
                f"spectrum={tuple(spectrum.shape)}, graph={tuple(graph.shape)}, "
                "请添加 --rebuild-embedding-cache"
            )
        return spectrum.contiguous(), graph.contiguous(), metadata

    spectrum_parts: list[torch.Tensor] = []
    graph_parts: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        current = records[start : start + batch_size]
        items = [dataset[record.dataset_idx] for record in current]
        for record, item in zip(current, items):
            inspection = inspect_smiles_domain(str(item.smiles or ""))
            if (
                not inspection.valid
                or inspection.canonical_smiles != record.canonical
                or inspection.normalized_formula != record.formula
            ):
                raise RuntimeError(
                    f"索引与 PhysChemRADataset 不一致: dataset_idx={record.dataset_idx}, "
                    f"index={record.formula}/{record.canonical}, "
                    f"dataset={inspection.normalized_formula}/{inspection.canonical_smiles}"
                )
        graph_batch = Batch.from_data_list(items).to(device, non_blocking=True)
        spectra = torch.stack([item.spectrum.float() for item in items], dim=0)
        # spectra: [B,1,1800]，与 DARE 光谱塔固定输入一致。
        if spectra.ndim != 3 or tuple(spectra.shape[1:]) != (1, 1800):
            raise ValueError(f"模拟光谱批次形状异常: {tuple(spectra.shape)}")
        spectra = spectra.to(device, non_blocking=True)
        with torch.inference_mode(), autocast_context(device, amp_enabled):
            spectrum_embeddings, graph_embeddings = contrastive(graph_batch, spectra)
        # 双塔输出: [B,D] -> CPU FP32 缓存。
        if (
            spectrum_embeddings.ndim != 2
            or graph_embeddings.shape != spectrum_embeddings.shape
            or spectrum_embeddings.shape[1] != contrastive.embed_dim
        ):
            raise ValueError("冻结双塔输出维度异常")
        spectrum_parts.append(spectrum_embeddings.detach().cpu().float())
        graph_parts.append(graph_embeddings.detach().cpu().float())
        completed = min(start + batch_size, len(records))
        if completed == len(current) or completed % max(batch_size * 20, 1) == 0 or completed == len(records):
            print(f"冻结双塔嵌入 {completed}/{len(records)}", flush=True)

    spectrum = torch.cat(spectrum_parts, dim=0).contiguous()
    graph = torch.cat(graph_parts, dim=0).contiguous()
    metadata = {
        "schema_version": 2,
        "domain_name": DOMAIN_NAME,
        "chemical_domain": domain_metadata(),
        "formula_elements": list(FORMULA_ELEMENTS),
        "records_signature": signature,
        "record_count": len(records),
        "embedding_dim": int(contrastive.embed_dim),
        "contrastive_sha256": contrastive_sha256,
        "structure_index": str(structure_index.resolve()),
        "structure_index_sha256": structure_index_sha256,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "method": "simulated_formula_adapter_frozen_base_embedding_cache",
            "chemical_domain": domain_metadata(),
            "formula_elements": list(FORMULA_ELEMENTS),
            "metadata": metadata,
            "records": [asdict(record) for record in records],
            "spectrum_embeddings": spectrum,
            "graph_embeddings": graph,
        },
        path,
    )
    return spectrum, graph, metadata


def make_training_batch(
    query_rows: list[int],
    records: list[SimulationRecord],
    formula_to_rows: dict[str, list[int]],
    spectrum_embeddings: torch.Tensor,
    graph_embeddings: torch.Tensor,
    candidates_per_query: int,
    rng: random.Random,
    base_hard_negative_k: int,
    negative_sampling: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从完整同式池构造候选；默认混合冻结双塔 hard negative 与随机负样本。"""
    batch_size = len(query_rows)
    candidate_rows = torch.zeros((batch_size, candidates_per_query), dtype=torch.long)
    candidate_mask = torch.zeros((batch_size, candidates_per_query), dtype=torch.bool)
    positive_indices = torch.zeros(batch_size, dtype=torch.long)
    formula_features: list[torch.Tensor] = []
    for batch_index, query_row in enumerate(query_rows):
        record = records[query_row]
        negatives = [row for row in formula_to_rows[record.formula] if row != query_row]
        if not negatives:
            raise RuntimeError(f"分子式组没有可用负样本: {record.formula}")
        negative_count = min(candidates_per_query - 1, len(negatives))
        if negative_sampling == "base_topk_mixed" and base_hard_negative_k > 0:
            # graph_embeddings: [R,D]，query_embedding: [D] -> base_scores: [R]。
            # 该分数来自冻结双塔，不使用真实标签、PubChem 顺序或候选位置。
            negative_tensor = torch.tensor(negatives, dtype=torch.long)
            base_scores = graph_embeddings[negative_tensor] @ spectrum_embeddings[query_row]
            hard_count = min(base_hard_negative_k, negative_count, len(negatives))
            hard_order = torch.argsort(base_scores, descending=True)[:hard_count]
            hard_rows = [negatives[int(index)] for index in hard_order.tolist()]
            remaining = [row for row in negatives if row not in set(hard_rows)]
            random_count = negative_count - len(hard_rows)
            random_rows = rng.sample(remaining, min(random_count, len(remaining)))
            selected = hard_rows + random_rows
            if len(selected) < negative_count:
                supplement = [row for row in remaining if row not in set(random_rows)]
                selected.extend(supplement[: negative_count - len(selected)])
            rng.shuffle(selected)
        else:
            selected = rng.sample(negatives, negative_count)
        candidates = [query_row, *selected]
        rng.shuffle(candidates)
        positive_index = candidates.index(query_row)
        candidate_rows[batch_index, : len(candidates)] = torch.tensor(candidates)
        candidate_mask[batch_index, : len(candidates)] = True
        positive_indices[batch_index] = positive_index
        formula_features.append(formula_vector(record.formula))

    query_base = spectrum_embeddings[torch.tensor(query_rows, dtype=torch.long)]
    # candidate_rows: [B,K]；索引 graph_embeddings [R,D] -> candidate_base [B,K,D]。
    candidate_base = graph_embeddings[candidate_rows]
    formula_batch = torch.stack(formula_features, dim=0)
    if query_base.ndim != 2 or candidate_base.ndim != 3:
        raise ValueError("训练批次嵌入维度异常")
    return query_base, candidate_base, formula_batch, candidate_mask, positive_indices


def sampled_listwise_loss(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_indices: torch.Tensor,
    hard_negative_k: int,
    hard_negative_weight: float,
    hard_negative_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算带同式硬负样本约束的批量 Listwise 损失。"""
    if scores.ndim != 2 or candidate_mask.shape != scores.shape:
        raise ValueError("scores 和 candidate_mask 必须为同形状 [B,K]")
    if positive_indices.shape != (scores.shape[0],):
        raise ValueError("positive_indices 必须为 [B]")
    masked_scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
    listwise = F.cross_entropy(masked_scores, positive_indices)
    # positive_indices: [B] -> [B,1]，逐行读取正样本分数。
    positive_scores = masked_scores.gather(1, positive_indices.unsqueeze(1)).squeeze(1)
    negative_mask = candidate_mask.clone()
    negative_mask.scatter_(1, positive_indices.unsqueeze(1), False)
    hard_count = min(hard_negative_k, scores.shape[1] - 1)
    hard_scores = masked_scores.masked_fill(~negative_mask, torch.finfo(scores.dtype).min)
    # hard_scores: [B,K] -> [B,H]，H 为批次统一硬负样本上限。
    hard_values = torch.topk(hard_scores, k=hard_count, dim=1).values
    finite_mask = torch.isfinite(hard_values) & (hard_values > torch.finfo(scores.dtype).min / 2)
    hard_terms = F.softplus(
        float(hard_negative_margin) - positive_scores.unsqueeze(1) + hard_values
    )
    hard_loss = (hard_terms * finite_mask).sum() / finite_mask.sum().clamp_min(1)
    total = listwise + float(hard_negative_weight) * hard_loss
    return total, {
        "listwise": float(listwise.detach().item()),
        "hard_negative": float(hard_loss.detach().item()),
    }


@torch.inference_mode()
def evaluate_full_formula_pools(
    model: FormulaConditionedDualAdapter,
    query_rows: list[int],
    records: list[SimulationRecord],
    formula_to_rows: dict[str, list[int]],
    spectrum_embeddings: torch.Tensor,
    graph_embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float | int | None]:
    """在每个验证分子式的完整模拟结构池上计算无保护排名。"""
    model.eval()
    queries_by_formula: dict[str, list[int]] = defaultdict(list)
    for row in query_rows:
        queries_by_formula[records[row].formula].append(row)
    ranks: list[int] = []
    candidate_counts: list[int] = []
    scale = model.logit_scale.exp().clamp(max=100.0)
    for formula in sorted(queries_by_formula):
        candidate_rows = formula_to_rows[formula]
        candidate_tensor = graph_embeddings[torch.tensor(candidate_rows, dtype=torch.long)]
        formula_feature = formula_vector(formula)
        with autocast_context(device, amp_enabled):
            # [N,D] -> [1,N,D]，单个分子式条件对应完整候选池。
            adapted_candidates = model.encode_molecule(
                candidate_tensor.to(device, non_blocking=True).unsqueeze(0),
                formula_feature.to(device, non_blocking=True).unsqueeze(0),
            ).squeeze(0)
        candidate_position = {row: index for index, row in enumerate(candidate_rows)}
        formula_queries = queries_by_formula[formula]
        for start in range(0, len(formula_queries), batch_size):
            current = formula_queries[start : start + batch_size]
            query_base = spectrum_embeddings[torch.tensor(current, dtype=torch.long)].to(
                device, non_blocking=True
            )
            # formula_feature: [F] -> [B,F]，同一公式组复用条件向量。
            formula_batch = formula_feature.to(device, non_blocking=True).unsqueeze(0).expand(
                len(current), -1
            )
            with autocast_context(device, amp_enabled):
                adapted_queries = model.encode_query(query_base, formula_batch)
                # [B,D] @ [D,N] -> [B,N] 完整同分子式候选得分。
                scores = scale * (adapted_queries @ adapted_candidates.transpose(0, 1))
            positives = torch.tensor(
                [candidate_position[row] for row in current],
                dtype=torch.long,
                device=device,
            )
            positive_scores = scores.gather(1, positives.unsqueeze(1))
            # 严格高于正样本的候选数加一即 best-tie rank。
            batch_ranks = 1 + (scores > positive_scores).sum(dim=1)
            ranks.extend(int(value) for value in batch_ranks.cpu().tolist())
            candidate_counts.extend([len(candidate_rows)] * len(current))
    values = np.asarray(ranks, dtype=np.int64)
    counts = np.asarray(candidate_counts, dtype=np.int64)
    return {
        "count": int(values.size),
        "formula_count": len(queries_by_formula),
        "top1": float(np.mean(values <= 1)) if values.size else 0.0,
        "top5": float(np.mean(values <= 5)) if values.size else 0.0,
        "top10": float(np.mean(values <= 10)) if values.size else 0.0,
        "top24": float(np.mean(values <= 24)) if values.size else 0.0,
        "mrr": float(np.mean(1.0 / values)) if values.size else 0.0,
        "median_rank": float(np.median(values)) if values.size else None,
        "mean_candidate_count": float(np.mean(counts)) if counts.size else 0.0,
        "max_candidate_count": int(np.max(counts)) if counts.size else 0,
        "tie_policy": "best_rank",
    }


def save_checkpoint(
    path: Path,
    model: FormulaConditionedDualAdapter,
    args: argparse.Namespace,
    contrastive_sha256: str,
    cache_metadata: dict[str, Any],
    train_formulas: list[str],
    validation_formulas: list[str],
    validation: dict[str, Any],
    epoch: int,
) -> None:
    """保存带完整域定义与隔离切分元数据的最佳适配器。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "method": "simulated_same_formula_candidate_adapter",
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "formula_dim": FORMULA_DIM,
            "formula_elements": list(FORMULA_ELEMENTS),
            "hidden_dim": args.hidden_dim,
            "residual_scale": model.residual_scale,
            "chemical_domain": domain_metadata(),
            "contrastive_weights": str(args.contrastive_weights.resolve()),
            "contrastive_sha256": contrastive_sha256,
            "embedding_cache": str(args.embedding_cache.resolve()),
            "embedding_cache_metadata": cache_metadata,
            "structure_index": str(args.structure_index.resolve()),
            "candidate_policy": "positive=self_graph; negatives=same_formula_other_canonical_online",
            "negative_sampling": args.negative_sampling,
            "base_hard_negative_k": args.base_hard_negative_k,
            "split_policy": "disjoint_normalized_formula_and_canonical_structure",
            "train_formulas": train_formulas,
            "validation_formulas": validation_formulas,
            "best_epoch": epoch,
            "best_validation": validation,
            "seed": args.seed,
            "split_seed": args.split_seed,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    configure_smoke(args)
    if tuple(ADAPTER_FORMULA_ELEMENTS) != tuple(FORMULA_ELEMENTS) or FORMULA_DIM != 10:
        raise RuntimeError(
            "Formula Adapter 与统一闭域词表不一致；必须统一为 "
            f"{list(FORMULA_ELEMENTS)}"
        )
    seed_everything(args.seed)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = not args.no_amp
    dataset = PhysChemRADataset(
        root=str(args.data_root), verbose=False, max_cached_chunks=1
    )
    records, index_audit, formula_to_rows = read_structure_index(
        args.structure_index,
        len(dataset),
        args.max_index_records,
        args.max_formulas,
        args.seed,
    )
    train_formulas, validation_formulas = split_formulas(
        formula_to_rows, args.val_fraction, args.split_seed
    )
    train_rows = limit_rows(
        [row for formula in train_formulas for row in formula_to_rows[formula]],
        args.max_train_samples,
        args.seed + 1,
    )
    validation_rows = limit_rows(
        [row for formula in validation_formulas for row in formula_to_rows[formula]],
        args.max_val_samples,
        args.seed + 2,
    )
    if len(train_rows) < 2 or len(validation_rows) < 2:
        raise RuntimeError(
            f"有效训练/验证查询不足: {len(train_rows)}/{len(validation_rows)}"
        )
    train_structures = {records[row].canonical for row in train_rows}
    validation_structures = {records[row].canonical for row in validation_rows}
    formula_overlap = set(train_formulas).intersection(validation_formulas)
    structure_overlap = train_structures.intersection(validation_structures)
    if formula_overlap or structure_overlap:
        raise RuntimeError(
            f"隔离切分失败: formula_overlap={len(formula_overlap)}, "
            f"structure_overlap={len(structure_overlap)}"
        )

    contrastive, _, contrastive_sha256 = load_contrastive_model(
        args.contrastive_weights, device, args.allow_legacy_contrastive
    )
    spectrum_embeddings, graph_embeddings, cache_metadata = load_or_build_embedding_cache(
        path=args.embedding_cache,
        records=records,
        dataset=dataset,
        contrastive=contrastive,
        contrastive_sha256=contrastive_sha256,
        structure_index=args.structure_index,
        batch_size=args.embedding_batch_size,
        device=device,
        amp_enabled=amp_enabled,
        rebuild=args.rebuild_embedding_cache,
    )
    del contrastive
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = FormulaConditionedDualAdapter(
        embedding_dim=int(graph_embeddings.shape[1]),
        formula_dim=FORMULA_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print(
        f"模拟同式 Adapter train/val={len(train_rows)}/{len(validation_rows)}，"
        f"formula={len(train_formulas)}/{len(validation_formulas)}，device={device}；"
        "正样本=自身图，负样本=同分子式其他 canonical 结构",
        flush=True,
    )
    baseline = evaluate_full_formula_pools(
        model,
        validation_rows,
        records,
        formula_to_rows,
        spectrum_embeddings,
        graph_embeddings,
        args.evaluation_batch_size,
        device,
        amp_enabled,
    )
    print(f"冻结双塔基线验证: {baseline}", flush=True)
    best_validation = baseline
    best_epoch = 0
    best_key = (float(baseline["top1"]), float(baseline["mrr"]), float(baseline["top5"]))
    save_checkpoint(
        args.output_weights,
        model,
        args,
        contrastive_sha256,
        cache_metadata,
        train_formulas,
        validation_formulas,
        baseline,
        0,
    )
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train_rows)
        random.Random(args.seed + epoch).shuffle(order)
        epoch_rng = random.Random(args.seed * 1009 + epoch)
        loss_sum = 0.0
        listwise_sum = 0.0
        hard_sum = 0.0
        step_count = 0
        total_steps = (len(order) + args.batch_size - 1) // args.batch_size
        for start in range(0, len(order), args.batch_size):
            query_rows = order[start : start + args.batch_size]
            query_base, candidate_base, formula_batch, candidate_mask, positives = make_training_batch(
                query_rows,
                records,
                formula_to_rows,
                spectrum_embeddings,
                graph_embeddings,
                args.candidates_per_query,
                epoch_rng,
                args.base_hard_negative_k,
                args.negative_sampling,
            )
            query_base = query_base.to(device, non_blocking=True)
            candidate_base = candidate_base.to(device, non_blocking=True)
            formula_batch = formula_batch.to(device, non_blocking=True)
            candidate_mask = candidate_mask.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                scores = model(query_base, candidate_base, formula_batch)
                loss, components = sampled_listwise_loss(
                    scores,
                    candidate_mask,
                    positives,
                    args.hard_negative_k,
                    args.hard_negative_weight,
                    args.hard_negative_margin,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"训练损失出现非有限值: epoch={epoch}, step={step_count + 1}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"梯度范数出现非有限值: epoch={epoch}, step={step_count + 1}"
                )
            optimizer.step()
            step_count += 1
            loss_sum += float(loss.detach().item())
            listwise_sum += components["listwise"]
            hard_sum += components["hard_negative"]
            if step_count == 1 or step_count % args.log_interval == 0 or step_count == total_steps:
                print(
                    f"Epoch {epoch}/{args.epochs} step {step_count}/{total_steps} "
                    f"loss={float(loss.detach().item()):.4f}",
                    flush=True,
                )

        validation = evaluate_full_formula_pools(
            model,
            validation_rows,
            records,
            formula_to_rows,
            spectrum_embeddings,
            graph_embeddings,
            args.evaluation_batch_size,
            device,
            amp_enabled,
        )
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(step_count, 1),
            "listwise_loss": listwise_sum / max(step_count, 1),
            "hard_negative_loss": hard_sum / max(step_count, 1),
            "validation": validation,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
            f"val Top1/5/10/24={validation['top1']:.3f}/{validation['top5']:.3f}/"
            f"{validation['top10']:.3f}/{validation['top24']:.3f} "
            f"MRR={validation['mrr']:.4f}",
            flush=True,
        )
        key = (
            float(validation["top1"]),
            float(validation["mrr"]),
            float(validation["top5"]),
        )
        if key > best_key:
            best_key = key
            best_validation = validation
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_weights,
                model,
                args,
                contrastive_sha256,
                cache_metadata,
                train_formulas,
                validation_formulas,
                validation,
                epoch,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"验证指标连续 {args.patience} 轮未改善，提前停止。",
                    flush=True,
                )
                break

    report = {
        "schema_version": 2,
        "method": "simulated_same_formula_candidate_adapter",
        "status": "completed",
        "chemical_domain": domain_metadata(),
        "formula_elements": list(FORMULA_ELEMENTS),
        "data": {
            "structure_index": str(args.structure_index.resolve()),
            "index_audit": index_audit,
            "retained_records": len(records),
            "candidate_formula_count": len(formula_to_rows),
        },
        "split": {
            "policy": "disjoint_normalized_formula_and_canonical_structure",
            "train_formula_count": len(train_formulas),
            "validation_formula_count": len(validation_formulas),
            "train_query_count": len(train_rows),
            "validation_query_count": len(validation_rows),
            "formula_overlap": len(formula_overlap),
            "canonical_structure_overlap": len(structure_overlap),
            "train_formulas": train_formulas,
            "validation_formulas": validation_formulas,
            "split_seed": args.split_seed,
        },
        "candidate_policy": {
            "positive": "query_record_self_graph",
            "negative": "same_normalized_formula_other_canonical_only",
            "sampling": (
                "complete_same_formula_base_cosine_topk_plus_random_without_replacement_each_epoch"
                if args.negative_sampling == "base_topk_mixed"
                else "online_random_without_replacement_each_epoch"
            ),
            "negative_sampling": args.negative_sampling,
            "base_hard_negative_k": args.base_hard_negative_k,
            "pubchem_pool_used": False,
            "pubchem_rank_or_teacher_used": False,
            "training_candidate_limit": args.candidates_per_query,
            "validation_candidates": "complete_same_formula_simulation_pool",
        },
        "contrastive_weights": str(args.contrastive_weights.resolve()),
        "contrastive_sha256": contrastive_sha256,
        "embedding_cache": str(args.embedding_cache.resolve()),
        "embedding_cache_metadata": cache_metadata,
        "baseline_validation": baseline,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "weights": str(args.output_weights.resolve()),
        "history": history,
        "smoke": bool(args.smoke),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"权重: {args.output_weights.resolve()}\n报告: {args.report.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

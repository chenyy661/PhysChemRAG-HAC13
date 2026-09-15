"""在固定 GraphTower/FAISS 坐标下微调 DARE 光谱侧的隔离脚本。

该脚本只更新 DARE 的光谱特征提取器和投影头，GraphTower 保持冻结，因此输出的
光谱向量仍然可以直接查询现有 FAISS。合成配对用于保持全库坐标，真实实验配对
用于修正域偏移。Level4 测试集只用于独立验证，不参与训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Batch, Data
from torch_geometric.utils.smiles import from_smiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physchemrag.config import (  # noqa: E402
    DATA_DIR,
    LEVEL4_EXPERIMENTAL_DIR,
    MODULE1_DARE_WEIGHTS,
    MODULE2_DARE_FUSION_WEIGHTS,
    MODULE2_GRAPH_WEIGHTS,
    OUTPUT_REPORTS_DIR,
    WAVE_LEN,
)
from physchemrag.module2_crossmodal.graph_tower import GraphTower  # noqa: E402
from physchemrag.module2_crossmodal.spectrum_tower import DARESpectrumTower  # noqa: E402
from physchemrag.module4_cascade.spectral_views import prepare_masked_spectrum  # noqa: E402
from physchemrag.shared.chemical_domain import domain_metadata, inspect_smiles_domain  # noqa: E402
from physchemrag.shared.dataset import PhysChemRADataset  # noqa: E402
from physchemrag.shared.molecular_formula import formula_from_smiles, normalize_formula  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-dir",
        type=Path,
        nargs="+",
        default=[PROJECT_ROOT / "data" / "level1_experimental_alignment_expanded"],
        help="一个或多个真实配对目录；按给定顺序合并并按光谱内容去重",
    )
    parser.add_argument(
        "--exclude-dir",
        type=Path,
        nargs="*",
        default=[LEVEL4_EXPERIMENTAL_DIR / "test"],
        help="训练前按 CAS 和非立体连接结构排除的测试目录",
    )
    parser.add_argument("--synthetic-root", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--synthetic-index",
        type=Path,
        default=DATA_DIR / "conformers_v2/training_index_decontaminated_all_external_domain_hac13.jsonl",
        help="只允许从该去污染闭域索引读取模拟 dataset_idx",
    )
    parser.add_argument("--synthetic-count", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--real-batch-size",
        type=int,
        default=64,
        help="真实配对 batch；正式配置固定为 64，避免改变优化步数",
    )
    parser.add_argument(
        "--synthetic-batch-size",
        type=int,
        default=128,
        help="模拟配对 batch；正式配置固定为 128，避免大 batch 欠训练",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=50000,
        help="合成索引块大小；应与底层物理 chunk 大小一致，减少重复反序列化",
    )
    parser.add_argument("--lr-encoder", type=float, default=1e-6)
    parser.add_argument("--lr-projection", type=float, default=2e-6)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument("--real-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=10, help="每隔多少个训练 step 打印一次进度")
    parser.add_argument(
        "--preload-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将选中的模拟样本预载入内存，避免每个 epoch 重复反序列化物理 chunk",
    )
    parser.add_argument(
        "--preload-max-samples",
        type=int,
        default=50000,
        help="预载入上限；超过该数量时自动回退为按 chunk 懒加载",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只执行真实数据合并、去污染和拆分审计，不加载模型或开始训练",
    )
    parser.add_argument(
        "--graph-weights",
        type=Path,
        default=MODULE2_GRAPH_WEIGHTS,
        help="冻结图塔权重；可传双塔 checkpoint，脚本会提取 graph_tower 参数",
    )
    parser.add_argument(
        "--initial-projection",
        type=Path,
        default=MODULE2_DARE_FUSION_WEIGHTS,
        help="作为微调起点的光谱投影头权重",
    )
    parser.add_argument(
        "--initial-dare",
        type=Path,
        default=MODULE1_DARE_WEIGHTS,
        help="作为微调起点的 DARE 光谱编码器权重",
    )
    parser.add_argument(
        "--encoder-output",
        type=Path,
        default=PROJECT_ROOT / "weights" / "module1_dare_physics_finetuned_alignment_smoke.pth",
    )
    parser.add_argument(
        "--projection-output",
        type=Path,
        default=PROJECT_ROOT / "weights" / "module2_dare_fusion_dare_finetuned_alignment_smoke.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_REPORTS_DIR / "dare_spectrum_alignment_smoke.json",
    )
    args = parser.parse_args()
    if args.synthetic_count < 0 or args.epochs < 1:
        parser.error("synthetic-count 不能为负，epochs 至少为 1；0 表示使用闭域索引全部样本")
    if args.block_size < args.synthetic_batch_size:
        parser.error("block-size 必须不小于 synthetic-batch-size")
    if args.preload_max_samples < 0:
        parser.error("preload-max-samples 不能为负")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def scaffold_key(smiles: str) -> str:
    """生成真实配对拆分使用的保守骨架键。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core.GetNumAtoms() > 0:
        return "murcko:" + Chem.MolToSmiles(
            core,
            canonical=True,
            isomericSmiles=False,
        )
    elements = defaultdict(int)
    for atom in mol.GetAtoms():
        elements[atom.GetSymbol()] += 1
    element_key = ",".join(f"{key}{elements[key]}" for key in sorted(elements))
    bond_key = ",".join(
        str(sum(bond.GetBondTypeAsDouble() == order for bond in mol.GetBonds()))
        for order in (1.0, 2.0, 3.0)
    )
    return f"acyclic:{element_key}:bonds={bond_key}"


def graph_from_smiles(smiles: str) -> Data:
    data = from_smiles(smiles)
    if data is None:
        raise ValueError(f"GraphTower 无法解析结构: {smiles}")
    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()
    return data


def spectrum_sha256(value: torch.Tensor) -> str:
    """按连续 float32 光谱数组计算稳定内容哈希。"""
    spectrum = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    return hashlib.sha256(spectrum.numpy().tobytes()).hexdigest()


def load_real_records(directories: list[Path] | Path) -> tuple[list[dict], dict]:
    """合并真实配对目录，并按光谱内容进行确定性去重。"""
    if isinstance(directories, Path):
        directories = [directories]
    records: list[dict] = []
    global_reasons: Counter[str] = Counter()
    seen_spectra: set[str] = set()
    source_summaries: list[dict] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"真实配对目录不存在: {directory}")
        source = str(directory.resolve())
        reasons: Counter[str] = Counter()
        input_count = 0
        eligible_before_dedup = 0
        duplicate_spectrum = 0
        retained_after_dedup = 0
        for path in sorted(directory.glob("*.pt")):
            input_count += 1
            payload = torch.load(path, map_location="cpu", weights_only=False)
            spectrum = torch.as_tensor(payload.get("spectrum"), dtype=torch.float32).view(-1)
            if spectrum.numel() != WAVE_LEN or not torch.isfinite(spectrum).all():
                raise ValueError(f"真实光谱异常: {path}")
            smiles = str(payload.get("connectivity_smiles") or payload.get("smiles") or "")
            formula = normalize_formula(payload.get("molecular_formula") or formula_from_smiles(smiles) or "")
            domain = inspect_smiles_domain(smiles, formula)
            if not domain.valid:
                for reason in domain.reasons:
                    reasons[reason] += 1
                    global_reasons[reason] += 1
                continue
            eligible_before_dedup += 1
            digest = spectrum_sha256(spectrum)
            if digest in seen_spectra:
                duplicate_spectrum += 1
                continue
            seen_spectra.add(digest)
            masked_spectrum, valid_mask = prepare_masked_spectrum(
                spectrum,
                payload.get("valid_mask"),
            )
            records.append({
                "cas": str(payload.get("cas", path.stem)),
                "canonical": canonical(smiles),
                "scaffold": scaffold_key(smiles),
                "spectrum_hash": digest,
                "source": source,
                "source_path": str(path.resolve()),
                "spectrum": masked_spectrum.squeeze(0).squeeze(0),
                "valid_fraction": float(valid_mask.float().mean().item()),
                "graph": graph_from_smiles(smiles),
            })
            retained_after_dedup += 1
        source_summaries.append({
            "source": source,
            "input": input_count,
            "hac13_eligible_before_dedup": eligible_before_dedup,
            "domain_excluded": input_count - eligible_before_dedup,
            "domain_reasons": dict(reasons),
            "duplicate_spectrum": duplicate_spectrum,
            "retained_after_dedup": retained_after_dedup,
        })
    return records, {
        "input": sum(row["input"] for row in source_summaries),
        "eligible_before_dedup": sum(row["hac13_eligible_before_dedup"] for row in source_summaries),
        "domain_excluded": sum(row["domain_excluded"] for row in source_summaries),
        "duplicate_spectrum": sum(row["duplicate_spectrum"] for row in source_summaries),
        "retained_after_dedup": len(records),
        "reasons": dict(global_reasons),
        "sources": source_summaries,
    }


def load_synthetic_indices(path: Path, dataset_length: int) -> tuple[list[int], dict]:
    """读取闭域去污染索引，并拒绝没有域声明或越界的索引。"""
    if not path.is_file():
        raise FileNotFoundError(f"闭域模拟索引不存在: {path}")
    indices: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        chemical_domain = header.get("chemical_domain", {})
        if chemical_domain != domain_metadata():
            raise ValueError(f"模拟索引化学域声明不一致: {chemical_domain}")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_idx = int(row.get("dataset_idx", -1))
            if not 0 <= dataset_idx < dataset_length:
                raise ValueError(f"模拟索引第 {line_number} 行 dataset_idx 越界: {dataset_idx}")
            indices.append(dataset_idx)
    if len(indices) < 2:
        raise RuntimeError("闭域模拟索引可用样本少于 2 条")
    return indices, {"header": header, "rows": len(indices)}


def load_exclusion_keys(directories: list[Path]) -> tuple[set[str], set[str], set[str]]:
    """读取冻结目录中的 CAS、非立体连接结构和光谱哈希。"""
    excluded_cas: set[str] = set()
    excluded_connectivity: set[str] = set()
    excluded_spectra: set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"排除目录不存在: {directory}")
        for path in sorted(directory.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            excluded_cas.add(str(payload.get("cas", path.stem)))
            spectrum = torch.as_tensor(payload.get("spectrum"), dtype=torch.float32).view(-1)
            if spectrum.numel() != WAVE_LEN or not torch.isfinite(spectrum).all():
                raise ValueError(f"冻结目录光谱异常: {path}")
            excluded_spectra.add(spectrum_sha256(spectrum))
            smiles = str(payload.get("connectivity_smiles") or payload.get("smiles") or "")
            if smiles:
                excluded_connectivity.add(canonical(smiles))
    return excluded_cas, excluded_connectivity, excluded_spectra


def exclude_real_records(
    records: list[dict],
    excluded_cas: set[str],
    excluded_connectivity: set[str],
    excluded_spectra: set[str],
) -> tuple[list[dict], dict]:
    """在拆分前排除测试身份，防止真实配对监督泄漏。"""
    kept = []
    excluded_by_cas = 0
    excluded_by_connectivity_only = 0
    excluded_by_spectrum_only = 0
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        source_counts[row["source"]]["available_after_dedup"] += 1
        if row["cas"] in excluded_cas:
            excluded_by_cas += 1
            source_counts[row["source"]]["excluded_by_cas"] += 1
            continue
        if row["canonical"] in excluded_connectivity:
            excluded_by_connectivity_only += 1
            source_counts[row["source"]]["excluded_by_connectivity_only"] += 1
            continue
        if row["spectrum_hash"] in excluded_spectra:
            excluded_by_spectrum_only += 1
            source_counts[row["source"]]["excluded_by_spectrum_only"] += 1
            continue
        kept.append(row)
        source_counts[row["source"]]["final_retained"] += 1
    if len(kept) < 100:
        raise RuntimeError(f"去污染后的真实配对数量过少: {len(kept)}")
    if {row["cas"] for row in kept} & excluded_cas:
        raise RuntimeError("真实配对 CAS 排除失败")
    if {row["canonical"] for row in kept} & excluded_connectivity:
        raise RuntimeError("真实配对连接结构排除失败")
    if {row["spectrum_hash"] for row in kept} & excluded_spectra:
        raise RuntimeError("真实配对光谱哈希排除失败")
    return kept, {
        "available_real_records": len(records),
        "excluded_by_cas": excluded_by_cas,
        "excluded_by_connectivity_only": excluded_by_connectivity_only,
        "excluded_by_spectrum_only": excluded_by_spectrum_only,
        "retained_real_records": len(kept),
        "sources": {source: dict(counts) for source, counts in source_counts.items()},
    }


def build_real_source_audit(domain_summary: dict, exclusion_summary: dict) -> list[dict]:
    """合并域过滤、光谱去重与冻结排除的逐来源统计。"""
    excluded_sources = exclusion_summary.get("sources", {})
    output = []
    for source_row in domain_summary.get("sources", []):
        source = source_row["source"]
        excluded = excluded_sources.get(source, {})
        output.append({
            **source_row,
            "excluded_by_cas": int(excluded.get("excluded_by_cas", 0)),
            "excluded_by_connectivity_only": int(
                excluded.get("excluded_by_connectivity_only", 0)
            ),
            "excluded_by_spectrum_only": int(excluded.get("excluded_by_spectrum_only", 0)),
            "final_retained": int(excluded.get("final_retained", 0)),
        })
    return output


def split_scaffold_disjoint(
    records: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """按完整骨架组拆分，并使验证样本数尽量不超过目标比例。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["scaffold"]].append(index)
    shuffled_groups = list(groups.values())
    random.Random(seed).shuffle(shuffled_groups)
    target = max(1, round(len(records) * val_fraction))
    validation_groups: list[list[int]] = []
    training_groups: list[list[int]] = []
    validation_count = 0
    for group in shuffled_groups:
        if validation_count + len(group) <= target:
            validation_groups.append(group)
            validation_count += len(group)
        else:
            training_groups.append(group)
    if validation_count < 2:
        # 极端小数据下选择最小可移动骨架组，仍不拆散组内记录。
        for group in sorted(training_groups, key=len):
            remaining = sum(len(item) for item in training_groups) - len(group)
            if remaining >= 2:
                training_groups.remove(group)
                validation_groups.append(group)
                validation_count += len(group)
                if validation_count >= 2:
                    break
    validation = [index for group in validation_groups for index in group]
    training = [index for group in training_groups for index in group]
    if len(validation) < 2 or len(training) < 2:
        raise RuntimeError("骨架隔离拆分后训练集或验证集过小")
    train_scaffolds = {records[index]["scaffold"] for index in training}
    val_scaffolds = {records[index]["scaffold"] for index in validation}
    if train_scaffolds & val_scaffolds:
        raise RuntimeError("骨架隔离拆分失败")
    train_connectivity = {records[index]["canonical"] for index in training}
    val_connectivity = {records[index]["canonical"] for index in validation}
    if train_connectivity & val_connectivity:
        raise RuntimeError("同一连接结构的多条实验谱被拆分到训练集和验证集")
    return sorted(training), sorted(validation)


def load_models(
    initial_dare: Path,
    initial_projection: Path,
    graph_weights: Path,
) -> tuple[DARESpectrumTower, DARESpectrumTower, GraphTower]:
    if not initial_dare.is_file():
        raise FileNotFoundError(f"缺少初始 DARE 权重: {initial_dare}")
    model = DARESpectrumTower(str(initial_dare), final_dim=512).to(DEVICE)
    baseline = DARESpectrumTower(str(initial_dare), final_dim=512).to(DEVICE)
    graph = GraphTower(node_in_dim=9, edge_in_dim=3, final_dim=512).to(DEVICE)
    graph_state = torch.load(graph_weights, map_location=DEVICE, weights_only=False)
    if isinstance(graph_state, dict) and "model_state_dict" in graph_state:
        graph_state = graph_state["model_state_dict"]
    if any(str(key).startswith("graph_tower.") for key in graph_state):
        graph_state = {
            str(key).removeprefix("graph_tower."): value
            for key, value in graph_state.items()
            if str(key).startswith("graph_tower.")
        }
    graph.load_state_dict(graph_state, strict=True)
    if not initial_projection.is_file():
        raise FileNotFoundError(f"缺少初始投影权重: {initial_projection}")
    projection_state = torch.load(
        initial_projection,
        map_location=DEVICE,
        weights_only=False,
    )
    if isinstance(projection_state, dict):
        for wrapper_key in ("state_dict", "projection_state_dict", "model_state_dict"):
            nested = projection_state.get(wrapper_key)
            if isinstance(nested, dict):
                projection_state = nested
                break
    model.projection_head.load_state_dict(projection_state, strict=True)
    baseline.projection_head.load_state_dict(projection_state, strict=True)

    # 只解冻 DARE 的特征提取器；分类器和域判别器不参与本次对齐微调。
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.encoder.feature_extractor.parameters():
        parameter.requires_grad = True
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = True
    baseline.eval()
    graph.eval()
    return model, baseline, graph


def encode_graphs(graph_model: GraphTower, graphs: list[Data], batch_size: int = 128) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size]).to(DEVICE)
            z, _ = graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_pretrain=True)
            outputs.append(F.normalize(z, p=2, dim=-1).cpu().clone().detach())
    return torch.cat(outputs, dim=0).clone().detach()


def spectrum_features(model: DARESpectrumTower, spectra: torch.Tensor) -> torch.Tensor:
    """绕过 DARESpectrumTower.forward 的 no_grad 包装，保留特征梯度。"""
    _, features, _ = model.encoder(spectra)
    return features


def baseline_embeddings(model: DARESpectrumTower, spectra: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, features, _ = model.encoder(spectra)
        return F.normalize(model.projection_head(features), p=2, dim=-1).clone().detach()


def symmetric_infonce(z_spec: torch.Tensor, z_mol: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z_spec @ z_mol.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def rank_metrics(z_spec: torch.Tensor, z_mol: torch.Tensor) -> dict:
    logits = F.normalize(z_spec, p=2, dim=-1) @ F.normalize(z_mol, p=2, dim=-1).t()
    labels = torch.arange(logits.size(0), device=logits.device)
    order = logits.argsort(dim=1, descending=True)
    locations = (order == labels[:, None]).nonzero(as_tuple=False)
    ranks = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
    ranks[locations[:, 0]] = locations[:, 1] + 1
    return {
        "count": int(ranks.numel()),
        "top1": float((ranks <= 1).float().mean().item()),
        "top5": float((ranks <= 5).float().mean().item()),
        "top24": float((ranks <= 24).float().mean().item()),
        "mean_rank": float(ranks.float().mean().item()),
        "median_rank": float(ranks.float().median().item()),
    }


def fetch_batch(
    dataset: PhysChemRADataset,
    absolute_indices: list[int],
    excluded_connectivity: set[str],
    preloaded: dict[int, Data] | None = None,
) -> tuple[torch.Tensor, Batch, int]:
    kept = []
    skipped = 0
    for index in absolute_indices:
        if preloaded is not None:
            row = preloaded.get(int(index))
            if row is None:
                skipped += 1
                continue
        else:
            row = dataset[index]
        smiles = str(getattr(row, "smiles", "") or "")
        if not smiles or canonical(smiles) in excluded_connectivity:
            skipped += 1
            continue
        row.x = row.x.float()
        row.edge_attr = row.edge_attr.float()
        kept.append(row)
    if len(kept) < 2:
        raise RuntimeError("合成批次在 Level4 结构去污染后少于 2 条")
    spectra = torch.stack([row.spectrum.float() for row in kept])
    return spectra, Batch.from_data_list(kept), skipped


def preload_synthetic_samples(
    dataset: PhysChemRADataset,
    indices: list[int],
    excluded_connectivity: set[str],
) -> dict[int, Data]:
    """按排序后的索引一次读取模拟样本，训练 epoch 间只保留内存对象。"""
    cached: dict[int, Data] = {}
    for position, absolute_index in enumerate(sorted(indices), start=1):
        row = dataset[absolute_index]
        smiles = str(getattr(row, "smiles", "") or "")
        if not smiles or canonical(smiles) in excluded_connectivity:
            continue
        row.x = row.x.float()
        row.edge_attr = row.edge_attr.float()
        cached[int(absolute_index)] = row
        if position == 1 or position % 5000 == 0 or position == len(indices):
            print(
                f"预载入模拟样本 {position}/{len(indices)}，有效={len(cached)}",
                flush=True,
            )
    if len(cached) < 2:
        raise RuntimeError("预载入后有效模拟样本少于 2 条")
    return cached


def synthetic_batches(indices: list[int], block_size: int, batch_size: int, generator: torch.Generator):
    """块内洗牌，避免 Windows 下随机索引反复解压大 chunk。"""
    grouped: dict[int, list[int]] = defaultdict(list)
    for dataset_idx in indices:
        grouped[int(dataset_idx) // block_size].append(int(dataset_idx))
    blocks = list(grouped.values())
    block_order = torch.randperm(len(blocks), generator=generator).tolist()
    for block_number in block_order:
        block = blocks[block_number]
        local = torch.randperm(len(block), generator=generator).tolist()
        for offset in range(0, len(local), batch_size):
            batch = local[offset : offset + batch_size]
            if len(batch) >= 2:
                yield [block[value] for value in batch]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    started = time.time()
    available_real, domain_real_summary = load_real_records(args.real_dir)
    excluded_cas, excluded_connectivity, excluded_spectra = load_exclusion_keys(args.exclude_dir)
    real, exclusion_summary = exclude_real_records(
        available_real,
        excluded_cas,
        excluded_connectivity,
        excluded_spectra,
    )
    real_source_audit = build_real_source_audit(domain_real_summary, exclusion_summary)
    print(
        f"DARE 真实配对: available={len(available_real)}, retained={len(real)}, "
        f"excluded_cas={exclusion_summary['excluded_by_cas']}, "
        f"excluded_structure_only={exclusion_summary['excluded_by_connectivity_only']}, "
        f"excluded_spectrum_only={exclusion_summary['excluded_by_spectrum_only']}",
        flush=True,
    )
    for row in real_source_audit:
        print(
            f"真实来源 {row['source']}: input={row['input']}, "
            f"hac13={row['hac13_eligible_before_dedup']}, "
            f"duplicate={row['duplicate_spectrum']}, "
            f"frozen={row['excluded_by_cas'] + row['excluded_by_connectivity_only'] + row['excluded_by_spectrum_only']}, "
            f"retained={row['final_retained']}",
            flush=True,
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = split_scaffold_disjoint(
        real,
        args.val_fraction,
        args.seed,
    )
    if args.preflight_only:
        preflight_report = {
            "schema_version": 1,
            "mode": "preflight_only",
            "chemical_domain": domain_metadata(),
            "real_count": len(real),
            "real_domain_summary": domain_real_summary,
            "real_source_audit": real_source_audit,
            "train_count": len(train_indices),
            "validation_count": len(val_indices),
            "split_strategy": "scaffold_disjoint",
            "requested_validation_fraction": args.val_fraction,
            "actual_validation_fraction": len(val_indices) / len(real),
            "train_validation_scaffold_overlap": 0,
            "train_validation_connectivity_overlap": 0,
            "test_overlap_after_exclusion": 0,
            "excluded_cas_key_count": len(excluded_cas),
            "excluded_connectivity_key_count": len(excluded_connectivity),
            "excluded_spectrum_hash_count": len(excluded_spectra),
            "exclusion_summary": exclusion_summary,
            "parameters": {
                key: (
                    [str(item) for item in value]
                    if isinstance(value, list)
                    else str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in vars(args).items()
            },
            "elapsed_seconds": time.time() - started,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(preflight_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"预检完成: train={len(train_indices)}, val={len(val_indices)}, "
            "scaffold_overlap=0, connectivity_overlap=0",
            flush=True,
        )
        print(f"预检报告: {args.report.resolve()}", flush=True)
        return

    model, baseline, graph_model = load_models(
        args.initial_dare,
        args.initial_projection,
        args.graph_weights,
    )
    real_spectra = torch.stack([row["spectrum"] for row in real]).unsqueeze(1)
    real_graph_z = encode_graphs(graph_model, [row["graph"] for row in real])
    real_baseline_z = baseline_embeddings(baseline, real_spectra.to(DEVICE)).cpu()

    synthetic_dataset = PhysChemRADataset(
        root=str(args.synthetic_root),
        verbose=False,
        max_cached_chunks=2,
    )
    synthetic_indices, synthetic_index_summary = load_synthetic_indices(
        args.synthetic_index,
        len(synthetic_dataset),
    )
    if args.synthetic_count > len(synthetic_indices):
        raise ValueError(
            f"synthetic-count={args.synthetic_count} 超过闭域索引大小 {len(synthetic_indices)}"
        )
    if 0 < args.synthetic_count < len(synthetic_indices):
        order = torch.randperm(len(synthetic_indices), generator=generator)[: args.synthetic_count].tolist()
        synthetic_indices = sorted(synthetic_indices[index] for index in order)

    preloaded_synthetic: dict[int, Data] | None = None
    if args.preload_synthetic and (
        args.preload_max_samples <= 0 or len(synthetic_indices) <= args.preload_max_samples
    ):
        preloaded_synthetic = preload_synthetic_samples(
            synthetic_dataset,
            synthetic_indices,
            excluded_connectivity,
        )

    encoder_params = [p for p in model.encoder.feature_extractor.parameters() if p.requires_grad]
    projection_params = [p for p in model.projection_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr_encoder},
            {"params": projection_params, "lr": args.lr_projection},
        ],
        weight_decay=1e-5,
    )
    train_real = torch.tensor(train_indices, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        _, initial_val_feat, _ = model.encoder(real_spectra[val_indices].to(DEVICE))
        initial_val_z = F.normalize(
            model.projection_head(initial_val_feat),
            p=2,
            dim=-1,
        )
        initial_metrics = rank_metrics(
            initial_val_z,
            real_graph_z[val_indices].to(DEVICE),
        )
    initial_metrics.update({"epoch": 0, "loss": None})
    initial_key = (
        initial_metrics["top24"],
        initial_metrics["top5"],
        initial_metrics["top1"],
        -initial_metrics["mean_rank"],
    )
    best = {"key": initial_key, **initial_metrics}
    best_state = {
        "encoder": {
            key: value.detach().cpu().clone()
            for key, value in model.encoder.state_dict().items()
        },
        "projection": {
            key: value.detach().cpu().clone()
            for key, value in model.projection_head.state_dict().items()
        },
    }
    history = [{"split": "scaffold_validation_baseline", **initial_metrics}]
    print(
        "Epoch 00 基线 | "
        f"val Top1/5/24={initial_metrics['top1']:.3f}/"
        f"{initial_metrics['top5']:.3f}/{initial_metrics['top24']:.3f} | "
        f"rank={initial_metrics['mean_rank']:.2f}",
        flush=True,
    )
    synthetic_skipped = 0
    print(
        f"DARE 训练计划: epochs={args.epochs}, synthetic={len(synthetic_indices)}, "
        f"real_steps≈{max(1, len(train_indices) // max(args.real_batch_size, 1))}, "
        f"synthetic_steps/epoch≈{max(1, len(synthetic_indices) // max(args.synthetic_batch_size, 1))}, "
        f"block_size={args.block_size}, preload={preloaded_synthetic is not None}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        graph_model.eval()
        real_order = train_real[torch.randperm(train_real.numel(), generator=generator)]
        real_steps = []
        for offset in range(0, len(real_order), args.real_batch_size):
            batch = real_order[offset : offset + args.real_batch_size].tolist()
            if len(batch) >= 2:
                real_steps.append(batch)
        synth_steps = list(synthetic_batches(synthetic_indices, args.block_size, args.synthetic_batch_size, generator))
        steps = max(len(real_steps), len(synth_steps))
        losses = []
        for step in range(steps):
            real_idx = real_steps[step % len(real_steps)]
            synth_idx = synth_steps[step % len(synth_steps)]
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=DEVICE)

            real_spec = real_spectra[real_idx].to(DEVICE)
            _, real_feat, _ = model.encoder(real_spec)
            real_z = F.normalize(model.projection_head(real_feat), p=2, dim=-1)
            real_target = real_graph_z[real_idx].to(DEVICE)
            real_loss = symmetric_infonce(real_z, real_target, args.temperature)
            real_anchor = (1.0 - (real_z * real_baseline_z[real_idx].to(DEVICE)).sum(dim=-1)).mean()
            total = total + args.real_weight * (real_loss + args.anchor_weight * real_anchor)

            synth_spectra, synth_batch, skipped = fetch_batch(
                synthetic_dataset,
                synth_idx,
                excluded_connectivity,
                preloaded_synthetic,
            )
            synthetic_skipped += skipped
            synth_spectra = synth_spectra.to(DEVICE)
            synth_batch = synth_batch.to(DEVICE)
            with torch.no_grad():
                synth_graph_z, _ = graph_model(
                    synth_batch.x,
                    synth_batch.edge_index,
                    synth_batch.edge_attr,
                    synth_batch.batch,
                    return_pretrain=True,
                )
                synth_graph_z = F.normalize(synth_graph_z, p=2, dim=-1).clone().detach()
                synth_base_z = baseline_embeddings(baseline, synth_spectra).clone().detach()
            _, synth_feat, _ = model.encoder(synth_spectra)
            synth_z = F.normalize(model.projection_head(synth_feat), p=2, dim=-1)
            synth_loss = symmetric_infonce(synth_z, synth_graph_z, args.temperature)
            synth_anchor = (1.0 - (synth_z * synth_base_z).sum(dim=-1)).mean()
            total = total + args.synthetic_weight * (synth_loss + args.anchor_weight * synth_anchor)

            total.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params + projection_params, max_norm=1.0)
            optimizer.step()
            losses.append(float(total.item()))
            if (step + 1) % max(args.log_interval, 1) == 0 or step + 1 == steps:
                print(f"Epoch {epoch:02d}/{args.epochs} | step {step + 1}/{steps} | loss={losses[-1]:.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            _, val_feat, _ = model.encoder(real_spectra[val_indices].to(DEVICE))
            val_z = F.normalize(model.projection_head(val_feat), p=2, dim=-1)
            val_metrics = rank_metrics(val_z, real_graph_z[val_indices].to(DEVICE))
        val_metrics["epoch"] = epoch
        val_metrics["loss"] = float(np.mean(losses)) if losses else None
        history.append({"split": "scaffold_validation", **val_metrics})
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={val_metrics['loss']:.4f} | "
            f"val Top1/5/24={val_metrics['top1']:.3f}/{val_metrics['top5']:.3f}/{val_metrics['top24']:.3f} | "
            f"rank={val_metrics['mean_rank']:.2f}",
            flush=True,
        )
        key = (val_metrics["top24"], val_metrics["top5"], val_metrics["top1"], -val_metrics["mean_rank"])
        if key > best["key"]:
            best = {"key": key, **val_metrics}
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()},
                "projection": {k: v.detach().cpu().clone() for k, v in model.projection_head.state_dict().items()},
            }

    args.encoder_output.parent.mkdir(parents=True, exist_ok=True)
    args.projection_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "method": "dare_spectrum_alignment_domain_hac13",
            "state_dict": best_state["encoder"],
            "chemical_domain": domain_metadata(),
            "synthetic_index": str(args.synthetic_index.resolve()),
            "report": str(args.report.resolve()),
        },
        args.encoder_output,
    )
    torch.save(
        {
            "schema_version": 1,
            "method": "dare_spectrum_alignment_projection_domain_hac13",
            "state_dict": best_state["projection"],
            "chemical_domain": domain_metadata(),
            "synthetic_index": str(args.synthetic_index.resolve()),
            "report": str(args.report.resolve()),
        },
        args.projection_output,
    )
    report = {
        "schema_version": 1,
        "device": str(DEVICE),
        "real_count": len(real),
        "synthetic_count": len(synthetic_indices),
        "synthetic_index": str(args.synthetic_index.resolve()),
        "synthetic_index_summary": synthetic_index_summary,
        "chemical_domain": domain_metadata(),
        "real_domain_summary": domain_real_summary,
        "real_source_audit": real_source_audit,
        "train_count": len(train_indices),
        "validation_count": len(val_indices),
        "split_strategy": "scaffold_disjoint",
        "requested_validation_fraction": args.val_fraction,
        "actual_validation_fraction": len(val_indices) / len(real),
        "train_validation_scaffold_overlap": 0,
        "train_validation_connectivity_overlap": 0,
        "test_overlap_after_exclusion": 0,
        "excluded_cas_key_count": len(excluded_cas),
        "excluded_connectivity_key_count": len(excluded_connectivity),
        "excluded_spectrum_hash_count": len(excluded_spectra),
        "exclusion_summary": exclusion_summary,
        "synthetic_excluded_rows_skipped_across_epochs": synthetic_skipped,
        "initial_projection": str(args.initial_projection.resolve()),
        "initial_dare": str(args.initial_dare.resolve()),
        "graph_weights": str(args.graph_weights.resolve()),
        "encoder_output": str(args.encoder_output.resolve()),
        "projection_output": str(args.projection_output.resolve()),
        "parameters": {
            key: (
                [str(item) for item in value]
                if isinstance(value, list)
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "best_validation": {
            **{key: value for key, value in best.items() if key != "key"},
            "selection_key": list(best["key"]),
        },
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DARE 微调编码器: {args.encoder_output.resolve()}")
    print(f"DARE 微调投影头: {args.projection_output.resolve()}")
    print(f"报告: {args.report.resolve()}")


if __name__ == "__main__":
    main()

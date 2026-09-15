"""训练无 PubChem 顺序依赖的光谱-分子图对比学习双塔。

训练目标是光谱与真实分子图在同一嵌入空间中的对齐。候选检索阶段只使用
双塔余弦相似度
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physchemrag.config import (  # noqa: E402
    DATA_DIR,
    MODULE1_DARE_WEIGHTS,
    MODULE2_DARE_FUSION_WEIGHTS,
    MODULE2_GRAPH_WEIGHTS,
    OUTPUT_REPORTS_DIR,
    WEIGHTS_DIR,
)
from physchemrag.module4_cascade.learned_cascade import (  # noqa: E402
    SpectrumGraphContrastiveModel,
    symmetric_infonce_loss,
)
from physchemrag.shared.dataset import PhysChemRADataset  # noqa: E402
from physchemrag.shared.molecular_formula import canonical_smiles, formula_from_smiles  # noqa: E402
from physchemrag.shared.chemical_domain import (  # noqa: E402
    DOMAIN_NAME,
    domain_metadata,
    domain_metadata_matches,
    inspect_smiles_domain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--structure-index",
        type=Path,
        default=DATA_DIR
        / "conformers_v2"
        / "training_index_decontaminated_all_external_domain_hac13.jsonl",
        help="按连接结构切分使用的模拟索引；正式外部评测应使用去污染索引",
    )
    parser.add_argument(
        "--exclude-dir",
        type=Path,
        nargs="*",
        default=[],
        help="从模拟结构切分中排除的真实谱目录，可传入多个目录",
    )
    parser.add_argument("--dare-weights", type=Path, default=MODULE1_DARE_WEIGHTS)
    parser.add_argument("--graph-init-weights", type=Path, default=MODULE2_GRAPH_WEIGHTS, help="现有图塔预训练权重；传空路径可随机初始化")
    parser.add_argument(
        "--no-graph-init",
        action="store_true",
        help="不加载图塔初始权重，从随机初始化开始；用于严格去污染主线",
    )
    parser.add_argument("--spectrum-init-weights", type=Path, default=MODULE2_DARE_FUSION_WEIGHTS, help="现有 DARE 投影头权重")
    parser.add_argument(
        "--freeze-spectrum-tower",
        action="store_true",
        help="冻结已验证的 DARE 编码器与投影头，只训练去污染图塔和温度参数",
    )
    parser.add_argument("--output-weights", type=Path, default=WEIGHTS_DIR / "module4_spectrum_graph_contrastive_domain_hac13_best.pth")
    parser.add_argument("--report", type=Path, default=OUTPUT_REPORTS_DIR / "module4_spectrum_graph_contrastive_domain_hac13_training.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=50000, help="按原始 chunk 大小分组读取样本，减少 Windows 磁盘随机访问")
    parser.add_argument(
        "--chunk-cache-size",
        type=int,
        default=16,
        help="内存中保留的数据 chunk 数；本项目 16 个 chunk 约需数十 GiB RAM",
    )
    parser.add_argument(
        "--batch-sampler",
        choices=("formula_balanced", "chunk"),
        default="formula_balanced",
        help="训练批次采样策略；formula_balanced 强制每批包含同分子式异构体",
    )
    parser.add_argument(
        "--split-group-level",
        choices=("structure", "formula"),
        default="formula",
        help="训练/验证切分粒度；formula 默认用于严格分子式 OOD 审计，structure 仅用于旧版诊断",
    )
    parser.add_argument("--log-interval", type=int, default=25, help="每隔多少个 batch 打印一次进度")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 上启用 bfloat16/float16 混合精度；可用 --no-amp 关闭",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader 后台进程数；启用全量 chunk 预载入时必须为 0",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="每个 DataLoader worker 预取的 batch 数",
    )
    parser.add_argument(
        "--preload-all-chunks",
        action="store_true",
        help="训练前将全部 processed chunk 载入内存；本机有足够内存时显著减少 GPU 等待",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=0.20)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--max-train-samples", type=int, default=100000, help="每轮使用的训练样本上限；0 表示使用全部训练集")
    parser.add_argument("--max-val-samples", type=int, default=20000, help="验证样本上限；0 表示使用全部验证集")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--strict-smoke",
        action="store_true",
        help="smoke 仍限制训练/验证样本，但先完整解析结构索引并应用去污染排除；用于验证排除逻辑",
    )
    parser.add_argument(
        "--include-unindexed",
        action="store_true",
        help="将不在 structure-index 中的样本放回训练切分；严格去污染训练默认不启用",
    )
    parser.add_argument(
        "--disable-domain-filter",
        action="store_true",
        help="关闭闭域结构门控，仅用于复现旧版训练",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(items):
    """将 PyG 图、光谱和分子式组成一个批次。"""
    graph_batch = Batch.from_data_list(items)
    spectra = torch.stack([item.spectrum.float() for item in items])
    formulas = []
    structures = []
    for item in items:
        # Data 对象随 chunk 缓存复用；避免每个 epoch 重复执行 RDKit 解析。
        formula = getattr(item, "_formula_cache", None)
        structure = getattr(item, "_structure_cache", None)
        if formula is None:
            formula = formula_from_smiles(str(item.smiles)) or ""
            item._formula_cache = formula
        if structure is None:
            structure = canonical_smiles(str(item.smiles), isomeric=False) or ""
            item._structure_cache = structure
        formulas.append(formula)
        structures.append(structure)
    return graph_batch, spectra, formulas, structures


def preload_all_chunks(dataset: PhysChemRADataset) -> None:
    """将全部约 600 MiB 的数据 chunk 载入 LRU，避免每个 batch 反序列化。"""
    chunk_files = list(getattr(dataset, "_chunk_files", []))
    if not chunk_files:
        raise RuntimeError("数据集没有可预载入的 processed chunk")
    cache = getattr(dataset, "_chunk_cache", None)
    if cache is None:
        raise RuntimeError("数据集缺少 chunk 缓存实现")
    print(f"开始预载入 {len(chunk_files)} 个数据 chunk（约 9.3 GiB）...", flush=True)
    started = time.perf_counter()
    for chunk_index, chunk_file in enumerate(chunk_files):
        chunk_path = Path(dataset.processed_dir) / chunk_file
        if chunk_index not in cache:
            cache[chunk_index] = torch.load(chunk_path, weights_only=False)
        if (chunk_index + 1) == 1 or (chunk_index + 1) % 2 == 0 or chunk_index + 1 == len(chunk_files):
            print(
                f"预载入 chunk {chunk_index + 1}/{len(chunk_files)}；"
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    dataset.max_cached_chunks = max(int(dataset.max_cached_chunks), len(chunk_files))
    print(
        f"全部 chunk 已在内存中，耗时 {time.perf_counter() - started:.1f}s；"
        f"cache={len(cache)}",
        flush=True,
    )


def limit_indices(
    indices: list[int], limit: int, seed: int, *, preserve_locality: bool = False
) -> list[int]:
    if limit <= 0 or len(indices) <= limit:
        return list(indices)
    if preserve_locality:
        ordered = sorted(indices)
        start = random.Random(seed).randrange(0, len(ordered) - limit + 1)
        return ordered[start : start + limit]
    order = list(indices)
    random.Random(seed).shuffle(order)
    return order[:limit]


def load_excluded_structures(directories: list[Path]) -> set[str]:
    """读取外部真实谱目录中的连接结构，用于模拟训练去污染。"""
    excluded: set[str] = set()
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            smiles = str(payload.get("connectivity_smiles") or payload.get("smiles") or "")
            key = canonical_smiles(smiles, isomeric=False)
            if key:
                excluded.add(key)
    return excluded


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


def build_filtered_structure_split(
    dataset_length: int,
    structure_index: Path,
    excluded_structures: set[str],
    val_fraction: float,
    seed: int,
    include_unindexed: bool = False,
    apply_domain_filter: bool = True,
    split_group_level: str = "structure",
) -> tuple[list[int], list[int], dict[str, int], dict[str, list[int]]]:
    """按连接结构或分子式分组切分，并过滤外部结构。"""
    if split_group_level not in {"structure", "formula"}:
        raise ValueError(f"未知切分粒度: {split_group_level}")
    if split_group_level == "formula" and include_unindexed:
        raise ValueError("公式级切分不能包含未索引样本：其分子式未经过结构索引审计")
    if not structure_index.is_file():
        raise FileNotFoundError(f"模拟结构索引不存在: {structure_index}")
    groups: dict[str, list[int]] = defaultdict(list)
    seen = bytearray(dataset_length)
    parsed = 0
    excluded_rows = 0
    domain_rejections: Counter[str] = Counter()
    formula_groups: dict[str, list[int]] = defaultdict(list)
    dataset_formulas: dict[int, str] = {}
    strict_decontaminated_index = False
    with structure_index.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header.strip():
            raise ValueError("模拟结构索引缺少头信息")
        try:
            header_payload = json.loads(header)
        except json.JSONDecodeError as exc:
            raise ValueError("模拟结构索引首行不是有效 JSON 元数据") from exc
        strict_decontaminated_index = bool(header_payload.get("decontamination"))
        if apply_domain_filter and not domain_metadata_matches(
            header_payload.get("chemical_domain")
        ):
            raise ValueError(
                f"模拟结构索引缺少一致的 {DOMAIN_NAME} 元数据: {structure_index}"
            )
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("dataset_idx") is None:
                continue
            dataset_idx = int(payload["dataset_idx"])
            if not 0 <= dataset_idx < dataset_length:
                raise ValueError(f"模拟索引第 {line_number} 行 dataset_idx 越界: {dataset_idx}")
            smiles = str(payload.get("smiles") or "")
            key = canonical_smiles(smiles, isomeric=False)
            if not key:
                key = f"invalid:{dataset_idx}"
            seen[dataset_idx] = 1
            parsed += 1
            if key in excluded_structures:
                excluded_rows += 1
                continue
            if apply_domain_filter:
                accepted, reasons = domain_result(
                    inspect_smiles_domain(
                        smiles,
                        require_single_component=True,
                        require_neutral_closed_shell=True,
                    )
                )
                if not accepted:
                    for reason in reasons:
                        domain_rejections[reason] += 1
                    continue
            groups[key].append(dataset_idx)
            formula = formula_from_smiles(smiles) or ""
            if formula:
                formula_groups[formula].append(dataset_idx)
                dataset_formulas[dataset_idx] = formula
    unindexed = [dataset_idx for dataset_idx, was_seen in enumerate(seen) if not was_seen]
    if include_unindexed:
        if apply_domain_filter:
            raise ValueError(
                "闭域训练不允许 --include-unindexed：未索引样本缺少可审计结构，"
                "无法验证元素、重原子数及中性单组分约束"
            )
        if strict_decontaminated_index:
            raise ValueError(
                "严格去污染结构索引不允许 --include-unindexed：索引缺失行代表已排除样本"
            )
        for dataset_idx in unindexed:
            groups[f"unindexed:{dataset_idx}"].append(dataset_idx)
    if not groups:
        raise RuntimeError("去污染后没有可用模拟结构组")
    if split_group_level == "formula":
        # 严格分子式 OOD 切分：同一分子式的所有连接结构只能进入一侧。
        formula_to_indices: dict[str, list[int]] = defaultdict(list)
        retained_indices = [index for rows in groups.values() for index in rows]
        for dataset_idx in retained_indices:
            formula = dataset_formulas.get(dataset_idx, "")
            if formula:
                formula_to_indices[formula].append(dataset_idx)
        group_rows = list(formula_to_indices.values())
    else:
        group_rows = list(groups.values())
    random.Random(seed).shuffle(group_rows)
    retained_length = sum(len(row) for row in group_rows)
    target_val_size = max(1, round(retained_length * val_fraction))
    remaining = target_val_size
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in group_rows:
        if len(indices) <= remaining:
            val_indices.extend(indices)
            remaining -= len(indices)
        else:
            train_indices.extend(indices)
    if remaining:
        raise RuntimeError(f"无法在结构隔离约束下补足验证集，缺少 {remaining} 个样本")
    train_indices.sort()
    val_indices.sort()
    train_formula_set = {
        dataset_formulas[index] for index in train_indices if dataset_formulas.get(index)
    }
    val_formula_set = {
        dataset_formulas[index] for index in val_indices if dataset_formulas.get(index)
    }
    split_summary = {
        "index_rows": parsed,
        "structure_groups": len(group_rows),
        "split_group_level": split_group_level,
        "excluded_structures": len(excluded_structures),
        "excluded_index_rows": excluded_rows,
        "retained_samples": retained_length,
        "unindexed_samples": len(unindexed),
        "unindexed_included": bool(include_unindexed),
        "strict_decontaminated_index": strict_decontaminated_index,
        "domain_name": DOMAIN_NAME,
        "domain_filter_enabled": apply_domain_filter,
        "domain_rejections": dict(domain_rejections),
        "train_formula_count": len(train_formula_set),
        "validation_formula_count": len(val_formula_set),
        "formula_overlap_count": len(train_formula_set & val_formula_set),
    }
    if split_group_level == "formula" and split_summary["formula_overlap_count"]:
        raise RuntimeError("公式级切分出现 train/val 分子式重叠")
    return train_indices, val_indices, split_summary, dict(formula_groups)


class ChunkBlockBatchSampler(Sampler[list[int]]):
    """按数据集索引局部成块并批量采样，避免随机洗牌反复加载全部 chunk。"""

    def __init__(self, indices: list[int], batch_size: int, block_size: int, seed: int, shuffle: bool) -> None:
        if batch_size < 2 or block_size < batch_size:
            raise ValueError("batch_size 至少为 2，且 block_size 不得小于 batch_size")
        grouped: dict[int, list[int]] = {}
        for index in sorted(int(index) for index in indices):
            grouped.setdefault(index // block_size, []).append(index)
        self.blocks = list(grouped.values())
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        blocks = [list(block) for block in self.blocks]
        generator = random.Random(self.seed + self.epoch)
        if self.shuffle:
            generator.shuffle(blocks)
        carry: list[int] = []
        for block in blocks:
            if self.shuffle:
                generator.shuffle(block)
            # 跨 block 携带尾部索引，避免小样本 smoke 或稀疏抽样时 batch 数为 0。
            entries = carry + block
            full_size = (len(entries) // self.batch_size) * self.batch_size
            for start in range(0, full_size, self.batch_size):
                yield entries[start : start + self.batch_size]
            carry = entries[full_size:]
        if len(carry) >= 2:
            yield carry

    def __len__(self) -> int:
        total = sum(len(block) for block in self.blocks)
        return total // self.batch_size + int(total % self.batch_size >= 2)


class FormulaBalancedBatchSampler(Sampler[list[int]]):
    """覆盖全部索引，并兼顾同分子式困难负样本与磁盘 chunk 局部性。"""

    def __init__(
        self,
        indices: list[int],
        formula_groups: dict[str, list[int]],
        batch_size: int,
        block_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        if batch_size < 2 or block_size < batch_size:
            raise ValueError("batch_size 至少为 2，且 block_size 不得小于 batch_size")
        allowed = set(int(index) for index in indices)
        self.indices = sorted(allowed)
        self.groups = {
            str(formula): sorted(set(int(index) for index in rows).intersection(allowed))
            for formula, rows in formula_groups.items()
        }
        self.groups = {formula: rows for formula, rows in self.groups.items() if len(rows) >= 2}
        if not self.groups:
            raise ValueError("没有至少含两个样本的同分子式训练组")
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _chunk_id(self, index: int) -> int:
        return int(index) // self.block_size

    def _paired_seeds(self, rng: random.Random) -> tuple[list[list[int]], list[list[int]]]:
        """优先产生同一磁盘 chunk 内的同分子式样本对。"""
        local_pairs: list[list[int]] = []
        cross_chunk_pairs: list[list[int]] = []
        formulas = list(self.groups)
        if self.shuffle:
            rng.shuffle(formulas)
        for formula in formulas:
            rows_by_chunk: dict[int, list[int]] = defaultdict(list)
            for index in self.groups[formula]:
                rows_by_chunk[self._chunk_id(index)].append(index)
            leftovers: list[int] = []
            for rows in rows_by_chunk.values():
                if self.shuffle:
                    rng.shuffle(rows)
                pair_end = len(rows) - len(rows) % 2
                local_pairs.extend([rows[start : start + 2] for start in range(0, pair_end, 2)])
                leftovers.extend(rows[pair_end:])
            if self.shuffle:
                rng.shuffle(leftovers)
            pair_end = len(leftovers) - len(leftovers) % 2
            cross_chunk_pairs.extend(
                [leftovers[start : start + 2] for start in range(0, pair_end, 2)]
            )
        if self.shuffle:
            rng.shuffle(local_pairs)
            rng.shuffle(cross_chunk_pairs)
        return local_pairs, cross_chunk_pairs

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batch_count = max(1, (len(self.indices) + self.batch_size - 1) // self.batch_size)
        local_pairs, cross_chunk_pairs = self._paired_seeds(rng)
        pair_seeds = local_pairs + cross_chunk_pairs
        if len(pair_seeds) < batch_count:
            raise ValueError(
                "同分子式样本对不足，无法在覆盖全部样本时保证每个批次含困难负样本: "
                f"pairs={len(pair_seeds)}, batches={batch_count}"
            )

        # 每批先放入一个同式样本对，再尽量使用相同物理 chunk 的样本填满。
        batches = [list(pair) for pair in pair_seeds[:batch_count]]
        assigned = {index for batch in batches for index in batch}
        remaining_by_chunk: dict[int, list[int]] = defaultdict(list)
        for index in self.indices:
            if index not in assigned:
                remaining_by_chunk[self._chunk_id(index)].append(index)
        chunk_order = list(remaining_by_chunk)
        if self.shuffle:
            rng.shuffle(chunk_order)
            for rows in remaining_by_chunk.values():
                rng.shuffle(rows)

        batch_chunks = [set(self._chunk_id(index) for index in batch) for batch in batches]
        for chunk_id in chunk_order:
            rows = remaining_by_chunk[chunk_id]
            candidates = [
                batch_index
                for batch_index, chunks in enumerate(batch_chunks)
                if chunk_id in chunks and len(batches[batch_index]) < self.batch_size
            ]
            if self.shuffle:
                rng.shuffle(candidates)
            candidates.extend(
                sorted(
                    (
                        batch_index
                        for batch_index in range(batch_count)
                        if batch_index not in candidates and len(batches[batch_index]) < self.batch_size
                    ),
                    key=lambda batch_index: (len(batch_chunks[batch_index]), -len(batches[batch_index])),
                )
            )
            offset = 0
            for batch_index in candidates:
                if offset >= len(rows):
                    break
                available = self.batch_size - len(batches[batch_index])
                take = min(available, len(rows) - offset)
                if take <= 0:
                    continue
                batches[batch_index].extend(rows[offset : offset + take])
                batch_chunks[batch_index].add(chunk_id)
                offset += take
            if offset != len(rows):
                raise RuntimeError("formula_balanced 批次容量不足")

        if sorted(index for batch in batches for index in batch) != self.indices:
            raise RuntimeError("formula_balanced 未能恰好覆盖全部训练索引")
        if any(not 2 <= len(batch) <= self.batch_size for batch in batches):
            raise RuntimeError("formula_balanced 产生了非法批次大小")

        # 同一主 chunk 的 batch 连续执行；每批先访问少量辅 chunk，最后保留主 chunk。
        primary_chunks = [
            Counter(self._chunk_id(index) for index in batch).most_common(1)[0][0]
            for batch in batches
        ]
        grouped_batches: dict[int, list[int]] = defaultdict(list)
        for batch_index, primary_chunk in enumerate(primary_chunks):
            grouped_batches[primary_chunk].append(batch_index)
        ordered_chunks = list(grouped_batches)
        if self.shuffle:
            rng.shuffle(ordered_chunks)
            for batch_indices in grouped_batches.values():
                rng.shuffle(batch_indices)
        ordered_batches: list[list[int]] = []
        for primary_chunk in ordered_chunks:
            for batch_index in grouped_batches[primary_chunk]:
                batch = batches[batch_index]
                ordered_batches.append(
                    sorted(
                        batch,
                        key=lambda index: (
                            self._chunk_id(index) == primary_chunk,
                            self._chunk_id(index),
                            index,
                        ),
                    )
                )
        return ordered_batches

    def locality_summary(self) -> dict[str, float | int]:
        batches = self._batches()
        chunk_counts = [len({self._chunk_id(index) for index in batch}) for batch in batches]
        return {
            "batches": len(batches),
            "mean_chunks_per_batch": float(np.mean(chunk_counts)),
            "max_chunks_per_batch": max(chunk_counts),
            "single_chunk_batches": sum(count == 1 for count in chunk_counts),
        }

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        return max(1, (len(self.indices) + self.batch_size - 1) // self.batch_size)


def load_initial_weights(model: SpectrumGraphContrastiveModel, graph_path: Path | None, spectrum_path: Path | None) -> dict[str, str | None]:
    """加载现有跨模态权重作为 warm start；架构不匹配时立即报错。"""
    used: dict[str, str | None] = {"graph": None, "spectrum": None}
    if graph_path is not None and graph_path.is_file():
        graph_state = torch.load(graph_path, map_location="cpu", weights_only=False)
        if isinstance(graph_state, dict) and "model_state_dict" in graph_state:
            graph_state = graph_state["model_state_dict"]
        model.graph_tower.load_state_dict(graph_state, strict=True)
        used["graph"] = str(graph_path.resolve())
    if spectrum_path is not None and spectrum_path.is_file():
        spectrum_state = torch.load(spectrum_path, map_location="cpu", weights_only=False)
        if isinstance(spectrum_state, dict) and "model_state_dict" in spectrum_state:
            spectrum_state = spectrum_state["model_state_dict"]
        if isinstance(spectrum_state, dict):
            for wrapper_key in ("state_dict", "projection_state_dict"):
                nested = spectrum_state.get(wrapper_key)
                if isinstance(nested, dict):
                    spectrum_state = nested
                    break
        model.spectrum_tower.projection_head.load_state_dict(spectrum_state, strict=True)
        used["spectrum"] = str(spectrum_path.resolve())
    return used


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """同时报告批内指标和完整验证集同分子式池指标。"""
    model.eval()
    loss_sum = 0.0
    samples = 0
    correct1 = 0
    correct5 = 0
    spectrum_parts: list[torch.Tensor] = []
    molecule_parts: list[torch.Tensor] = []
    formula_parts: list[str] = []
    structure_parts: list[str] = []
    for graph_batch, spectra, formulas, structures in loader:
        graph_batch = graph_batch.to(device, non_blocking=True)
        spectra = spectra.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            z_spec, z_mol = model(graph_batch, spectra)
            loss, _ = symmetric_infonce_loss(
                z_spec,
                z_mol,
                model.logit_scale,
                formulas,
                structure_ids=structures,
            )
        batch_size = int(spectra.shape[0])
        logits = model.logit_scale.exp().float().clamp(1.0, 100.0) * (
            z_spec.float() @ z_mol.float().transpose(0, 1)
        )
        pred1 = logits.argmax(dim=1)
        pred5 = logits.topk(min(5, batch_size), dim=1).indices
        # predicted_structures: [B] / [B,5]；同一连接结构的任意副本均判为正确。
        correct1 += sum(structures[int(index)] == structures[row] for row, index in enumerate(pred1.tolist()))
        correct5 += sum(
            any(structures[index] == structures[row] for index in candidate_rows)
            for row, candidate_rows in enumerate(pred5.tolist())
        )
        loss_sum += float(loss.detach().item()) * batch_size
        samples += batch_size
        spectrum_parts.append(z_spec.detach().cpu().float())
        molecule_parts.append(z_mol.detach().cpu().float())
        formula_parts.extend(str(value) for value in formulas)
        structure_parts.extend(str(value) for value in structures)
    same_top1 = same_top5 = 0
    same_queries = 0
    if spectrum_parts:
        spectra_all = torch.cat(spectrum_parts, dim=0)
        molecules_all = torch.cat(molecule_parts, dim=0)
        formula_to_rows: dict[str, list[int]] = defaultdict(list)
        for row, formula in enumerate(formula_parts):
            formula_to_rows[formula].append(row)
        for rows in formula_to_rows.values():
            candidate_by_structure: dict[str, int] = {}
            for row in rows:
                candidate_by_structure.setdefault(structure_parts[row], row)
            if len(candidate_by_structure) < 2:
                continue
            row_tensor = torch.tensor(rows, dtype=torch.long)
            candidate_rows = list(candidate_by_structure.values())
            candidate_tensor = torch.tensor(candidate_rows, dtype=torch.long)
            # [Q,D] @ [D,C] -> [Q,C]；C 是该分子式下唯一连接结构候选数。
            logits = spectra_all[row_tensor] @ molecules_all[candidate_tensor].transpose(0, 1)
            target_by_structure = {
                structure: index for index, structure in enumerate(candidate_by_structure)
            }
            labels = torch.tensor(
                [target_by_structure[structure_parts[row]] for row in rows],
                dtype=torch.long,
            )
            same_top1 += int((logits.argmax(dim=1) == labels).sum().item())
            same_top5 += int(
                (logits.topk(min(5, len(candidate_rows)), dim=1).indices == labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )
            same_queries += len(rows)
    return {
        "loss": loss_sum / max(samples, 1),
        "top1": correct1 / max(samples, 1),
        "top5": correct5 / max(samples, 1),
        "samples": float(samples),
        "same_formula_top1": same_top1 / max(same_queries, 1),
        "same_formula_top5": same_top5 / max(same_queries, 1),
        "same_formula_queries": float(same_queries),
    }


def main() -> None:
    args = parse_args()
    apply_domain_filter = not args.disable_domain_filter
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 8)
        args.block_size = max(args.batch_size, min(args.block_size, 32))
        args.log_interval = 1
        args.max_train_samples = args.max_train_samples or 32
        args.max_val_samples = args.max_val_samples or 32
    if (
        args.epochs < 1
        or args.batch_size < 2
        or args.chunk_cache_size < 1
        or args.num_workers < 0
        or args.prefetch_factor < 1
    ):
        raise ValueError(
            "epochs 必须为正，batch-size 至少为 2，chunk-cache-size 至少为 1，"
            "num-workers 不得为负，prefetch-factor 至少为 1"
        )
    if args.preload_all_chunks and args.num_workers != 0:
        raise ValueError("--preload-all-chunks 必须配合 --num-workers 0，避免 Windows worker 复制内存")
    if args.no_graph_init:
        args.graph_init_weights = None
    if args.freeze_spectrum_tower and not args.spectrum_init_weights.is_file():
        raise FileNotFoundError(
            "--freeze-spectrum-tower 要求 --spectrum-init-weights 指向已验证投影头"
        )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()

    # 小型 LRU 缓存配合局部 batch 采样，避免随机访问反复反序列化约 600 MiB 的 chunk。
    dataset = PhysChemRADataset(
        root=str(args.data_root),
        verbose=False,
        max_cached_chunks=args.chunk_cache_size,
    )
    if args.preload_all_chunks:
        preload_all_chunks(dataset)
    excluded_structures = load_excluded_structures(args.exclude_dir)
    if args.smoke and not args.strict_smoke and not apply_domain_filter:
        # smoke 只验证数据管线和前向形状，避免为小样本测试扫描完整结构索引文件。
        probe_indices = list(range(min(len(dataset), max(args.max_train_samples + args.max_val_samples, 8))))
        random.Random(args.seed).shuffle(probe_indices)
        split = max(2, len(probe_indices) // 2)
        train_indices, val_indices = probe_indices[split:], probe_indices[:split]
    else:
        train_indices, val_indices, split_summary, formula_groups = build_filtered_structure_split(
            len(dataset),
            args.structure_index,
            excluded_structures,
            args.val_fraction,
            args.seed,
            args.include_unindexed,
            apply_domain_filter,
            args.split_group_level,
        )
    if args.smoke and not args.strict_smoke and not apply_domain_filter:
        split_summary = {
            "index_rows": 0,
            "structure_groups": 0,
            "excluded_structures": len(excluded_structures),
            "excluded_index_rows": 0,
            "retained_samples": len(train_indices) + len(val_indices),
            "smoke_split": True,
            "exclusion_applied": False,
        }
        formula_groups = {}
    elif args.smoke:
        split_summary["smoke_split"] = True
        split_summary["exclusion_applied"] = True
        split_summary["split_group_level"] = args.split_group_level
    train_indices = limit_indices(
        train_indices,
        args.max_train_samples,
        args.seed + 1,
        preserve_locality=bool(args.smoke and args.strict_smoke),
    )
    val_indices = limit_indices(
        val_indices,
        args.max_val_samples,
        args.seed + 2,
        preserve_locality=bool(args.smoke and args.strict_smoke),
    )
    if len(train_indices) < 2 or len(val_indices) < 2:
        raise RuntimeError("训练/验证样本不足")

    if args.batch_sampler == "formula_balanced" and formula_groups:
        try:
            train_sampler = FormulaBalancedBatchSampler(
                train_indices,
                formula_groups,
                args.batch_size,
                args.block_size,
                args.seed,
                shuffle=True,
            )
            # 提前构造一次，确保样本上限应用后仍能满足每批至少一个同式对。
            train_sampler._batches()
        except ValueError:
            if not args.smoke:
                raise
            print("smoke 子集没有足够同分子式样本对，训练退回 chunk sampler。", flush=True)
            train_sampler = ChunkBlockBatchSampler(
                train_indices, args.batch_size, args.block_size, args.seed, shuffle=True
            )
        # 验证必须覆盖完整 val；同式指标在 evaluate 中按完整分子式组另行计算。
        val_sampler = ChunkBlockBatchSampler(
            val_indices, args.batch_size, args.block_size, args.seed + 1, shuffle=False
        )
        split_summary["formula_balanced_train_samples"] = int(
            len(train_indices) if isinstance(train_sampler, FormulaBalancedBatchSampler) else 0
        )
        split_summary["formula_balanced_train_full_coverage"] = bool(
            isinstance(train_sampler, FormulaBalancedBatchSampler)
        )
        split_summary["validation_full_coverage"] = True
    else:
        train_sampler = ChunkBlockBatchSampler(train_indices, args.batch_size, args.block_size, args.seed, shuffle=True)
        val_sampler = ChunkBlockBatchSampler(val_indices, args.batch_size, args.block_size, args.seed + 1, shuffle=False)
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    model = SpectrumGraphContrastiveModel(dare_weights=str(args.dare_weights)).to(device)
    initialization = load_initial_weights(model, args.graph_init_weights, args.spectrum_init_weights)
    if args.freeze_spectrum_tower:
        for parameter in model.spectrum_tower.parameters():
            parameter.requires_grad_(False)
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_key = None
    best_validation = None
    history = []
    print(
        f"对比学习训练 train/val={len(train_indices)}/{len(val_indices)} "
        f"batch={args.batch_size} device={device}；不使用 PubChem rank；"
        f"去污染结构={len(excluded_structures)}；batch_sampler={args.batch_sampler}；"
        f"freeze_spectrum={args.freeze_spectrum_tower}；"
        f"amp={amp_enabled}({amp_dtype})；workers={args.num_workers}；"
        f"prefetch={args.prefetch_factor}",
        flush=True,
    )
    print(
        f"训练 batch 数/epoch={len(train_loader)}，验证 batch 数={len(val_loader)}；"
        "正在加载第一个 chunk 和第一个 batch...",
        flush=True,
    )
    if isinstance(train_sampler, FormulaBalancedBatchSampler):
        locality = train_sampler.locality_summary()
        print(
            "训练 batch chunk 局部性: "
            f"mean={locality['mean_chunks_per_batch']:.2f}, "
            f"max={locality['max_chunks_per_batch']}, "
            f"single={locality['single_chunk_batches']}/{locality['batches']}, "
            f"cache={args.chunk_cache_size}",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_spectrum_tower:
            # 冻结参数时同时关闭投影头 Dropout，保持真实域坐标系确定不变。
            model.spectrum_tower.eval()
        train_sampler.set_epoch(epoch)
        losses = []
        train_iterator = iter(train_loader)
        for step in range(1, len(train_loader) + 1):
            load_started = time.perf_counter()
            graph_batch, spectra, formulas, structures = next(train_iterator)
            load_elapsed = time.perf_counter() - load_started
            compute_started = time.perf_counter()
            if epoch == 1 and step == 1:
                print(
                    "第一个 batch 已加载，开始 GPU 前向和反向计算；"
                    f"batch={len(formulas)}，nodes={int(graph_batch.num_nodes)}，"
                    f"edges={int(graph_batch.num_edges)}，数据加载={load_elapsed:.2f}s",
                    flush=True,
                )
            graph_batch = graph_batch.to(device, non_blocking=True)
            spectra = spectra.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                z_spec, z_mol = model(graph_batch, spectra)
                loss, _ = symmetric_infonce_loss(
                    z_spec,
                    z_mol,
                    model.logit_scale,
                    formulas,
                    structure_ids=structures,
                    hard_negative_weight=args.hard_negative_weight,
                    hard_negative_margin=args.hard_negative_margin,
                )
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            if epoch == 1 and step == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                print(
                    "第一个 batch GPU 前向/反向完成；"
                    f"计算={time.perf_counter() - compute_started:.2f}s，"
                    f"总计={time.perf_counter() - load_started:.2f}s",
                    flush=True,
                )
            losses.append(float(loss.detach().item()))
            if step == 1 or step % max(1, args.log_interval) == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss={losses[-1]:.4f}", flush=True)
        validation = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "validation": validation,
            "temperature": float((1.0 / model.logit_scale.exp().clamp(1.0, 100.0)).detach().item()),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} loss={row['train_loss']:.4f} "
            f"val batchTop1/5={validation['top1']:.3f}/{validation['top5']:.3f} "
            f"sameFormula Top1/5={validation['same_formula_top1']:.3f}/{validation['same_formula_top5']:.3f} "
            f"temp={row['temperature']:.4f}",
            flush=True,
        )
        same_formula_available = validation["same_formula_queries"] > 0
        validation_key = (
            int(same_formula_available),
            validation["same_formula_top1"],
            validation["same_formula_top5"],
            validation["top1"],
            validation["top5"],
            -validation["loss"],
        )
        if best_key is None or validation_key > best_key:
            best_key = validation_key
            best_validation = dict(validation)
            args.output_weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "method": "spectrum_graph_contrastive_no_pubchem_rank",
                    "model_state_dict": model.state_dict(),
                    "embed_dim": model.embed_dim,
                    "node_in_dim": model.node_in_dim,
                    "edge_in_dim": model.edge_in_dim,
                    "dare_weights": str(args.dare_weights.resolve()),
                    "initialization": initialization,
                    "train_indices": train_indices,
                    "validation_indices": val_indices,
                    "structure_index": str(args.structure_index.resolve()),
                    "excluded_directories": [str(path.resolve()) for path in args.exclude_dir],
                    "split_summary": split_summary,
                    "best_validation": validation,
                    "selection_metric": "same_formula_top1/top5_then_full_validation_top1/top5/loss",
                    "selection_key": list(validation_key),
                    "batch_sampler": args.batch_sampler,
                    "freeze_spectrum_tower": bool(args.freeze_spectrum_tower),
                    "seed": args.seed,
                    "chemical_domain": domain_metadata() if apply_domain_filter else None,
                    "domain_filter_enabled": apply_domain_filter,
                    "domain_rejections": split_summary.get("domain_rejections", {}),
                },
                args.output_weights,
            )

    report = {
        "schema_version": 1,
        "method": "spectrum_graph_contrastive_no_pubchem_rank",
        "status": "completed",
        "device": str(device),
        "train_count": len(train_indices),
        "validation_count": len(val_indices),
        "selection_metric": "same_formula_top1/top5_then_full_validation_top1/top5/loss",
        "best_validation": best_validation,
        "batch_sampler": args.batch_sampler,
        "freeze_spectrum_tower": bool(args.freeze_spectrum_tower),
        "weights": str(args.output_weights.resolve()),
        "initialization": initialization,
        "structure_index": str(args.structure_index.resolve()),
        "excluded_directories": [str(path.resolve()) for path in args.exclude_dir],
        "split_summary": split_summary,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "chemical_domain": domain_metadata() if apply_domain_filter else None,
        "domain_filter_enabled": apply_domain_filter,
        "domain_rejections": split_summary.get("domain_rejections", {}),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"权重: {args.output_weights.resolve()}\n报告: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()


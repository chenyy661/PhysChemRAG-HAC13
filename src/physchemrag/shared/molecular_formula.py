"""分子式规范化、解析和结构一致性校验工具。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_counts(formula: str | Mapping[str, int] | None) -> dict[str, int]:
    """将分子式转换为按元素排序的计数字典。"""
    if formula is None:
        return {}
    if isinstance(formula, Mapping):
        counts = Counter()
        for element, count in formula.items():
            element = str(element).strip()
            count = int(count)
            if not element or count < 0:
                raise ValueError(f"非法元素计数: {element}={count}")
            if count:
                counts[element] += count
        return dict(sorted(counts.items()))

    compact = re.sub(r"\s+", "", str(formula).strip())
    # RDKit 对带电分子会在分子式末尾附加 +、- 或带数字的电荷标记；
    # 分支约束比较的是元素计数，因此这里去除电荷而不改变元素组成。
    compact = re.sub(r"[+-]\d*$", "", compact)
    if not compact:
        return {}
    counts = Counter()
    position = 0
    for match in _FORMULA_TOKEN.finditer(compact):
        if match.start() != position:
            raise ValueError(f"无法解析分子式: {formula}")
        element, count_text = match.groups()
        count = int(count_text or "1")
        if count <= 0:
            raise ValueError(f"元素计数必须为正数: {formula}")
        counts[element] += count
        position = match.end()
    if position != len(compact):
        raise ValueError(f"无法解析分子式: {formula}")
    return dict(sorted(counts.items()))


def normalize_formula(formula: str | Mapping[str, int] | None) -> str:
    """返回无空格、元素按 Hill 规则排序的分子式。"""
    counts = formula_counts(formula)
    if not counts:
        return ""
    elements = list(counts)
    if "C" in counts:
        ordered = ["C"]
        if "H" in counts:
            ordered.append("H")
        ordered.extend(element for element in elements if element not in {"C", "H"})
    else:
        ordered = elements
    return "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in ordered
    )


def formula_from_smiles(smiles: str | None) -> str | None:
    """由 SMILES 计算规范化分子式；无法解析时返回 None。"""
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return normalize_formula(rdMolDescriptors.CalcMolFormula(mol))


def canonical_smiles(smiles: str | None, isomeric: bool = False) -> str | None:
    """返回规范化 SMILES，供候选去重和命中判断使用。"""
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    if not isomeric:
        # 连接结构评估忽略同位素和可移除的显式氢，避免同一分子重复占用候选名额。
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def formulas_match(
    left: str | Mapping[str, int] | None,
    right: str | Mapping[str, int] | None,
) -> bool:
    """执行元素计数级精确匹配。"""
    try:
        return formula_counts(left) == formula_counts(right) and bool(formula_counts(left))
    except (TypeError, ValueError):
        return False


__all__ = [
    "canonical_smiles",
    "formula_counts",
    "formula_from_smiles",
    "formulas_match",
    "normalize_formula",
]

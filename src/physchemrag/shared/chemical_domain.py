"""闭域结构检索任务的统一化学空间定义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem

from physchemrag.shared.molecular_formula import (
    canonical_smiles,
    formula_counts,
    formula_from_smiles,
    normalize_formula,
)


DOMAIN_NAME = "domain_hac13"
DOMAIN_VERSION = 1
FORMULA_ELEMENTS = ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I")
ALLOWED_ELEMENTS = frozenset(FORMULA_ELEMENTS)
MIN_HEAVY_ATOMS = 1
MAX_HEAVY_ATOMS = 13

# 兼容更显式的命名；所有别名均指向同一份闭域定义。
CHEMICAL_DOMAIN_ELEMENT_ORDER = FORMULA_ELEMENTS
ALLOWED_CHEMICAL_ELEMENTS = ALLOWED_ELEMENTS
MIN_HEAVY_ATOM_COUNT = MIN_HEAVY_ATOMS
MAX_HEAVY_ATOM_COUNT = MAX_HEAVY_ATOMS


@dataclass(frozen=True)
class DomainInspection:
    """单条分子式或结构的闭域检查结果。"""

    valid: bool
    reason: str
    reasons: tuple[str, ...]
    normalized_formula: str
    canonical_smiles: str
    heavy_atom_count: int | None
    elements: tuple[str, ...]
    unsupported_elements: tuple[str, ...]
    component_count: int | None = None
    formal_charge: int | None = None
    radical_electrons: int | None = None

    @property
    def eligible(self) -> bool:
        """返回是否属于闭域，兼容 eligibility 命名。"""
        return self.valid

    def as_dict(self) -> dict[str, Any]:
        """返回适合写入 JSON 审计清单的字典。"""
        payload = asdict(self)
        for key in ("reasons", "elements", "unsupported_elements"):
            payload[key] = list(payload[key])
        payload["eligible"] = self.valid
        payload["domain_name"] = DOMAIN_NAME
        return payload

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 就绪字典，作为 :meth:`as_dict` 的兼容名称。"""
        return self.as_dict()


# 兼容调用方使用的早期结果类型名称。
ChemicalDomainResult = DomainInspection
ChemicalDomainCheck = DomainInspection

def domain_metadata() -> dict[str, Any]:
    """返回可写入数据、缓存、权重和评测报告的域元数据。"""
    return {
        "name": DOMAIN_NAME,
        "version": DOMAIN_VERSION,
        "allowed_elements": list(FORMULA_ELEMENTS),
        "formula_elements": list(FORMULA_ELEMENTS),
        "min_heavy_atoms": MIN_HEAVY_ATOMS,
        "max_heavy_atoms": MAX_HEAVY_ATOMS,
    }


def domain_metadata_matches(value: Any) -> bool:
    """判断对象是否严格声明当前闭域元数据。

    新权重和缓存必须保存完整的 ``domain_metadata()`` 字典；旧版仅保存
    ``domain_hac13`` 字符串的对象只能作为历史文件识别，不能进入正式主线。
    """
    return isinstance(value, dict) and value == domain_metadata()


def _inspection(
    *,
    reasons: list[str],
    normalized_formula: str = "",
    canonical: str = "",
    heavy_atom_count: int | None = None,
    elements: set[str] | tuple[str, ...] = (),
    unsupported: set[str] | tuple[str, ...] = (),
    component_count: int | None = None,
    formal_charge: int | None = None,
    radical_electrons: int | None = None,
) -> DomainInspection:
    unique_reasons = tuple(dict.fromkeys(reasons))
    return DomainInspection(
        valid=not unique_reasons,
        reason=unique_reasons[0] if unique_reasons else "ok",
        reasons=unique_reasons,
        normalized_formula=normalized_formula,
        canonical_smiles=canonical,
        heavy_atom_count=heavy_atom_count,
        elements=tuple(sorted(elements)),
        unsupported_elements=tuple(sorted(unsupported)),
        component_count=component_count,
        formal_charge=formal_charge,
        radical_electrons=radical_electrons,
    )


def inspect_formula_domain(formula: str | dict[str, int] | None) -> DomainInspection:
    """检查分子式是否仅含允许元素且重原子数位于 1--13。"""
    try:
        counts = formula_counts(formula)
        normalized = normalize_formula(counts)
    except (TypeError, ValueError):
        return _inspection(reasons=["invalid_formula"])
    if not counts:
        return _inspection(reasons=["invalid_formula"])
    elements = set(counts)
    unsupported = elements - ALLOWED_ELEMENTS
    heavy_atom_count = sum(count for element, count in counts.items() if element != "H")
    reasons: list[str] = []
    if unsupported:
        reasons.append("unsupported_element")
    if not MIN_HEAVY_ATOMS <= heavy_atom_count <= MAX_HEAVY_ATOMS:
        reasons.append("heavy_atom_out_of_range")
    return _inspection(
        reasons=reasons,
        normalized_formula=normalized,
        heavy_atom_count=heavy_atom_count,
        elements=elements,
        unsupported=unsupported,
    )


def inspect_smiles_domain(
    smiles: str | None,
    formula: str | None = None,
    *,
    require_single_component: bool = False,
    require_neutral_closed_shell: bool = False,
) -> DomainInspection:
    """检查结构的元素、HAC、可选分子式一致性和任务资格。"""
    raw = str(smiles or "")
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return _inspection(reasons=["invalid_smiles"])

    calculated_formula = normalize_formula(formula_from_smiles(raw) or "")
    canonical = canonical_smiles(raw, isomeric=False) or ""
    elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    unsupported = elements - ALLOWED_ELEMENTS
    heavy_atom_count = int(molecule.GetNumHeavyAtoms())
    component_count = len(Chem.GetMolFrags(molecule, asMols=False, sanitizeFrags=False))
    formal_charge = int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))
    radical_electrons = int(sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()))
    reasons: list[str] = []
    if unsupported:
        reasons.append("unsupported_element")
    if not MIN_HEAVY_ATOMS <= heavy_atom_count <= MAX_HEAVY_ATOMS:
        reasons.append("heavy_atom_out_of_range")
    if formula:
        try:
            expected_formula = normalize_formula(formula)
        except (TypeError, ValueError):
            expected_formula = ""
        if not expected_formula or expected_formula != calculated_formula:
            reasons.append("formula_structure_mismatch")
    if require_single_component and component_count != 1:
        reasons.append("multi_component")
    if require_neutral_closed_shell and (formal_charge != 0 or radical_electrons != 0):
        reasons.append("charged_or_radical")
    return _inspection(
        reasons=reasons,
        normalized_formula=calculated_formula,
        canonical=canonical,
        heavy_atom_count=heavy_atom_count,
        elements=elements,
        unsupported=unsupported,
        component_count=component_count,
        formal_charge=formal_charge,
        radical_electrons=radical_electrons,
    )


def require_formula_domain(formula: str | None) -> DomainInspection:
    """验证分子式并在越域时给出可解释错误。"""
    result = inspect_formula_domain(formula)
    if not result.valid:
        raise ValueError(
            f"分子式不属于 {DOMAIN_NAME}: {formula!r}; "
            f"reasons={list(result.reasons)}, unsupported={list(result.unsupported_elements)}, "
            f"heavy_atoms={result.heavy_atom_count}"
        )
    return result


def formula_element_vector(formula: str | dict[str, int] | None) -> tuple[int, ...]:
    """按固定元素顺序返回严格 10 维分子式计数向量。"""
    result = require_formula_domain(formula)
    counts = formula_counts(result.normalized_formula)
    return tuple(int(counts.get(element, 0)) for element in FORMULA_ELEMENTS)


def check_formula_domain(formula: str | dict[str, int] | None) -> dict[str, Any]:
    """返回分子式检查的 JSON 就绪字典。"""
    return inspect_formula_domain(formula).to_dict()


def check_smiles_domain(
    smiles: str | None,
    formula: str | None = None,
    *,
    require_single_component: bool = False,
    require_neutral_closed_shell: bool = False,
) -> dict[str, Any]:
    """返回结构检查的 JSON 就绪字典。"""
    return inspect_smiles_domain(
        smiles,
        formula,
        require_single_component=require_single_component,
        require_neutral_closed_shell=require_neutral_closed_shell,
    ).to_dict()


def check_structure_domain(
    smiles: str | None,
    formula: str | None = None,
    *,
    require_single_component: bool = False,
    require_neutral_closed_shell: bool = False,
) -> dict[str, Any]:
    """返回结构检查字典，作为 :func:`check_smiles_domain` 的兼容名称。"""
    return check_smiles_domain(
        smiles,
        formula,
        require_single_component=require_single_component,
        require_neutral_closed_shell=require_neutral_closed_shell,
    )


__all__ = [
    "ALLOWED_CHEMICAL_ELEMENTS",
    "ALLOWED_ELEMENTS",
    "CHEMICAL_DOMAIN_ELEMENT_ORDER",
    "ChemicalDomainCheck",
    "ChemicalDomainResult",
    "DOMAIN_NAME",
    "DOMAIN_VERSION",
    "DomainInspection",
    "FORMULA_ELEMENTS",
    "MAX_HEAVY_ATOM_COUNT",
    "MAX_HEAVY_ATOMS",
    "MIN_HEAVY_ATOM_COUNT",
    "MIN_HEAVY_ATOMS",
    "check_formula_domain",
    "check_smiles_domain",
    "check_structure_domain",
    "domain_metadata",
    "domain_metadata_matches",
    "formula_element_vector",
    "inspect_formula_domain",
    "inspect_smiles_domain",
    "require_formula_domain",
]

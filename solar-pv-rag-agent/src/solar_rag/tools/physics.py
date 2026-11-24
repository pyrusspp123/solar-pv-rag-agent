"""物性计算工具：光伏材料常用的无外部依赖物理量计算。"""
from __future__ import annotations

from langchain_core.tools import tool

# Shockley-Queisser 单结极限效率近似表（AM1.5G，单位：带隙 eV -> 效率 %）
_SQ_TABLE: list[tuple[float, float]] = [
    (0.7, 22.0),
    (0.9, 28.0),
    (1.1, 32.8),
    (1.2, 33.4),
    (1.3, 33.6),
    (1.34, 33.7),
    (1.4, 33.5),
    (1.5, 32.9),
    (1.6, 32.0),
    (1.7, 30.9),
    (1.8, 29.6),
    (2.0, 26.7),
    (2.2, 23.4),
    (2.5, 18.8),
    (3.0, 12.9),
]


def _interp(x: float, table: list[tuple[float, float]]) -> float | None:
    if x < table[0][0] or x > table[-1][0]:
        return None
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


@tool
def bandgap_to_wavelength(bandgap_ev: float) -> str:
    """将半导体带隙（eV）转换为对应的光吸收截止波长（nm）：λ = 1240 / Eg。

    参数 bandgap_ev 为带隙值，单位电子伏特（eV）。
    """
    if bandgap_ev <= 0:
        return "带隙必须为正数（eV）。"
    wavelength_nm = 1240.0 / bandgap_ev
    return (
        f"带隙 {bandgap_ev:.3f} eV 对应的吸收截止波长约为 {wavelength_nm:.1f} nm，"
        f"即能量高于该带隙的光子（波长更短）才能被材料吸收。"
    )


@tool
def sq_efficiency(bandgap_ev: float) -> str:
    """估算单结太阳能电池的 Shockley-Queisser 理论极限效率（%）。

    参数 bandgap_ev 为吸收层带隙（eV），范围约 0.7~3.0 eV。
    """
    eff = _interp(bandgap_ev, _SQ_TABLE)
    if eff is None:
        return f"带隙 {bandgap_ev} eV 超出估算范围（0.7~3.0 eV），请调整后再试。"
    return (
        f"带隙 {bandgap_ev:.2f} eV 的单结太阳能电池，"
        f"Shockley-Queisser 理论极限效率约为 {eff:.1f}%。"
    )

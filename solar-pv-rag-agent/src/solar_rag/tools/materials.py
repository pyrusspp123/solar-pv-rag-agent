"""材料数据库工具：查询本地示例材料库，可选对接 Materials Project。"""
from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_core.tools import tool

from ..config import get_config

_CFG = get_config()


def _search_local(query: str) -> list[dict]:
    """在本地 CSV 材料库中模糊检索。"""
    csv_path = Path(_CFG.materials_csv)
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    q = query.strip().lower()
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            blob = json.dumps(row, ensure_ascii=False).lower()
            if q in blob:
                rows.append(row)
    return rows


def _query_materials_project(query: str, limit: int = 5) -> list[dict]:
    """查询 Materials Project 在线数据库（需 API key）。"""
    api_key = _CFG.materials_project_api_key
    if not api_key:
        return []
    url = (
        "https://api.materialsproject.org/materials/summary/"
        f"?formula={urllib.parse.quote(query)}&_limit={limit}"
    )
    req = urllib.request.Request(url, headers={"X-API-KEY": api_key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [{"_error": f"Materials Project 查询失败：{exc}"}]
    return data.get("data", [])


@tool
def material_database(query: str) -> str:
    """查询光伏材料数据库，返回材料名称、带隙、效率、稳定性等物性信息。

    输入材料名或关键词（如 'MAPbI3'、'钙钛矿'、'CIGS'）。优先使用本地示例
    材料库，若配置了 Materials Project API key 也会一并查询在线数据库。
    """
    local = _search_local(query)
    online = _query_materials_project(query)

    out_lines = []
    if local:
        out_lines.append(f"本地材料库命中 {len(local)} 条：")
        for r in local[:10]:
            out_lines.append("  - " + " | ".join(f"{k}={v}" for k, v in r.items()))
    if online and not (online and "_error" in online[0]):
        out_lines.append(f"Materials Project 命中 {len(online)} 条：")
        for r in online[:5]:
            out_lines.append("  - " + json.dumps(r, ensure_ascii=False)[:500])
    elif online and "_error" in online[0]:
        out_lines.append(online[0]["_error"])

    if not out_lines:
        return f"未找到与 '{query}' 相关的材料记录。可尝试输入化学式或材料类型关键词。"
    return "\n".join(out_lines)

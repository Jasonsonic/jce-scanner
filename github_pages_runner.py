from __future__ import annotations

import html
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
SITE_DIR = ROOT / "site"


def esc(value) -> str:
    return "" if pd.isna(value) else html.escape(str(value))


def score_class(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 80:
        return "good"
    if score >= 70:
        return "watch"
    return "low"


def generate_dashboard(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path).sort_values(["jce_entry_score", "four_line_width_pct"], ascending=[False, True])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    counts = df["recommendation"].value_counts().to_dict()
    rows = []
    for _, r in df.iterrows():
        rows.append(f'''<tr class="{score_class(float(r["jce_entry_score"]))}">
<td>{esc(r["symbol"])}</td><td class="score">{float(r["jce_entry_score"]):.1f}</td><td>{esc(r["recommendation"])}</td>
<td>{esc(r["entry_state"])}</td><td>{"是" if bool(r["short_mas_above_ma60"]) else "否"}</td>
<td>{float(r["compression_score_40"]):.1f}</td><td>{float(r["entry_score_30"]):.1f}</td>
<td>{float(r["half_year_position_score_10"]):.1f}</td><td>{float(r["stability_score_10"]):.1f}</td>
<td>{float(r["volume_score_10"]):.1f}</td><td>{float(r["four_line_width_pct"]):.2f}%</td>
<td>{float(r["close_to_ma60_pct"]):.2f}%</td><td>{float(r["half_year_position_pct"]):.1f}%</td>
<td>{float(r["two_day_volume_ratio"]):.2f}×</td><td>{esc(r["volume_state"])}</td></tr>''')
    cards = "".join(f'<div class="card"><strong>{html.escape(str(k))}</strong><span>{v}</span></div>' for k, v in counts.items())
    doc = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JCE Scanner V3</title>
<style>body{{margin:0;background:#f4f6f8;color:#17202a;font-family:Inter,"Microsoft YaHei",sans-serif}}header{{padding:28px 5vw 18px;background:white;border-bottom:1px solid #e5e9ee}}h1{{margin:0 0 8px;font-size:28px}}.meta{{color:#65717e;font-size:14px}}.cards{{display:flex;gap:12px;flex-wrap:wrap;padding:18px 5vw}}.card{{min-width:150px;background:white;padding:14px 16px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.05)}}.card strong{{display:block;font-size:13px;color:#65717e}}.card span{{display:block;font-size:26px;margin-top:4px}}.actions{{padding:0 5vw 16px}}.actions a{{display:inline-block;margin-right:10px;padding:9px 13px;background:#17202a;color:white;border-radius:8px;text-decoration:none}}.table-wrap{{margin:0 5vw 40px;overflow:auto;background:white;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.05)}}table{{width:100%;border-collapse:collapse;min-width:1500px}}th,td{{padding:11px 12px;border-bottom:1px solid #edf0f3;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4),th:last-child,td:last-child{{text-align:left}}th{{position:sticky;top:0;background:#f8fafb;font-size:12px;color:#53606d}}.score{{font-weight:700}}tr.excellent .score{{color:#087f5b}}tr.good .score{{color:#1971c2}}tr.watch .score{{color:#e67700}}footer{{padding:20px 5vw 36px;color:#73808c;font-size:12px}}</style></head><body>
<header><h1>JCE Scanner V3</h1><div class="meta">仅分析最新交易日，用于下一交易日开盘前筛选 · 更新：{generated_at} · 共 {len(df)} 只</div></header>
<section class="cards">{cards}</section><div class="actions"><a href="jce_scan_v3.xlsx">下载 Excel</a><a href="jce_scan_v3.csv">下载 CSV</a></div>
<div class="table-wrap"><table><thead><tr><th>代码</th><th>建仓分</th><th>建议</th><th>位置状态</th><th>三条短均线均高于MA60</th><th>四线压缩/40</th><th>贴近MA60/30</th><th>半年位置/10</th><th>稳定性/10</th><th>量能/10</th><th>四线宽度</th><th>收盘距MA60</th><th>半年位置</th><th>两日量比</th><th>量价状态</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<footer>MA5、MA8、MA13之间不要求排列顺序；均线方向不参与核心评分。技术研究工具，不构成投资建议。</footer></body></html>'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, str(ROOT / "jce_scan.py"), "--batch-size", "3", "--pause", "20"], cwd=ROOT)
    if result.returncode != 0:
        return result.returncode
    csv_path = OUTPUT_DIR / "jce_scan_v3.csv"
    xlsx_path = OUTPUT_DIR / "jce_scan_v3.xlsx"
    generate_dashboard(csv_path, SITE_DIR / "index.html")
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")
    (SITE_DIR / "jce_scan_v3.csv").write_bytes(csv_path.read_bytes())
    (SITE_DIR / "jce_scan_v3.xlsx").write_bytes(xlsx_path.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

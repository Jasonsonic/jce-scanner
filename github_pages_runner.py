from __future__ import annotations

import html
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SITE = ROOT / "site"


def esc(value) -> str:
    return "" if pd.isna(value) else html.escape(str(value))


def generate_site(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["qualified", "jce_score"], ascending=[False, False])
    qualified = int(df["qualified"].sum())
    top_count = int(df["grade"].isin(["A+", "A"]).sum())
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows = []
    for _, r in df.iterrows():
        qualified_text = "通过" if bool(r["qualified"]) else "未通过"
        rows.append(
            f"""
<tr class="{'qualified' if bool(r['qualified']) else 'rejected'}">
<td>{esc(r['symbol'])}</td>
<td class="score">{float(r['jce_score']):.1f}</td>
<td>{esc(r['grade'])}</td>
<td>{qualified_text}</td>
<td>{esc(r['entry_route'])}</td>
<td>{esc(r['recommendation'])}</td>
<td>{float(r['compression_score_25']):.1f}</td>
<td>{float(r['entry_score_20']):.1f}</td>
<td>{float(r['trend_score_15']):.1f}</td>
<td>{float(r['reversal_score_15']):.1f}</td>
<td>{float(r['three_month_low_score_10']):.1f}</td>
<td>{float(r['volume_score_10']):.1f}</td>
<td>{float(r['vcp_score_5']):.1f}</td>
<td>{float(r['four_line_width_pct']):.2f}%</td>
<td>{float(r['close_to_ma60_pct']):.2f}%</td>
<td>{float(r['three_month_low_distance_pct']):.2f}%</td>
<td>{float(r['relative_strength_20d_vs_spy_pct']):.2f}%</td>
<td>{esc(r['trigger_reasons'])}</td>
<td>{esc(r['rejection_reasons'])}</td>
</tr>
"""
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JCE Professional Scanner V4</title>
<style>
body {{ margin:0; background:#f3f5f7; color:#17202a; font-family:Inter,"Microsoft YaHei",sans-serif; }}
header {{ background:#fff; padding:28px 5vw 18px; border-bottom:1px solid #e6e9ed; }}
h1 {{ margin:0 0 8px; font-size:28px; }}
.meta {{ color:#687582; font-size:14px; }}
.cards {{ display:flex; flex-wrap:wrap; gap:12px; padding:18px 5vw; }}
.card {{ min-width:150px; background:#fff; border-radius:12px; padding:14px 16px; box-shadow:0 2px 10px rgba(0,0,0,.05); }}
.card strong {{ display:block; color:#687582; font-size:13px; }}
.card span {{ display:block; margin-top:5px; font-size:26px; }}
.actions {{ padding:0 5vw 16px; }}
.actions a {{ display:inline-block; padding:9px 13px; margin-right:10px; background:#17202a; color:#fff; border-radius:8px; text-decoration:none; }}
.table-wrap {{ margin:0 5vw 40px; overflow:auto; background:#fff; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.05); }}
table {{ border-collapse:collapse; width:100%; min-width:2100px; }}
th,td {{ padding:10px 11px; border-bottom:1px solid #edf0f3; white-space:nowrap; text-align:right; }}
th {{ background:#f8fafb; color:#596674; font-size:12px; position:sticky; top:0; }}
td:first-child,th:first-child,td:nth-child(5),th:nth-child(5),td:nth-child(6),th:nth-child(6),td:nth-last-child(-n+2),th:nth-last-child(-n+2) {{ text-align:left; }}
.score {{ font-weight:700; }}
tr.qualified .score {{ color:#087f5b; }}
tr.rejected {{ color:#8a949e; }}
footer {{ padding:0 5vw 36px; color:#76828e; font-size:12px; }}
</style>
</head>
<body>
<header>
<h1>JCE Professional Scanner V4</h1>
<div class="meta">仅使用最新完整交易日数据，为下一交易日提供候选 · 更新时间：{updated}</div>
</header>
<section class="cards">
<div class="card"><strong>扫描股票</strong><span>{len(df)}</span></div>
<div class="card"><strong>通过资格筛选</strong><span>{qualified}</span></div>
<div class="card"><strong>A+ / A</strong><span>{top_count}</span></div>
</section>
<div class="actions">
<a href="jce_v4.xlsx">下载 Excel</a>
<a href="jce_v4.csv">下载 CSV</a>
</div>
<div class="table-wrap">
<table>
<thead><tr>
<th>代码</th><th>总分</th><th>等级</th><th>资格</th><th>路线</th><th>建议</th>
<th>压缩/25</th><th>建仓/20</th><th>趋势/15</th><th>反转/15</th>
<th>三月低位/10</th><th>量价/10</th><th>VCP/5</th>
<th>四线宽度</th><th>距MA60</th><th>距三月低点</th><th>20日相对SPY</th>
<th>推荐原因</th><th>未通过原因</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
<footer>系统采用“资格筛选 + 质量评分”两层结构。任何技术评分均不保证上涨，也不构成投资建议。</footer>
</body>
</html>"""
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "index.html").write_text(document, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "jce_scan.py"), "--batch-size", "3", "--pause", "20"],
        cwd=ROOT,
    )
    if completed.returncode:
        return completed.returncode

    csv_path = OUTPUT / "jce_v4.csv"
    xlsx_path = OUTPUT / "jce_v4.xlsx"
    generate_site(csv_path)
    (SITE / "jce_v4.csv").write_bytes(csv_path.read_bytes())
    (SITE / "jce_v4.xlsx").write_bytes(xlsx_path.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

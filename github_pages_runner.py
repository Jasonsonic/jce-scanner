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
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "watch"
    return "low"


def generate_dashboard(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path).sort_values(
        ["jce_score", "compression_pct"],
        ascending=[False, True],
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stage_counts = df["stage"].value_counts().to_dict()

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"""
            <tr class="{score_class(float(r['jce_score']))}">
              <td>{esc(r['symbol'])}</td>
              <td class="score">{float(r['jce_score']):.1f}</td>
              <td>{esc(r['stage'])}</td>
              <td>{float(r['trend_score_25']):.1f}</td>
              <td>{float(r['compression_score_25']):.1f}</td>
              <td>{float(r['position_score_20']):.1f}</td>
              <td>{float(r['momentum_score_20']):.1f}</td>
              <td>{float(r['market_score_10']):.1f}</td>
              <td>{float(r['compression_pct']):.2f}%</td>
              <td>{float(r['half_year_position_pct']):.1f}%</td>
              <td>{float(r['prev_close_to_ma60_pct']):.2f}%</td>
              <td>{float(r['two_day_volume_ratio']):.2f}×</td>
              <td>{esc(r['volume_state'])}</td>
            </tr>
            """
        )

    cards = "".join(
        f'<div class="card"><strong>{html.escape(str(k))}</strong><span>{v}</span></div>'
        for k, v in stage_counts.items()
    )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JCE Scanner</title>
<style>
:root {{ font-family: Inter, "Microsoft YaHei", sans-serif; }}
body {{ margin:0; background:#f4f6f8; color:#17202a; }}
header {{ padding:28px 5vw 18px; background:white; border-bottom:1px solid #e5e9ee; }}
h1 {{ margin:0 0 8px; font-size:28px; }}
.meta {{ color:#65717e; font-size:14px; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; padding:18px 5vw; }}
.card {{ min-width:130px; background:white; padding:14px 16px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.05); }}
.card strong {{ display:block; font-size:13px; color:#65717e; }}
.card span {{ display:block; font-size:26px; margin-top:4px; }}
.actions {{ padding:0 5vw 16px; }}
.actions a {{ display:inline-block; margin-right:10px; padding:9px 13px; background:#17202a; color:white; border-radius:8px; text-decoration:none; }}
.table-wrap {{ margin:0 5vw 40px; overflow:auto; background:white; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.05); }}
table {{ width:100%; border-collapse:collapse; min-width:1280px; }}
th,td {{ padding:11px 12px; border-bottom:1px solid #edf0f3; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(3),td:nth-child(3),th:last-child,td:last-child {{ text-align:left; }}
th {{ position:sticky; top:0; background:#f8fafb; font-size:12px; color:#53606d; }}
.score {{ font-weight:700; }}
tr.excellent .score {{ color:#087f5b; }}
tr.good .score {{ color:#1971c2; }}
tr.watch .score {{ color:#e67700; }}
footer {{ padding:20px 5vw 36px; color:#73808c; font-size:12px; }}
</style>
</head>
<body>
<header>
<h1>JCE Scanner</h1>
<div class="meta">最近更新：{generated_at} · 共分析 {len(df)} 只股票</div>
</header>
<section class="cards">{cards}</section>
<div class="actions">
<a href="jce_scan_v2.xlsx">下载 Excel</a>
<a href="jce_scan_v2.csv">下载 CSV</a>
</div>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>代码</th><th>总分</th><th>阶段</th><th>趋势/25</th><th>压缩/25</th>
<th>位置/20</th><th>启动/20</th><th>市场/10</th><th>均线压缩</th>
<th>半年位置</th><th>昨日距MA60</th><th>两日量比</th><th>量价状态</th>
</tr>
</thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
<footer>技术研究工具，不构成投资建议。免费行情源可能出现延迟、缺失或限流。</footer>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "jce_scan.py"),
            "--batch-size",
            "3",
            "--pause",
            "20",
        ],
        cwd=ROOT,
    )
    if result.returncode != 0:
        return result.returncode

    csv_path = OUTPUT_DIR / "jce_scan_v2.csv"
    xlsx_path = OUTPUT_DIR / "jce_scan_v2.xlsx"

    generate_dashboard(csv_path, SITE_DIR / "index.html")
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")
    (SITE_DIR / "jce_scan_v2.csv").write_bytes(csv_path.read_bytes())
    (SITE_DIR / "jce_scan_v2.xlsx").write_bytes(xlsx_path.read_bytes())

    print(f"网页已生成：{SITE_DIR / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

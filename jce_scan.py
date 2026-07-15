
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class JCEConfig:
    period: str = "1y"
    interval: str = "1d"
    batch_size: int = 8
    pause_seconds: float = 8.0
    max_retries: int = 4
    retry_base_seconds: float = 30.0
    platform_days: int = 4
    min_rows: int = 130


def load_watchlist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"original_symbol", "yahoo_symbol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"自选股文件缺少列: {sorted(missing)}")
    df = df.dropna(subset=["yahoo_symbol"]).copy()
    df["original_symbol"] = df["original_symbol"].astype(str).str.strip()
    df["yahoo_symbol"] = df["yahoo_symbol"].astype(str).str.strip()
    return df[df["yahoo_symbol"] != ""]


def batched(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def extract_one(downloaded: pd.DataFrame, ticker: str, batch: list[str]) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        l0 = downloaded.columns.get_level_values(0)
        l1 = downloaded.columns.get_level_values(1)
        if ticker in l0:
            frame = downloaded[ticker].copy()
        elif ticker in l1:
            frame = downloaded.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if len(batch) != 1:
            return pd.DataFrame()
        frame = downloaded.copy()

    frame.columns = [str(c).title() for c in frame.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in frame.columns for c in needed):
        return pd.DataFrame()
    return frame[needed].dropna(subset=["Close"])



def cache_file(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}.csv"


def load_cached_prices(cache_dir: Path, ticker: str, min_rows: int) -> pd.DataFrame:
    path = cache_file(cache_dir, ticker)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame.columns = [str(c).title() for c in frame.columns]
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if all(c in frame.columns for c in needed) and len(frame) >= min_rows:
            return frame[needed].dropna(subset=["Close"])
    except Exception:
        pass
    return pd.DataFrame()


def save_cached_prices(cache_dir: Path, ticker: str, frame: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_file(cache_dir, ticker), encoding="utf-8-sig")

def download_prices(
    symbols: list[str],
    cfg: JCEConfig,
    cache_dir: Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}

    # 先读取本地缓存。正常情况下第二次运行几乎不需要重新请求全部股票。
    pending: list[str] = []
    for ticker in symbols:
        cached = pd.DataFrame() if refresh else load_cached_prices(cache_dir, ticker, cfg.min_rows)
        if not cached.empty:
            result[ticker] = cached
        else:
            pending.append(ticker)

    print(f"缓存命中 {len(result)} 只；需要联网下载 {len(pending)} 只。")
    if not pending:
        return result

    total_batches = (len(pending) + cfg.batch_size - 1) // cfg.batch_size

    for n, batch in enumerate(batched(pending, cfg.batch_size), start=1):
        print(f"下载第 {n}/{total_batches} 批，共 {len(batch)} 只。")
        success = False
        last_error = None

        for attempt in range(1, cfg.max_retries + 1):
            try:
                data = yf.download(
                    tickers=batch,
                    period=cfg.period,
                    interval=cfg.interval,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,     # 关闭并发，显著降低触发限流的概率
                    progress=False,
                    timeout=45,
                )

                downloaded_count = 0
                for ticker in batch:
                    frame = extract_one(data, ticker, batch)
                    if not frame.empty:
                        result[ticker] = frame
                        save_cached_prices(cache_dir, ticker, frame)
                        downloaded_count += 1

                if downloaded_count:
                    success = True
                    missing = [t for t in batch if t not in result]
                    if missing:
                        print(f"  本批仍缺少：{', '.join(missing)}")
                    break

            except Exception as exc:
                last_error = exc

            if attempt < cfg.max_retries:
                wait = cfg.retry_base_seconds * (2 ** (attempt - 1))
                print(f"  第 {attempt} 次失败，等待 {wait:.0f} 秒后重试……")
                time.sleep(wait)

        if not success:
            print(
                f"  本批下载失败。Yahoo 可能正在限制当前IP。错误：{last_error}",
                file=sys.stderr,
            )

        # 每批之间强制停顿，避免145只连续高速请求。
        if n < total_batches:
            time.sleep(cfg.pause_seconds)

    return result

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def slope_pct(series: pd.Series, lookback: int = 5) -> float:
    if len(series) <= lookback or pd.isna(series.iloc[-1]) or pd.isna(series.iloc[-1-lookback]):
        return np.nan
    old = series.iloc[-1-lookback]
    return float((series.iloc[-1] - old) / old * 100) if old else np.nan


def score_half_year_position(pos: float) -> tuple[float, str]:
    # 用户指定：30%-40% 五星；70%-80% 三星；80%-90% 两星。
    if 0.30 <= pos < 0.70:
        return 10.0, "★★★★★"
    if 0.20 <= pos < 0.30 or 0.70 <= pos < 0.80:
        return 6.0, "★★★☆☆"
    if 0.10 <= pos < 0.20 or 0.80 <= pos < 0.90:
        return 3.0, "★★☆☆☆"
    return 0.0, "★☆☆☆☆"


def score_prev_close_ma60(distance_pct: float, above: bool) -> tuple[float, str]:
    if not above:
        return 0.0, "跌破MA60"
    d = abs(distance_pct)
    if d <= 1:
        return 8.0, "★★★★★"
    if d <= 2:
        return 6.5, "★★★★☆"
    if d <= 4:
        return 4.5, "★★★☆☆"
    if d <= 6:
        return 2.0, "★★☆☆☆"
    return 0.0, "★☆☆☆☆"


def score_one(df: pd.DataFrame, market: dict[str, pd.DataFrame], cfg: JCEConfig) -> dict:
    if len(df) < cfg.min_rows:
        raise ValueError(f"数据不足，仅 {len(df)} 行")

    d = df.copy()
    for p in (5, 8, 13, 20, 60, 120):
        d[f"MA{p}"] = d["Close"].rolling(p).mean()
    d["ATR14"] = compute_atr(d, 14)
    d["VOL20"] = d["Volume"].rolling(20).mean()

    latest = d.iloc[-1]
    prev = d.iloc[-2]
    recent4 = d.iloc[-4:]
    prior4 = d.iloc[-8:-4]

    # ---------- 1. Trend 25 ----------
    ma_order = latest["MA5"] > latest["MA8"] > latest["MA13"] > latest["MA60"]
    slopes = {p: slope_pct(d[f"MA{p}"], 5) for p in (5, 8, 13, 60)}
    trend_score = 8.0 if ma_order else 0.0
    trend_score += 3.0 if slopes[5] > 0 else 0.0
    trend_score += 3.0 if slopes[8] > 0 else 0.0
    trend_score += 4.0 if slopes[13] > 0 else 0.0
    trend_score += 7.0 if slopes[60] > 0 else 0.0

    # ---------- 2. Compression 25 ----------
    mas = [latest["MA5"], latest["MA8"], latest["MA13"], latest["MA60"]]
    compression = (max(mas) - min(mas)) / latest["MA60"] if latest["MA60"] else np.nan
    if compression <= 0.01:
        ma_comp_score = 12.0
    elif compression <= 0.015:
        ma_comp_score = 10.0
    elif compression <= 0.02:
        ma_comp_score = 8.0
    elif compression <= 0.03:
        ma_comp_score = 4.0
    else:
        ma_comp_score = 0.0

    platform_high = recent4["High"].max()
    platform_low = recent4["Low"].min()
    platform_range = (platform_high - platform_low) / platform_low if platform_low else np.nan
    platform_score = 4.0 if platform_range <= 0.06 else (2.0 if platform_range <= 0.09 else 0.0)

    vol_ratio_4 = recent4["Volume"].mean() / prior4["Volume"].mean() if prior4["Volume"].mean() else np.nan
    platform_volume_score = 4.0 if vol_ratio_4 < 0.90 else (2.0 if vol_ratio_4 < 1.0 else 0.0)

    atr_recent = recent4["ATR14"].mean()
    atr_prior = prior4["ATR14"].mean()
    atr_ratio = atr_recent / atr_prior if atr_prior else np.nan
    atr_score = 5.0 if atr_ratio < 0.95 else (2.5 if atr_ratio < 1.0 else 0.0)
    compression_score = ma_comp_score + platform_score + platform_volume_score + atr_score

    # ---------- 3. Position 20 ----------
    half = d.iloc[-126:]
    half_high = half["High"].max()
    half_low = half["Low"].min()
    half_pos = (latest["Close"] - half_low) / (half_high - half_low) if half_high > half_low else np.nan
    half_score, half_stars = score_half_year_position(float(half_pos))

    prev_close_to_ma60_pct = (prev["Close"] - prev["MA60"]) / prev["MA60"] * 100
    support_score, support_stars = score_prev_close_ma60(
        float(prev_close_to_ma60_pct), bool(prev["Close"] >= prev["MA60"])
    )

    distance_to_breakout = (platform_high - latest["Close"]) / platform_high if platform_high else np.nan
    if -0.03 <= distance_to_breakout <= 0.01:
        breakout_position_score = 2.0
    elif 0.01 < distance_to_breakout <= 0.03:
        breakout_position_score = 1.0
    else:
        breakout_position_score = 0.0
    position_score = half_score + support_score + breakout_position_score

    # ---------- 4. Momentum 20 ----------
    recent2 = d.iloc[-2:]
    baseline20 = d.iloc[-22:-2]
    volume_expansion_ratio = recent2["Volume"].mean() / baseline20["Volume"].mean() if baseline20["Volume"].mean() else np.nan
    two_day_return = latest["Close"] / d["Close"].iloc[-3] - 1

    if volume_expansion_ratio >= 2.5:
        volume_expansion_score = 12.0
    elif volume_expansion_ratio >= 2.0:
        volume_expansion_score = 10.5
    elif volume_expansion_ratio >= 1.5:
        volume_expansion_score = 8.0
    elif volume_expansion_ratio >= 1.2:
        volume_expansion_score = 5.0
    elif volume_expansion_ratio >= 0.9:
        volume_expansion_score = 2.0
    else:
        volume_expansion_score = 0.0

    if two_day_return > 0:
        direction_score = 4.0
        volume_state = "放量上涨" if volume_expansion_ratio >= 1.2 else "价格走强/尚未明显放量"
    elif two_day_return <= -0.05 and volume_expansion_ratio >= 1.5:
        direction_score = 0.0
        volume_expansion_score = max(0.0, volume_expansion_score - 5.0)
        volume_state = "高风险放量下跌"
    elif two_day_return <= -0.03 and volume_expansion_ratio >= 1.5:
        direction_score = 0.0
        volume_expansion_score = max(0.0, volume_expansion_score - 3.0)
        volume_state = "放量下跌/疑似抛压"
    else:
        direction_score = 1.0
        volume_state = "量价中性"

    # Smart-money trace: up days volume vs down days volume over 20 sessions.
    last20 = d.iloc[-20:].copy()
    daily_ret = last20["Close"].pct_change()
    up_avg = last20.loc[daily_ret > 0, "Volume"].mean()
    down_avg = last20.loc[daily_ret < 0, "Volume"].mean()
    smart_ratio = up_avg / down_avg if pd.notna(up_avg) and pd.notna(down_avg) and down_avg else np.nan
    smart_money_score = 4.0 if smart_ratio >= 1.20 else (2.0 if smart_ratio >= 1.0 else 0.0)
    momentum_score = min(20.0, volume_expansion_score + direction_score + smart_money_score)

    # ---------- 5. Market 10 ----------
    market_score = 0.0
    market_flags = []
    for ticker, points in (("SPY", 5.0), ("QQQ", 5.0)):
        m = market.get(ticker)
        if m is not None and len(m) >= 60:
            ma60 = m["Close"].rolling(60).mean().iloc[-1]
            if m["Close"].iloc[-1] >= ma60:
                market_score += points
                market_flags.append(f"{ticker}>MA60")
            else:
                market_flags.append(f"{ticker}<MA60")
        else:
            market_flags.append(f"{ticker}无数据")

    total = round(trend_score + compression_score + position_score + momentum_score + market_score, 1)

    hard_match = (
        ma_order
        and compression <= 0.02
        and all(slopes[p] > 0 for p in (5, 8, 13, 60))
        and platform_range <= 0.06
        and prev["Close"] >= prev["MA60"]
    )

    if "高风险放量下跌" in volume_state:
        stage = "风险型"
    elif hard_match and volume_expansion_ratio >= 1.5 and two_day_return > 0:
        stage = "启动型"
    elif hard_match:
        stage = "蓄势型"
    elif compression <= 0.03 and platform_range <= 0.09:
        stage = "形成中"
    else:
        stage = "暂不符合"

    return {
        "date": d.index[-1].date().isoformat(),
        "jce_score": total,
        "stage": stage,
        "trend_score_25": trend_score,
        "compression_score_25": compression_score,
        "position_score_20": position_score,
        "momentum_score_20": momentum_score,
        "market_score_10": market_score,
        "ma_order": ma_order,
        "ma5_slope_5d_pct": slopes[5],
        "ma8_slope_5d_pct": slopes[8],
        "ma13_slope_5d_pct": slopes[13],
        "ma60_slope_5d_pct": slopes[60],
        "compression_pct": compression * 100,
        "platform_range_pct": platform_range * 100,
        "platform_volume_ratio": vol_ratio_4,
        "atr_ratio": atr_ratio,
        "half_year_position_pct": half_pos * 100,
        "half_year_position_stars": half_stars,
        "prev_close_to_ma60_pct": prev_close_to_ma60_pct,
        "support_stars": support_stars,
        "distance_to_breakout_pct": distance_to_breakout * 100,
        "two_day_volume_ratio": volume_expansion_ratio,
        "two_day_return_pct": two_day_return * 100,
        "volume_state": volume_state,
        "smart_money_ratio": smart_ratio,
        "market_flags": "；".join(market_flags),
        "close": float(latest["Close"]),
        "ma5": float(latest["MA5"]),
        "ma8": float(latest["MA8"]),
        "ma13": float(latest["MA13"]),
        "ma60": float(latest["MA60"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="JCE Scanner V2")
    parser.add_argument("--watchlist", default="config/watchlist.csv")
    parser.add_argument("--output", default="output/jce_scan_v2.xlsx")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--refresh", action="store_true", help="忽略本地缓存并重新下载全部行情")
    parser.add_argument("--batch-size", type=int, default=8, help="每批下载股票数，限流时可改为3或5")
    parser.add_argument("--pause", type=float, default=8.0, help="批次间等待秒数")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = JCEConfig(
        period=args.period,
        batch_size=max(1, args.batch_size),
        pause_seconds=max(0.0, args.pause),
    )
    watchlist = load_watchlist(root / args.watchlist)

    stock_symbols = watchlist["yahoo_symbol"].tolist()
    all_symbols = list(dict.fromkeys(stock_symbols + ["SPY", "QQQ"]))
    mapping = dict(zip(watchlist["yahoo_symbol"], watchlist["original_symbol"]))

    print(f"开始扫描 {len(stock_symbols)} 只自选股……")
    prices = download_prices(
        all_symbols,
        cfg,
        cache_dir=root / "data_cache",
        refresh=args.refresh,
    )
    market = {k: prices.get(k) for k in ("SPY", "QQQ")}

    rows, errors = [], []
    for ticker in stock_symbols:
        frame = prices.get(ticker)
        if frame is None or frame.empty:
            errors.append({"symbol": mapping.get(ticker, ticker), "yahoo_symbol": ticker, "error": "无行情数据"})
            continue
        try:
            row = score_one(frame, market, cfg)
            row["symbol"] = mapping.get(ticker, ticker)
            row["yahoo_symbol"] = ticker
            rows.append(row)
        except Exception as exc:
            errors.append({"symbol": mapping.get(ticker, ticker), "yahoo_symbol": ticker, "error": str(exc)})

    if not rows:
        print("没有得到可分析结果。", file=sys.stderr)
        return 2

    result = pd.DataFrame(rows).sort_values(
        ["jce_score", "compression_pct"], ascending=[False, True]
    )
    candidates = result[result["stage"].isin(["启动型", "蓄势型", "形成中"])]
    risk = result[result["stage"] == "风险型"]

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="全部结果", index=False)
        candidates.to_excel(writer, sheet_name="JCE候选", index=False)
        risk.to_excel(writer, sheet_name="风险型", index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="失败", index=False)

    csv_path = output_path.with_suffix(".csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    show = [
        "symbol", "jce_score", "stage",
        "trend_score_25", "compression_score_25",
        "position_score_20", "momentum_score_20", "market_score_10",
        "half_year_position_pct", "prev_close_to_ma60_pct",
        "two_day_volume_ratio", "volume_state",
    ]
    print("\nJCE V2 Top：")
    print(result[show].head(args.top).to_string(index=False))
    print(f"\nExcel：{output_path}")
    print(f"CSV：{csv_path}")
    print(f"成功：{len(result)}；失败：{len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

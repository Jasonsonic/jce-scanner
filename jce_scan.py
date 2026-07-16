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
class Config:
    period: str = "1y"
    batch_size: int = 3
    pause_seconds: float = 20.0
    max_retries: int = 4
    retry_base_seconds: float = 30.0
    min_rows: int = 130

    # Compression route hard filters
    short_ma_tolerance_pct: float = 0.20
    max_four_line_width_pct: float = 2.00
    max_close_ma60_distance_pct: float = 5.00
    max_five_day_gain_pct: float = 20.00
    max_half_year_position_pct: float = 90.00

    # Reversal route
    decline_lookback_days: int = 15
    max_small_up_days: int = 4
    small_up_limit_pct: float = 3.00
    min_decline_pct: float = 8.00


def load_watchlist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"original_symbol", "yahoo_symbol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"自选股文件缺少列: {sorted(missing)}")
    df = df.dropna(subset=["yahoo_symbol"]).copy()
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    return df[df["yahoo_symbol"] != ""]


def batches(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).title() for c in frame.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in frame.columns for c in needed):
        return pd.DataFrame()
    return frame[needed].dropna(subset=["Close"])


def extract_ticker(downloaded: pd.DataFrame, ticker: str, batch: list[str]) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)
        if ticker in level0:
            return normalize_frame(downloaded[ticker].copy())
        if ticker in level1:
            return normalize_frame(downloaded.xs(ticker, axis=1, level=1).copy())
        return pd.DataFrame()
    return normalize_frame(downloaded.copy()) if len(batch) == 1 else pd.DataFrame()


def cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}.csv"


def load_cache(cache_dir: Path, ticker: str, min_rows: int) -> pd.DataFrame:
    path = cache_path(cache_dir, ticker)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame = normalize_frame(frame)
        if len(frame) >= min_rows:
            return frame
    except Exception:
        pass
    return pd.DataFrame()


def save_cache(cache_dir: Path, ticker: str, frame: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path(cache_dir, ticker), encoding="utf-8-sig")


def download_prices(
    symbols: list[str],
    cfg: Config,
    cache_dir: Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    pending: list[str] = []

    for ticker in symbols:
        cached = pd.DataFrame() if refresh else load_cache(cache_dir, ticker, cfg.min_rows)
        if cached.empty:
            pending.append(ticker)
        else:
            result[ticker] = cached

    print(f"缓存命中 {len(result)} 只；需联网下载 {len(pending)} 只。", flush=True)
    total = math.ceil(len(pending) / cfg.batch_size) if pending else 0

    for batch_no, batch in enumerate(batches(pending, cfg.batch_size), start=1):
        print(f"[下载 {batch_no}/{total}] {', '.join(batch)}", flush=True)
        downloaded_count = 0
        last_error: Exception | None = None

        for attempt in range(cfg.max_retries):
            try:
                raw = yf.download(
                    tickers=batch,
                    period=cfg.period,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,
                    progress=False,
                    timeout=45,
                )
                for ticker in batch:
                    frame = extract_ticker(raw, ticker, batch)
                    if not frame.empty:
                        result[ticker] = frame
                        save_cache(cache_dir, ticker, frame)
                        downloaded_count += 1
                if downloaded_count:
                    break
            except Exception as exc:
                last_error = exc

            if attempt < cfg.max_retries - 1:
                wait = cfg.retry_base_seconds * (2 ** attempt)
                print(f"  请求失败，{wait:.0f}秒后重试。", flush=True)
                time.sleep(wait)

        missing = [ticker for ticker in batch if ticker not in result]
        if missing:
            print(f"  无数据：{', '.join(missing)}；错误：{last_error}", file=sys.stderr, flush=True)

        if batch_no < total:
            print(f"  等待 {cfg.pause_seconds:.0f} 秒以降低限流风险。", flush=True)
            time.sleep(cfg.pause_seconds)

    return result


def pct_change(new: float, old: float) -> float:
    return (new / old - 1.0) * 100 if old else np.nan


def linear_slope_pct(series: pd.Series, lookback: int) -> float:
    values = series.dropna().iloc[-lookback:]
    if len(values) < lookback or values.mean() == 0:
        return np.nan
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values.to_numpy(dtype=float), 1)[0]
    return float(slope / values.mean() * 100)


def score_by_threshold(value: float, bands: list[tuple[float, float]], default: float = 0.0) -> float:
    for maximum, score in bands:
        if value <= maximum:
            return score
    return default


def reversal_signal(d: pd.DataFrame, cfg: Config) -> dict:
    latest = d.iloc[-1]
    previous = d.iloc[-2]
    bullish = bool(latest["Close"] > latest["Open"] and latest["Close"] > previous["Close"])
    bull_return = pct_change(float(latest["Close"]), float(previous["Close"]))

    cursor = len(d) - 2
    span_days = down_days = small_up_days = flat_days = 0
    earliest = cursor
    flat_limit_pct = 0.35

    while cursor >= 1 and span_days < cfg.decline_lookback_days:
        current_close = float(d["Close"].iloc[cursor])
        prior_close = float(d["Close"].iloc[cursor - 1])
        day_return = pct_change(current_close, prior_close)

        if day_return < -flat_limit_pct:
            down_days += 1
        elif 0 < day_return <= cfg.small_up_limit_pct and small_up_days < cfg.max_small_up_days:
            small_up_days += 1
        elif abs(day_return) <= flat_limit_pct and flat_days < 3:
            flat_days += 1
        else:
            break

        earliest = cursor - 1
        cursor -= 1
        span_days += 1

    start_close = float(d["Close"].iloc[earliest]) if span_days else float(previous["Close"])
    decline_pct = pct_change(float(previous["Close"]), start_close)

    baseline_volume = d["Volume"].iloc[-21:-1].mean()
    bull_volume_ratio = (
        float(latest["Volume"] / baseline_volume)
        if baseline_volume and pd.notna(baseline_volume)
        else np.nan
    )

    signal = bool(
        bullish
        and span_days >= 3
        and down_days >= 3
        and decline_pct <= -cfg.min_decline_pct
        and bull_return <= 10
    )

    # 15-point module.
    score = 0.0
    if signal:
        score += score_by_threshold(
            span_days,
            [(4, 4.0), (7, 6.0), (10, 7.0), (15, 8.0)],
            8.0,
        )
        abs_decline = abs(decline_pct)
        score += score_by_threshold(
            abs_decline,
            [(10, 3.0), (15, 5.0), (25, 6.0), (999, 4.0)],
            0.0,
        )
        if pd.notna(bull_volume_ratio) and bull_volume_ratio >= 1.2:
            score += 1.0
    return {
        "reversal_score_15": min(15.0, score),
        "reversal_signal": signal,
        "decline_span_days": span_days,
        "decline_down_days": down_days,
        "ignored_small_up_days": small_up_days,
        "decline_pct": decline_pct,
        "bull_day_return_pct": bull_return,
        "bull_volume_ratio": bull_volume_ratio,
    }


def compression_module(ma5: float, ma8: float, ma13: float, ma60: float) -> tuple[float, float]:
    width = pct_change(max(ma5, ma8, ma13, ma60), min(ma5, ma8, ma13, ma60))
    score = score_by_threshold(
        width,
        [(0.50, 25.0), (1.00, 22.0), (1.50, 18.0), (2.00, 13.0), (3.00, 6.0)],
    )
    return score, width


def entry_module(close: float, ma60: float) -> tuple[float, float]:
    distance = pct_change(close, ma60)
    if distance >= 0:
        score = score_by_threshold(
            distance,
            [(0.50, 20.0), (1.00, 18.0), (2.00, 15.0), (3.00, 10.0), (5.00, 4.0)],
        )
    else:
        below = abs(distance)
        score = score_by_threshold(below, [(0.25, 8.0), (0.50, 4.0)])
    return score, distance


def trend_module(
    d: pd.DataFrame,
    spy: pd.DataFrame | None,
) -> tuple[float, dict]:
    ma60_slope_20 = linear_slope_pct(d["MA60"], 20)
    stock_ret20 = pct_change(float(d["Close"].iloc[-1]), float(d["Close"].iloc[-21]))
    stock_ret60 = pct_change(float(d["Close"].iloc[-1]), float(d["Close"].iloc[-61]))

    spy_ret20 = spy_ret60 = np.nan
    if spy is not None and len(spy) >= 61:
        spy_ret20 = pct_change(float(spy["Close"].iloc[-1]), float(spy["Close"].iloc[-21]))
        spy_ret60 = pct_change(float(spy["Close"].iloc[-1]), float(spy["Close"].iloc[-61]))

    rs20 = stock_ret20 - spy_ret20 if pd.notna(spy_ret20) else np.nan
    rs60 = stock_ret60 - spy_ret60 if pd.notna(spy_ret60) else np.nan

    # MA60 may be flat or mildly falling; steep decline is penalized.
    if pd.isna(ma60_slope_20):
        slope_score = 0.0
    elif ma60_slope_20 >= 0.03:
        slope_score = 7.0
    elif ma60_slope_20 >= -0.02:
        slope_score = 5.0
    elif ma60_slope_20 >= -0.08:
        slope_score = 2.0
    else:
        slope_score = 0.0

    rs_score = 0.0
    if pd.notna(rs20):
        rs_score += 4.0 if rs20 >= 5 else 3.0 if rs20 >= 0 else 0.0
    if pd.notna(rs60):
        rs_score += 4.0 if rs60 >= 10 else 3.0 if rs60 >= 0 else 0.0

    return min(15.0, slope_score + rs_score), {
        "ma60_slope_20d_pct_per_day": ma60_slope_20,
        "stock_return_20d_pct": stock_ret20,
        "stock_return_60d_pct": stock_ret60,
        "relative_strength_20d_vs_spy_pct": rs20,
        "relative_strength_60d_vs_spy_pct": rs60,
    }


def three_month_low_module(d: pd.DataFrame) -> tuple[float, float, float]:
    recent = d.iloc[-63:]
    low = float(recent["Low"].min())
    distance = pct_change(float(d["Close"].iloc[-1]), low)
    score = score_by_threshold(
        distance,
        [(3.0, 10.0), (7.0, 8.0), (10.0, 6.0), (15.0, 3.0)],
    )
    return score, low, distance


def volume_module(d: pd.DataFrame) -> tuple[float, dict]:
    # Dry-up: the 5 sessions before the latest day compared with the prior 20.
    dry_window = d["Volume"].iloc[-6:-1]
    baseline = d["Volume"].iloc[-26:-6]
    dry_ratio = float(dry_window.mean() / baseline.mean()) if baseline.mean() else np.nan

    latest_baseline = d["Volume"].iloc[-21:-1].mean()
    latest_ratio = float(d["Volume"].iloc[-1] / latest_baseline) if latest_baseline else np.nan
    latest_return = pct_change(float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2]))

    dry_score = 0.0
    if pd.notna(dry_ratio):
        dry_score = 6.0 if dry_ratio <= 0.65 else 5.0 if dry_ratio <= 0.80 else 3.0 if dry_ratio <= 0.95 else 0.0

    expansion_score = 0.0
    if pd.notna(latest_ratio) and 0 <= latest_return <= 8:
        expansion_score = 4.0 if latest_ratio >= 1.5 else 3.0 if latest_ratio >= 1.2 else 1.0
    if latest_return > 10 or (latest_return <= -3 and pd.notna(latest_ratio) and latest_ratio >= 1.2):
        expansion_score = 0.0

    return min(10.0, dry_score + expansion_score), {
        "volume_dry_up_ratio": dry_ratio,
        "latest_volume_ratio": latest_ratio,
        "latest_day_return_pct": latest_return,
    }


def vcp_module(d: pd.DataFrame) -> tuple[float, dict]:
    # Compare average true ranges across three consecutive five-day windows.
    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    natr = tr / d["Close"] * 100

    old = float(natr.iloc[-15:-10].mean())
    middle = float(natr.iloc[-10:-5].mean())
    recent = float(natr.iloc[-5:].mean())

    contraction = old > middle > recent
    ratio = recent / old if old else np.nan

    if contraction and ratio <= 0.60:
        score = 5.0
    elif contraction and ratio <= 0.80:
        score = 4.0
    elif recent < old:
        score = 2.0
    else:
        score = 0.0
    return score, {
        "vcp_old_natr_pct": old,
        "vcp_middle_natr_pct": middle,
        "vcp_recent_natr_pct": recent,
        "vcp_contraction": contraction,
    }


def platform_diagnostics(d: pd.DataFrame) -> dict:
    recent8 = d.iloc[-8:]
    high = float(recent8["High"].max())
    low = float(recent8["Low"].min())
    range_pct = pct_change(high, low)
    return {
        "platform_8d_range_pct": range_pct,
        "platform_quality": "紧凑" if range_pct <= 6 else "一般" if range_pct <= 10 else "宽幅",
    }


def analyze_one(d: pd.DataFrame, spy: pd.DataFrame | None, cfg: Config) -> dict:
    if len(d) < cfg.min_rows:
        raise ValueError(f"数据不足，仅{len(d)}行")

    d = d.copy()
    for period in (5, 8, 13, 60):
        d[f"MA{period}"] = d["Close"].rolling(period).mean()

    latest = d.iloc[-1]
    ma5, ma8, ma13, ma60 = (float(latest[f"MA{x}"]) for x in (5, 8, 13, 60))
    close = float(latest["Close"])

    compression_score, four_line_width = compression_module(ma5, ma8, ma13, ma60)
    entry_score, close_ma60_distance = entry_module(close, ma60)
    trend_score, trend_info = trend_module(d, spy)
    reversal = reversal_signal(d, cfg)
    low_score, three_month_low, low_distance = three_month_low_module(d)
    volume_score, volume_info = volume_module(d)
    vcp_score, vcp_info = vcp_module(d)
    platform = platform_diagnostics(d)

    five_day_return = pct_change(close, float(d["Close"].iloc[-6]))
    half = d.iloc[-126:]
    half_low = float(half["Low"].min())
    half_high = float(half["High"].max())
    half_position = (
        (close - half_low) / (half_high - half_low) * 100
        if half_high > half_low else np.nan
    )

    tolerance = cfg.short_ma_tolerance_pct / 100
    short_mas_near_or_above = all(ma >= ma60 * (1 - tolerance) for ma in (ma5, ma8, ma13))

    compression_route = bool(
        short_mas_near_or_above
        and four_line_width <= cfg.max_four_line_width_pct
        and abs(close_ma60_distance) <= cfg.max_close_ma60_distance_pct
        and five_day_return <= cfg.max_five_day_gain_pct
        and half_position < cfg.max_half_year_position_pct
    )

    reversal_route = bool(
        reversal["reversal_signal"]
        and low_distance <= 15
        and five_day_return <= cfg.max_five_day_gain_pct
    )

    qualified = compression_route or reversal_route
    route = (
        "双重信号"
        if compression_route and reversal_route
        else "均线压缩"
        if compression_route
        else "连跌首阳"
        if reversal_route
        else "未通过"
    )

    total = round(
        compression_score
        + entry_score
        + trend_score
        + reversal["reversal_score_15"]
        + low_score
        + volume_score
        + vcp_score,
        1,
    )

    if not qualified:
        grade = "未入选"
    elif total >= 85:
        grade = "A+"
    elif total >= 75:
        grade = "A"
    elif total >= 65:
        grade = "B"
    else:
        grade = "C"

    reasons: list[str] = []
    rejects: list[str] = []

    if four_line_width <= 1:
        reasons.append("四线高度压缩")
    if abs(close_ma60_distance) <= 1:
        reasons.append("价格紧贴MA60")
    if short_mas_near_or_above:
        reasons.append("短期均线位于MA60附近上方")
    if low_distance <= 15:
        reasons.append("处于三个月低位")
    if reversal["reversal_signal"]:
        reasons.append("触发连跌首阳")
    if volume_info["volume_dry_up_ratio"] <= 0.8:
        reasons.append("前期量能枯竭")
    if volume_info["latest_volume_ratio"] >= 1.2:
        reasons.append("最新交易日放量")
    if vcp_info["vcp_contraction"]:
        reasons.append("波动逐步收缩")
    if platform["platform_8d_range_pct"] <= 6:
        reasons.append("8日平台紧凑")

    if not short_mas_near_or_above:
        rejects.append("短期均线未处于MA60附近上方")
    if four_line_width > cfg.max_four_line_width_pct:
        rejects.append("四线宽度超过2%")
    if abs(close_ma60_distance) > cfg.max_close_ma60_distance_pct:
        rejects.append("收盘价距离MA60超过5%")
    if five_day_return > cfg.max_five_day_gain_pct:
        rejects.append("近5日涨幅超过20%")
    if half_position >= cfg.max_half_year_position_pct:
        rejects.append("处于半年价格区间90%以上")
    if not qualified and not rejects:
        rejects.append("未形成压缩或连跌首阳资格")

    return {
        "date": d.index[-1].date().isoformat(),
        "qualified": qualified,
        "entry_route": route,
        "grade": grade,
        "jce_score": total,
        "recommendation": "明日重点观察" if grade in {"A+", "A"} else "候选观察" if grade in {"B", "C"} else "暂不入选",
        "trigger_reasons": "；".join(reasons) or "无突出加分信号",
        "rejection_reasons": "；".join(rejects),
        "compression_score_25": compression_score,
        "entry_score_20": entry_score,
        "trend_score_15": trend_score,
        "reversal_score_15": reversal["reversal_score_15"],
        "three_month_low_score_10": low_score,
        "volume_score_10": volume_score,
        "vcp_score_5": vcp_score,
        "short_mas_near_or_above_ma60": short_mas_near_or_above,
        "four_line_width_pct": four_line_width,
        "close_to_ma60_pct": close_ma60_distance,
        "five_day_return_pct": five_day_return,
        "half_year_position_pct": half_position,
        "three_month_low": three_month_low,
        "three_month_low_distance_pct": low_distance,
        "platform_8d_range_pct": platform["platform_8d_range_pct"],
        "platform_quality": platform["platform_quality"],
        "close": close,
        "ma5": ma5,
        "ma8": ma8,
        "ma13": ma13,
        "ma60": ma60,
        **trend_info,
        **reversal,
        **volume_info,
        **vcp_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="JCE Professional Scanner V4")
    parser.add_argument("--watchlist", default="config/watchlist.csv")
    parser.add_argument("--output", default="output/jce_v4.xlsx")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--pause", type=float, default=20.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = Config(
        period=args.period,
        batch_size=max(1, args.batch_size),
        pause_seconds=max(0.0, args.pause),
    )

    watchlist = load_watchlist(root / args.watchlist)
    symbols = watchlist["yahoo_symbol"].tolist()
    mapping = dict(zip(watchlist["yahoo_symbol"], watchlist["original_symbol"]))
    all_symbols = list(dict.fromkeys(symbols + ["SPY"]))

    print(f"开始扫描最新完整交易日：{len(symbols)}只自选股。", flush=True)
    prices = download_prices(
        all_symbols,
        cfg,
        cache_dir=root / "data_cache",
        refresh=args.refresh,
    )
    spy = prices.get("SPY")

    rows: list[dict] = []
    failures: list[dict] = []
    for index, ticker in enumerate(symbols, start=1):
        print(f"[分析 {index}/{len(symbols)}] {ticker}", flush=True)
        frame = prices.get(ticker)
        if frame is None or frame.empty:
            failures.append({"symbol": mapping.get(ticker, ticker), "yahoo_symbol": ticker, "error": "无行情数据"})
            continue
        try:
            row = analyze_one(frame, spy, cfg)
            row["symbol"] = mapping.get(ticker, ticker)
            row["yahoo_symbol"] = ticker
            rows.append(row)
        except Exception as exc:
            failures.append({"symbol": mapping.get(ticker, ticker), "yahoo_symbol": ticker, "error": str(exc)})

    if not rows:
        print("没有得到可分析结果。", file=sys.stderr)
        return 2

    result = pd.DataFrame(rows).sort_values(
        ["qualified", "jce_score", "four_line_width_pct"],
        ascending=[False, False, True],
    )
    qualified = result[result["qualified"]]
    top = qualified[qualified["grade"].isin(["A+", "A"])]
    rejected = result[~result["qualified"]]

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="全部结果", index=False)
        qualified.to_excel(writer, sheet_name="通过资格筛选", index=False)
        top.to_excel(writer, sheet_name="明日重点", index=False)
        rejected.to_excel(writer, sheet_name="未通过及原因", index=False)
        if failures:
            pd.DataFrame(failures).to_excel(writer, sheet_name="下载或计算失败", index=False)

    csv_path = output.with_suffix(".csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    display = [
        "symbol", "qualified", "entry_route", "grade", "jce_score",
        "recommendation", "trigger_reasons", "rejection_reasons",
        "four_line_width_pct", "close_to_ma60_pct",
        "three_month_low_distance_pct", "relative_strength_20d_vs_spy_pct",
    ]
    print("\nJCE Professional V4 排行榜：")
    print(result[display].head(args.top).to_string(index=False))
    print(f"\nExcel：{output}")
    print(f"CSV：{csv_path}")
    print(f"通过资格筛选：{len(qualified)}/{len(result)}；A+/A：{len(top)}；失败：{len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

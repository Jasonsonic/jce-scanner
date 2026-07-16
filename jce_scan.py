from __future__ import annotations

import argparse
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
    batch_size: int = 3
    pause_seconds: float = 20.0
    max_retries: int = 4
    retry_base_seconds: float = 30.0
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


def download_prices(symbols: list[str], cfg: JCEConfig, cache_dir: Path, refresh: bool = False) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    pending: list[str] = []
    for ticker in symbols:
        cached = pd.DataFrame() if refresh else load_cached_prices(cache_dir, ticker, cfg.min_rows)
        if not cached.empty:
            result[ticker] = cached
        else:
            pending.append(ticker)

    print(f"缓存命中 {len(result)} 只；需要联网下载 {len(pending)} 只。", flush=True)
    if not pending:
        return result

    total_batches = (len(pending) + cfg.batch_size - 1) // cfg.batch_size
    for n, batch in enumerate(batched(pending, cfg.batch_size), start=1):
        print(f"[批次 {n}/{total_batches}] 下载：{', '.join(batch)}", flush=True)
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
                    threads=False,
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
                        print(f"  本批缺少：{', '.join(missing)}", flush=True)
                    break
            except Exception as exc:
                last_error = exc
            if attempt < cfg.max_retries:
                wait = cfg.retry_base_seconds * (2 ** (attempt - 1))
                print(f"  第 {attempt} 次失败，等待 {wait:.0f} 秒后重试……", flush=True)
                time.sleep(wait)
        if not success:
            print(f"  本批下载失败：{last_error}", file=sys.stderr, flush=True)
        if n < total_batches:
            print(f"  等待 {cfg.pause_seconds:.0f} 秒，避免 Yahoo 限流……", flush=True)
            time.sleep(cfg.pause_seconds)
    return result


def score_compression(width_pct: float) -> float:
    if width_pct <= 0.50:
        return 40.0
    if width_pct <= 1.00:
        return 36.0
    if width_pct <= 1.50:
        return 31.0
    if width_pct <= 2.00:
        return 24.0
    if width_pct <= 3.00:
        return 12.0
    return 0.0


def score_close_distance(distance_pct: float) -> float:
    if distance_pct >= 0:
        if distance_pct <= 0.50:
            return 30.0
        if distance_pct <= 1.00:
            return 28.0
        if distance_pct <= 2.00:
            return 23.0
        if distance_pct <= 3.00:
            return 16.0
        if distance_pct <= 5.00:
            return 7.0
        return 0.0
    below = abs(distance_pct)
    if below <= 0.25:
        return 14.0
    if below <= 0.50:
        return 8.0
    return 0.0


def score_half_year_position(position: float) -> tuple[float, str]:
    if 0.30 <= position < 0.70:
        return 10.0, "★★★★★"
    if 0.20 <= position < 0.30 or 0.70 <= position < 0.80:
        return 6.0, "★★★☆☆"
    if 0.10 <= position < 0.20 or 0.80 <= position < 0.90:
        return 3.0, "★★☆☆☆"
    return 0.0, "★☆☆☆☆"


def score_stability(d: pd.DataFrame) -> tuple[float, float, float]:
    recent = d.iloc[-4:]
    platform_high = recent["High"].max()
    platform_low = recent["Low"].min()
    range_pct = (platform_high - platform_low) / platform_low * 100 if platform_low else np.nan
    max_abs_daily_return = recent["Close"].pct_change().abs().max() * 100
    four_day_return = (recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1) * 100
    score = 0.0
    if range_pct <= 4:
        score += 5.0
    elif range_pct <= 6:
        score += 4.0
    elif range_pct <= 8:
        score += 2.0
    if pd.notna(max_abs_daily_return):
        if max_abs_daily_return <= 3:
            score += 3.0
        elif max_abs_daily_return <= 5:
            score += 1.5
    if -3 <= four_day_return <= 6:
        score += 2.0
    return min(10.0, score), float(range_pct), float(four_day_return)


def score_volume(d: pd.DataFrame) -> tuple[float, float, float, str]:
    recent2 = d.iloc[-2:]
    baseline20 = d.iloc[-22:-2]
    baseline_avg = baseline20["Volume"].mean()
    ratio = recent2["Volume"].mean() / baseline_avg if baseline_avg else np.nan
    two_day_return = (d["Close"].iloc[-1] / d["Close"].iloc[-3] - 1) * 100
    score = 0.0
    state = "量能普通"
    if pd.notna(ratio):
        if 1.5 <= ratio < 2.5:
            score, state = 8.0, "近两日明显放量"
        elif 1.2 <= ratio < 1.5:
            score, state = 6.0, "近两日温和放量"
        elif 0.9 <= ratio < 1.2:
            score, state = 3.0, "量能接近常态"
        elif ratio >= 2.5:
            score, state = 6.0, "近两日巨量，需防止过热"
    if score > 0 and 0 <= two_day_return <= 6:
        score += 2.0
        state += "；价格温和走强"
    elif two_day_return > 10:
        score, state = 0.0, "近两日涨幅过大，已可能错过起涨点"
    elif two_day_return <= -3 and pd.notna(ratio) and ratio >= 1.2:
        score, state = 0.0, "放量下跌，疑似抛压"
    return min(10.0, score), float(ratio) if pd.notna(ratio) else np.nan, float(two_day_return), state




def score_reversal_after_decline(d: pd.DataFrame) -> dict:
    """
    识别“较长下跌阶段后，最新交易日出现确认阳线”。

    与V3.1不同：
    - 下跌阶段可以持续较长时间；
    - 中间允许最多2个“小幅上涨日”；
    - 小幅上涨默认定义为单日收盘涨幅不超过3%；
    - 一旦出现明显上涨、横盘过久或容忍次数耗尽，停止向前追溯。
    """
    if len(d) < 40:
        return {
            "reversal_score": 0.0,
            "reversal_signal": False,
            "decline_span_days": 0,
            "decline_down_days": 0,
            "ignored_small_up_days": 0,
            "decline_pct": np.nan,
            "bull_day_return_pct": np.nan,
            "bull_body_pct": np.nan,
            "bull_volume_ratio": np.nan,
            "reversal_state": "数据不足",
        }

    latest = d.iloc[-1]
    prev = d.iloc[-2]

    latest_bullish = latest["Close"] > latest["Open"]
    close_above_prev = latest["Close"] > prev["Close"]
    bull_day_return_pct = (latest["Close"] / prev["Close"] - 1) * 100
    bull_body_pct = (latest["Close"] / latest["Open"] - 1) * 100

    max_lookback = 15
    max_small_up_days = 4
    small_up_limit_pct = 3.0
    flat_limit_pct = 0.35

    cursor = len(d) - 2
    earliest_index = cursor
    decline_down_days = 0
    ignored_small_up_days = 0
    flat_days = 0
    span_days = 0

    while cursor >= 1 and span_days < max_lookback:
        today_close = float(d["Close"].iloc[cursor])
        prior_close = float(d["Close"].iloc[cursor - 1])
        day_return_pct = (today_close / prior_close - 1) * 100

        if day_return_pct < -flat_limit_pct:
            decline_down_days += 1
        elif 0 < day_return_pct <= small_up_limit_pct and ignored_small_up_days < max_small_up_days:
            ignored_small_up_days += 1
        elif abs(day_return_pct) <= flat_limit_pct and flat_days < 3:
            # 少量近似横盘日也允许存在，避免把缓慢下跌切断。
            flat_days += 1
        else:
            break

        earliest_index = cursor - 1
        cursor -= 1
        span_days += 1

    if span_days > 0:
        start_close = float(d["Close"].iloc[earliest_index])
        end_close = float(d["Close"].iloc[-2])
        decline_pct = (end_close / start_close - 1) * 100
    else:
        decline_pct = 0.0

    baseline_volume = d["Volume"].iloc[-21:-1].mean()
    bull_volume_ratio = float(latest["Volume"] / baseline_volume) if baseline_volume else np.nan

    score = 0.0

    # 下跌阶段持续时间：长周期不封顶，但20日以后得分不再继续增加。
    if span_days >= 15:
        score += 38
    elif span_days >= 10:
        score += 34
    elif span_days >= 7:
        score += 29
    elif span_days >= 5:
        score += 24
    elif span_days >= 3:
        score += 17

    # 实际下跌日数量，防止大量横盘或小涨被误判为持续下跌。
    if decline_down_days >= 10:
        score += 12
    elif decline_down_days >= 7:
        score += 10
    elif decline_down_days >= 5:
        score += 7
    elif decline_down_days >= 3:
        score += 4

    abs_decline = abs(decline_pct)
    if 8 <= abs_decline <= 20:
        score += 25
    elif 20 < abs_decline <= 30:
        score += 18
    elif abs_decline > 30:
        score += 9
    else:
        score += 0

    if latest_bullish:
        score += 8
    if close_above_prev:
        score += 8
    if latest["Close"] >= (prev["High"] + prev["Low"]) / 2:
        score += 4

    if pd.notna(bull_volume_ratio):
        if bull_volume_ratio >= 1.5:
            score += 5
        elif bull_volume_ratio >= 1.2:
            score += 4
        elif bull_volume_ratio >= 0.9:
            score += 2

    if 0.3 <= bull_day_return_pct <= 6:
        score += 4
    elif 6 < bull_day_return_pct <= 10:
        score += 1
    elif bull_day_return_pct > 10:
        score = min(score, 79)

    # 核心条件：阶段内至少3个真正下跌日，整体净跌至少8%，最新为确认阳线。
    reversal_signal = bool(
        span_days >= 3
        and decline_down_days >= 3
        and decline_pct <= -8
        and latest_bullish
        and close_above_prev
        and bull_day_return_pct <= 10
    )

    if reversal_signal and score >= 85:
        state = "A-长跌阶段后强确认阳线"
    elif reversal_signal and score >= 70:
        state = "B-下跌阶段后出现确认阳线"
    elif latest_bullish and decline_down_days >= 3:
        state = "C-初步止跌，确认不足"
    elif decline_down_days >= 3:
        state = "观察-仍处下跌阶段"
    else:
        state = "无反转信号"

    return {
        "reversal_score": round(min(100.0, score), 1),
        "reversal_signal": reversal_signal,
        "decline_span_days": span_days,
        "decline_down_days": decline_down_days,
        "ignored_small_up_days": ignored_small_up_days,
        "decline_pct": float(decline_pct),
        "bull_day_return_pct": float(bull_day_return_pct),
        "bull_body_pct": float(bull_body_pct),
        "bull_volume_ratio": bull_volume_ratio,
        "reversal_state": state,
    }


def score_three_month_low(distance_pct: float) -> tuple[float, str]:
    """
    当前收盘价相对过去约3个月（63个交易日）最低价的位置加分。

    distance_pct = (当前收盘价 / 3个月最低价 - 1) * 100
    """
    if distance_pct <= 3:
        return 10.0, "★★★★★ 极接近三个月低点"
    if distance_pct <= 7:
        return 8.0, "★★★★☆ 接近三个月低点"
    if distance_pct <= 10:
        return 6.0, "★★★☆☆ 处于三个月低位"
    if distance_pct <= 15:
        return 3.0, "★★☆☆☆ 略高于三个月低点"
    return 0.0, "★☆☆☆☆ 距三个月低点较远"


def score_one(df: pd.DataFrame, cfg: JCEConfig) -> dict:
    if len(df) < cfg.min_rows:
        raise ValueError(f"数据不足，仅 {len(df)} 行")
    d = df.copy()
    for p in (5, 8, 13, 60):
        d[f"MA{p}"] = d["Close"].rolling(p).mean()
    latest = d.iloc[-1]
    previous = d.iloc[-2]

    ma5, ma8, ma13, ma60 = (float(latest[f"MA{p}"]) for p in (5, 8, 13, 60))
    close = float(latest["Close"])
    if any(pd.isna(v) for v in (ma5, ma8, ma13, ma60)):
        raise ValueError("均线数据尚未形成")

    short_mas_above_ma60 = ma5 >= ma60 and ma8 >= ma60 and ma13 >= ma60
    short_ma_spread_pct = (max(ma5, ma8, ma13) - min(ma5, ma8, ma13)) / ma60 * 100
    four_line_width_pct = (max(ma5, ma8, ma13, ma60) - min(ma5, ma8, ma13, ma60)) / ma60 * 100

    compression_score = score_compression(four_line_width_pct)
    close_to_ma60_pct = (close - ma60) / ma60 * 100
    entry_score = score_close_distance(close_to_ma60_pct)

    half = d.iloc[-126:]
    half_high, half_low = float(half["High"].max()), float(half["Low"].min())
    half_position = (close - half_low) / (half_high - half_low) if half_high > half_low else np.nan
    position_score, position_stars = score_half_year_position(float(half_position))

    three_month = d.iloc[-63:]
    three_month_low = float(three_month["Low"].min())
    three_month_low_distance_pct = (
        (close / three_month_low - 1) * 100 if three_month_low > 0 else np.nan
    )
    three_month_low_bonus, three_month_low_state = score_three_month_low(
        float(three_month_low_distance_pct)
    )

    stability_score, four_day_range_pct, four_day_return_pct = score_stability(d)
    volume_score, two_day_volume_ratio, two_day_return_pct, volume_state = score_volume(d)
    reversal = score_reversal_after_decline(d)

    raw_score = compression_score + entry_score + position_score + stability_score + volume_score
    base_total_score = raw_score if short_mas_above_ma60 else min(raw_score, 69.0)
    total_score = min(100.0, base_total_score + three_month_low_bonus)

    if short_mas_above_ma60 and total_score >= 90:
        recommendation = "A-明日重点考虑"
    elif short_mas_above_ma60 and total_score >= 80:
        recommendation = "B-高质量候选"
    elif short_mas_above_ma60 and total_score >= 70:
        recommendation = "C-接近形成"
    elif close_to_ma60_pct > 5 or four_line_width_pct > 3:
        recommendation = "D-已发散或离MA60过远"
    elif not short_mas_above_ma60:
        recommendation = "D-短期均线未全部站上MA60"
    else:
        recommendation = "D-继续观察"

    if close_to_ma60_pct > 5:
        entry_state = "已远离MA60"
    elif close_to_ma60_pct < -0.5:
        entry_state = "明显跌破MA60"
    elif abs(close_to_ma60_pct) <= 1:
        entry_state = "非常接近MA60"
    elif abs(close_to_ma60_pct) <= 3:
        entry_state = "接近MA60"
    else:
        entry_state = "距离MA60一般"

    reversal_base_score = float(reversal["reversal_score"])
    reversal_score = min(100.0, reversal_base_score + three_month_low_bonus)
    priority_score = max(float(total_score), reversal_score)
    if reversal_score > total_score and reversal["reversal_signal"]:
        primary_signal = "连续下跌后首阳"
        final_recommendation = reversal["reversal_state"]
    else:
        primary_signal = "均线压缩贴近MA60"
        final_recommendation = recommendation

    return {
        "date": d.index[-1].date().isoformat(),
        "priority_score": round(priority_score, 1),
        "primary_signal": primary_signal,
        "final_recommendation": final_recommendation,
        "jce_entry_score": round(total_score, 1),
        "recommendation": recommendation,
        "entry_state": entry_state,
        "short_mas_above_ma60": short_mas_above_ma60,
        "compression_score_40": compression_score,
        "entry_score_30": entry_score,
        "half_year_position_score_10": position_score,
        "stability_score_10": stability_score,
        "volume_score_10": volume_score,
        "four_line_width_pct": four_line_width_pct,
        "short_ma_spread_pct": short_ma_spread_pct,
        "close_to_ma60_pct": close_to_ma60_pct,
        "half_year_position_pct": half_position * 100,
        "half_year_position_stars": position_stars,
        "three_month_low": three_month_low,
        "three_month_low_distance_pct": three_month_low_distance_pct,
        "three_month_low_bonus_10": three_month_low_bonus,
        "three_month_low_state": three_month_low_state,
        "four_day_range_pct": four_day_range_pct,
        "four_day_return_pct": four_day_return_pct,
        "two_day_volume_ratio": two_day_volume_ratio,
        "two_day_return_pct": two_day_return_pct,
        "volume_state": volume_state,
        "reversal_base_score": reversal_base_score,
        "reversal_score": reversal_score,
        "reversal_signal": reversal["reversal_signal"],
        "reversal_state": reversal["reversal_state"],
        "decline_span_days": reversal["decline_span_days"],
        "decline_down_days": reversal["decline_down_days"],
        "ignored_small_up_days": reversal["ignored_small_up_days"],
        "decline_pct": reversal["decline_pct"],
        "bull_day_return_pct": reversal["bull_day_return_pct"],
        "bull_body_pct": reversal["bull_body_pct"],
        "bull_volume_ratio": reversal["bull_volume_ratio"],
        "close": close,
        "ma5": ma5,
        "ma8": ma8,
        "ma13": ma13,
        "ma60": ma60,
        "prev_close_to_ma60_pct": (float(previous["Close"]) - float(previous["MA60"])) / float(previous["MA60"]) * 100,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="JCE Scanner V3.4：加入三个月低点加分")
    parser.add_argument("--watchlist", default="config/watchlist.csv")
    parser.add_argument("--output", default="output/jce_scan_v3_4.xlsx")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--pause", type=float, default=20.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = JCEConfig(period=args.period, batch_size=max(1, args.batch_size), pause_seconds=max(0.0, args.pause))
    watchlist = load_watchlist(root / args.watchlist)
    symbols = watchlist["yahoo_symbol"].tolist()
    mapping = dict(zip(watchlist["yahoo_symbol"], watchlist["original_symbol"]))

    print(f"开始扫描最新交易日，共 {len(symbols)} 只股票。", flush=True)
    prices = download_prices(symbols, cfg, cache_dir=root / "data_cache", refresh=args.refresh)

    rows, errors = [], []
    for i, ticker in enumerate(symbols, start=1):
        print(f"[分析 {i}/{len(symbols)}] {ticker}", flush=True)
        frame = prices.get(ticker)
        if frame is None or frame.empty:
            errors.append({"symbol": mapping.get(ticker, ticker), "yahoo_symbol": ticker, "error": "无行情数据"})
            continue
        try:
            row = score_one(frame, cfg)
            row["symbol"] = mapping.get(ticker, ticker)
            row["yahoo_symbol"] = ticker
            rows.append(row)
        except Exception as exc:
            errors.append({"symbol": mapping.get(ticker, ticker), "yahoo_symbol": ticker, "error": str(exc)})

    if not rows:
        print("没有得到可分析结果。", file=sys.stderr)
        return 2

    result = pd.DataFrame(rows).sort_values(["priority_score", "reversal_score", "jce_entry_score", "four_line_width_pct"], ascending=[False, False, False, True])
    candidates = result[(result["recommendation"].str.startswith(("A-", "B-", "C-"))) | (result["reversal_signal"])]
    top_candidates = result[(result["recommendation"].str.startswith(("A-", "B-"))) | (result["reversal_score"] >= 70)]

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="全部结果", index=False)
        candidates.to_excel(writer, sheet_name="JCE候选", index=False)
        top_candidates.to_excel(writer, sheet_name="重点候选", index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="失败", index=False)
    csv_path = output_path.with_suffix(".csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    show = ["symbol", "priority_score", "primary_signal", "final_recommendation", "jce_entry_score", "reversal_score", "decline_span_days", "decline_down_days", "ignored_small_up_days", "decline_pct", "bull_day_return_pct", "three_month_low_distance_pct", "three_month_low_bonus_10", "four_line_width_pct", "close_to_ma60_pct"]
    print("\nJCE V3.4 最新交易日排行榜：")
    print(result[show].head(args.top).to_string(index=False))
    print(f"\nExcel：{output_path}")
    print(f"CSV：{csv_path}")
    print(f"成功：{len(result)}；失败：{len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

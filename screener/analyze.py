"""
매집 레이더 - 분석

data/daily/*.csv 를 읽어 "가격은 횡보하는데 매집 흔적이 쌓이는" 종목에 점수를 매기고
결과를 docs/data/latest.json 으로 저장합니다. (GitHub Pages 화면이 이 파일을 읽음)

점수 (100점 만점)
  횡보   25점  기간 중 고가-저가 폭이 좁고, 추세선 기울기가 0 에 가까울수록
  OBV    25점  누적 거래량(OBV)이 꾸준히 오를수록
  자금흐름 20점  CMF(차이킨 머니플로)가 양수이고, 오른 날 거래량 > 내린 날 거래량일수록
  수급   30점  외국인+기관 누적 순매수가 시가총액 대비 크고, 순매수한 날이 많을수록

그리고 같은 규칙을 과거 시점에 적용해 "그때 뽑혔던 종목이 이후 어땠는지"도 함께 계산합니다.
"""
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"
OUT_DIR = ROOT / "docs" / "data"

# ---- 조건 (필요하면 여기 숫자만 바꾸면 됩니다) ----
WINDOW = 40              # 판단 기간(거래일). 약 2달
MIN_MCAP = 500e8         # 시가총액 500억 이상만
MIN_AVG_VALUE = 5e8      # 하루 평균 거래대금 5억 이상만
TOP_N = 60               # 화면에 보여 줄 종목 수
CHART_DAYS = 120         # 종목 차트에 보여 줄 기간
BT_TOP = 20              # 과거 검증 때 각 시점 상위 몇 종목을 볼지
BT_HOLD = [5, 20]        # 신호 후 며칠 뒤 수익률을 볼지
BT_STEP = 5              # 과거 검증 시점 간격(거래일)
EXCLUDE_WORDS = ["스팩"]  # 이름에 들어가면 제외
MAX_RANGE = 0.30         # 기간 중 (최고가-최저가)/평균가 가 이보다 크면 '횡보 아님'으로 제외
MAX_TREND = 0.12         # 추세선으로 본 기간 변화율이 ±12% 를 넘으면 제외

KST = dt.timezone(dt.timedelta(hours=9))


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def load_panel():
    files = sorted(DAILY_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("data/daily 에 자료가 없습니다. 먼저 collect.py 를 실행하세요.")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"ticker": str})
        df["date"] = f.stem
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["ticker"] = raw["ticker"].str.zfill(6)
    piv = {}
    for col in ["open", "high", "low", "close", "volume", "value", "mcap",
                "frgn_net", "inst_net", "pension_net"]:
        if col not in raw:
            raw[col] = np.nan
        piv[col] = raw.pivot_table(index="date", columns="ticker", values=col, aggfunc="last").sort_index()
    last = raw.sort_values("date").groupby("ticker").last()
    info = last[["name", "market"]]
    dates = list(piv["close"].index)
    tickers = list(piv["close"].columns)
    for k in piv:
        piv[k] = piv[k].reindex(index=dates, columns=tickers)
    # 거래정지일(시가·고가·저가 0) 처리: 종가로 채움
    for k in ["open", "high", "low"]:
        piv[k] = piv[k].where(piv[k] > 0, piv["close"])
    return piv, info, dates, tickers


def lin_slope_corr(Y):
    """Y: (W, N) 배열. 각 열에 대해 시간에 대한 기울기와 상관계수."""
    W = Y.shape[0]
    x = np.arange(W, dtype=float)
    xc = x - x.mean()
    yc = Y - Y.mean(axis=0)
    sxx = (xc ** 2).sum()
    sxy = (xc[:, None] * yc).sum(axis=0)
    syy = (yc ** 2).sum(axis=0)
    slope = sxy / sxx
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = sxy / np.sqrt(sxx * syy)
    return slope, np.nan_to_num(corr)


def score_at(piv, info, t):
    """t 번째 날짜까지의 WINDOW 일로 전 종목 점수 계산. DataFrame 반환."""
    W = WINDOW
    if t < W:
        return None
    sl = slice(t - W + 1, t + 1)
    C = piv["close"].iloc[sl].to_numpy(float)
    Cprev = piv["close"].iloc[t - W:t + 1].to_numpy(float)  # W+1 행 (OBV 용)
    H = piv["high"].iloc[sl].to_numpy(float)
    L = piv["low"].iloc[sl].to_numpy(float)
    V = piv["volume"].iloc[sl].to_numpy(float)
    VAL = piv["value"].iloc[sl].to_numpy(float)
    F = piv["frgn_net"].iloc[sl].fillna(0).to_numpy(float)
    I = piv["inst_net"].iloc[sl].fillna(0).to_numpy(float)
    P = piv["pension_net"].iloc[sl].fillna(0).to_numpy(float)
    mcap = piv["mcap"].iloc[t].to_numpy(float)
    tickers = piv["close"].columns

    ok = ~np.isnan(Cprev).any(axis=0) & ~np.isnan(V).any(axis=0)
    ok &= np.nan_to_num(mcap) >= MIN_MCAP
    avg_val = np.nanmean(VAL, axis=0)
    ok &= np.nan_to_num(avg_val) >= MIN_AVG_VALUE
    ok &= np.nan_to_num(V[-1]) > 0  # 마지막 날 거래정지 아님
    names = info["name"].reindex(tickers).fillna("").astype(str).to_numpy().astype(str)
    for w in EXCLUDE_WORDS:
        ok &= np.char.find(names, w) < 0

    # 1) 횡보
    mean_c = C.mean(axis=0)
    range_pct = (H.max(axis=0) - L.min(axis=0)) / mean_c
    logc = np.log(np.where(C > 0, C, np.nan))
    slope_c, _ = lin_slope_corr(np.nan_to_num(logc))
    trend_pct = np.expm1(slope_c * (W - 1))          # 추세선으로 본 기간 전체 변화율
    flat_range = clip01((0.30 - range_pct) / (0.30 - 0.10))
    flat_trend = clip01((0.15 - np.abs(trend_pct)) / (0.15 - 0.05))
    s_flat = flat_range * flat_trend

    # 2) OBV
    dirn = np.sign(np.diff(Cprev, axis=0))           # (W, N)
    obv = np.cumsum(dirn * V, axis=0)
    slope_obv, corr_obv = lin_slope_corr(obv)
    vol_sum = V.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        obv_ratio = slope_obv * (W - 1) / vol_sum     # 기간 거래량 중 '순유입'으로 쌓인 비율
    s_obv = clip01(np.nan_to_num(obv_ratio) / 0.25) * clip01(corr_obv)

    # 3) 자금흐름: CMF + 상승일/하락일 거래량
    hl = H - L
    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = np.where(hl > 0, ((C - L) - (H - C)) / hl, 0.0)
        cmf = (mfm * V).sum(axis=0) / vol_sum
        upv = (V * (dirn > 0)).sum(axis=0)
        dnv = (V * (dirn < 0)).sum(axis=0)
        ud = upv / np.where(dnv > 0, dnv, np.nan)
    cmf = np.nan_to_num(cmf)
    ud = np.nan_to_num(ud, nan=2.0)
    s_mf = 0.7 * clip01(cmf / 0.15) + 0.3 * clip01((ud - 1.0) / 0.8)

    # 4) 수급: 외국인 + 기관(기관합계에 연기금 포함)
    smart = F + I
    net_sum = smart.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        net_mcap = net_sum / mcap
    pos_ratio = (smart > 0).mean(axis=0)
    s_flow = 0.6 * clip01(np.nan_to_num(net_mcap) / 0.01) + 0.4 * clip01((pos_ratio - 0.5) / 0.25)

    total = 25 * s_flat + 25 * s_obv + 20 * s_mf + 30 * s_flow
    # 이미 크게 오르거나 내린 종목은 다른 점수가 높아도 후보에서 뺌
    flat_ok = (range_pct <= MAX_RANGE) & (np.abs(trend_pct) <= MAX_TREND)
    total = np.where(flat_ok, total, np.nan)

    df = pd.DataFrame({
        "score": total,
        "s_flat": 25 * s_flat, "s_obv": 25 * s_obv, "s_mf": 20 * s_mf, "s_flow": 30 * s_flow,
        "range_pct": range_pct, "trend_pct": trend_pct,
        "obv_ratio": obv_ratio, "obv_corr": corr_obv,
        "cmf": cmf, "ud_ratio": ud,
        "net_frgn": F.sum(axis=0), "net_inst": I.sum(axis=0), "net_pension": P.sum(axis=0),
        "net_mcap_pct": net_mcap * 100, "pos_ratio": pos_ratio,
        "close": C[-1], "mcap": mcap, "avg_value": avg_val,
    }, index=tickers)
    # 거래 조건(시총·거래대금 등)을 통과한 종목 전체를 돌려줌. 횡보 조건 탈락은 score 가 NaN
    return df[ok]


def backtest(piv, info, dates):
    """과거 시점마다 같은 규칙으로 상위 종목을 뽑고, 이후 수익률을 시장 중앙값과 비교."""
    close = piv["close"]
    n = len(dates)
    maxh = max(BT_HOLD)
    rows = []
    for t in range(WINDOW, n - min(BT_HOLD), BT_STEP):
        sc = score_at(piv, info, t)
        if sc is None or sc["score"].notna().sum() == 0:
            continue
        # 비교 기준(시장)은 거래 조건을 통과한 전체 종목
        top = sc.dropna(subset=["score"]).sort_values("score", ascending=False).head(BT_TOP).index
        row = {"date": dates[t], "count": int(len(top))}
        base = close.iloc[t]
        for h in BT_HOLD:
            if t + h >= n:
                row[f"r{h}"] = None
                row[f"m{h}"] = None
                row[f"win{h}"] = None
                continue
            fwd = close.iloc[t + h] / base - 1
            pick = fwd.reindex(top).dropna()
            univ = fwd.reindex(sc.index).dropna()
            row[f"r{h}"] = round(float(pick.mean() * 100), 2) if len(pick) else None
            row[f"m{h}"] = round(float(univ.median() * 100), 2) if len(univ) else None
            row[f"win{h}"] = round(float((pick > univ.median()).mean() * 100), 1) if len(pick) else None
        rows.append(row)

    summary = {}
    for h in BT_HOLD:
        valid = [r for r in rows if r.get(f"r{h}") is not None]
        if valid:
            summary[str(h)] = {
                "times": len(valid),
                "avg_pick": round(float(np.mean([r[f"r{h}"] for r in valid])), 2),
                "avg_market": round(float(np.mean([r[f"m{h}"] for r in valid])), 2),
                "beat_ratio": round(float(np.mean([r[f"r{h}"] > r[f"m{h}"] for r in valid]) * 100), 1),
            }
    return {"rows": rows[-30:], "summary": summary, "top": BT_TOP, "hold": BT_HOLD}


def series_for(piv, tickers, end_idx):
    start = max(0, end_idx - CHART_DAYS + 1)
    idx = piv["close"].index[start:end_idx + 1]
    out = {}
    for tk in tickers:
        c = piv["close"][tk].iloc[start:end_idx + 1]
        v = piv["volume"][tk].iloc[start:end_idx + 1].fillna(0)
        d = np.sign(c.diff().fillna(0))
        obv = (d * v).cumsum()
        net = (piv["frgn_net"][tk].iloc[start:end_idx + 1].fillna(0)
               + piv["inst_net"][tk].iloc[start:end_idx + 1].fillna(0)).cumsum()
        out[tk] = {
            "c": [None if pd.isna(x) else float(x) for x in c],
            "v": [float(x) for x in v],
            "obv": [float(x) for x in obv],
            "net": [round(float(x) / 1e8, 2) for x in net],  # 억 원
        }
    return [str(x) for x in idx], out


def r(x, n=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


def main():
    piv, info, dates, tickers = load_panel()
    t = len(dates) - 1
    if t < WINDOW:
        raise SystemExit(f"자료가 {len(dates)}일치뿐입니다. 최소 {WINDOW + 1}일치가 필요합니다.")
    allsc = score_at(piv, info, t)
    sc = allsc.dropna(subset=["score"]).sort_values("score", ascending=False)
    top = sc.head(TOP_N)

    cands = []
    for tk, row in top.iterrows():
        item = {"ticker": tk, "name": str(info.at[tk, "name"]), "market": str(info.at[tk, "market"])}
        for k, v in row.items():
            item[k] = r(v, 4 if abs(v or 0) < 100 else 0)
        cands.append(item)

    chart_dates, series = series_for(piv, list(top.index), t)
    bt = backtest(piv, info, dates)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "date": dates[t],
        "generated": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "days_loaded": len(dates),
        "universe": int(len(allsc)),
        "flat_count": int(len(sc)),
        "params": {"window": WINDOW, "min_mcap": MIN_MCAP, "min_avg_value": MIN_AVG_VALUE},
        "candidates": cands,
        "chart_dates": chart_dates,
        "series": series,
        "backtest": bt,
    }
    (OUT_DIR / "latest.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    hist = OUT_DIR / "history"
    hist.mkdir(exist_ok=True)
    (hist / f"{dates[t]}.json").write_text(json.dumps(
        [{"ticker": c["ticker"], "name": c["name"], "score": c["score"], "close": c["close"]} for c in cands[:30]],
        ensure_ascii=False), encoding="utf-8")
    print(f"{dates[t]} 분석 완료: 대상 {len(allsc)}종목, 횡보 통과 {len(sc)}종목, 상위 {len(cands)}종목 저장")


if __name__ == "__main__":
    main()

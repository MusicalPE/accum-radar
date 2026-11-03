"""
매집 레이더 - 분석

data/daily 의 일별 자료를 읽어 "가격은 횡보하는데 매집 흔적이 쌓이는" 종목에 점수를 매기고
결과를 docs/data/latest.json 으로 저장합니다. (GitHub Pages 화면이 이 파일을 읽음)

점수 (100점 만점, 화면에서 비중을 바꿔 볼 수 있음)
  횡보     25점  기간 중 고가-저가 폭이 좁고, 추세선 기울기가 0 에 가까울수록
  OBV      25점  누적 거래량(OBV)이 꾸준히 오를수록
  자금흐름 20점  CMF(차이킨 머니플로)가 양수이고, 오른 날 거래량 > 내린 날 거래량일수록
  수급     30점  외국인+기관 누적 순매수가 시가총액 대비 크고, 순매수한 날이 많을수록

판단 기간 20·40·60 거래일을 모두 계산해 두고 화면에서 고릅니다.
같은 규칙을 과거 시점에 적용해 "그때 뽑혔던 종목이 이후 어땠는지"도 계산합니다.
"""
import datetime as dt
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"
META_PATH = ROOT / "data" / "meta.csv"
OUT_DIR = ROOT / "docs" / "data"

# ---- 조건 (필요하면 여기 숫자만 바꾸면 됩니다) ----
WINDOWS = [20, 40, 60]   # 판단 기간(거래일). 화면에서 고름
DEFAULT_WINDOW = 40
MIN_MCAP = 500e8         # 시가총액 500억 이상만
MIN_AVG_VALUE = 5e8      # 하루 평균 거래대금 5억 이상만
EXCLUDE_WORDS = ["스팩"]  # 이름에 들어가면 제외
MAX_RANGE = 0.30         # (40일 기준) 기간 중 (최고가-최저가)/평균가 가 이보다 크면 '횡보 아님'으로 제외
MAX_TREND = 0.12         # (40일 기준) 추세선으로 본 기간 변화율이 ±12% 를 넘으면 제외
TOP_N = 60               # 차트 자료를 함께 넣을 상위 종목 수
TRACK_TOP = 30           # '신규 진입'·'연속 일수'를 셀 때 기준이 되는 상위 몇 종목
TRACK_DAYS = 30          # 연속 일수를 최대 며칠까지 거슬러 셀지
CHART_DAYS = 120         # 종목 차트에 보여 줄 기간
BT_TOP = 20              # 과거 검증 때 각 시점 상위 몇 종목을 볼지
BT_HOLD = [5, 20, 60]    # 신호 후 며칠 뒤 수익률을 볼지
BT_STEP = 5              # 과거 검증 시점 간격(거래일)
BUCKETS = [(80, 101, "80점 이상"), (70, 80, "70~79점"), (60, 70, "60~69점"), (50, 60, "50~59점"), (0, 50, "50점 미만")]

# 점수별 따로 검증: (열 이름, 화면 이름, 클수록 좋은지)
FACTORS = [
    ("score", "종합 점수", True),
    ("s_flat", "횡보 점수", True),
    ("s_obv", "OBV 점수", True),
    ("s_mf", "자금흐름 점수", True),
    ("s_flow", "수급 점수", True),
    ("range_pct", "가격 폭 (좁을수록)", False),
    ("obv_ratio", "OBV 비율", True),
    ("cmf", "CMF", True),
    ("net_mcap_pct", "외인+기관 순매수 (시총 대비)", True),
    ("pos_ratio", "외인+기관 순매수한 날 비율", True),
]
FACTOR_Q = 0.2           # 상위·하위 몇 % 끼리 비교할지

KST = dt.timezone(dt.timedelta(hours=9))
WEIGHTS = {"s_flat": 25, "s_obv": 25, "s_mf": 20, "s_flow": 30}


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def load_panel():
    files = sorted(list(DAILY_DIR.glob("*.csv.gz")) + list(DAILY_DIR.glob("*.csv")), key=lambda f: f.name[:8])
    if not files:
        raise SystemExit("data/daily 에 자료가 없습니다. 먼저 collect.py 를 실행하세요.")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"ticker": str})
        df["date"] = f.name[:8]
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
    info = last[["name", "market"]].copy()
    dates = list(piv["close"].index)
    tickers = list(piv["close"].columns)
    for k in piv:
        piv[k] = piv[k].reindex(index=dates, columns=tickers)
    # 거래정지일(시가·고가·저가 0) 처리: 종가로 채움
    for k in ["open", "high", "low"]:
        piv[k] = piv[k].where(piv[k] > 0, piv["close"])
    return piv, info, dates, tickers


def load_meta():
    if not META_PATH.exists():
        return pd.DataFrame()
    m = pd.read_csv(META_PATH, dtype={"ticker": str})
    m["ticker"] = m["ticker"].str.zfill(6)
    return m.set_index("ticker")


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
    corr = sxy / np.sqrt(sxx * syy)
    return slope, np.nan_to_num(corr)


def score_at(piv, info, t, W):
    """t 번째 날짜까지의 W 일로 점수 계산.
    거래 조건(시총·거래대금 등)을 통과한 종목 전체를 돌려주고, 횡보 조건에서 빠진 종목은 score 가 NaN."""
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

    # 기간이 짧으면 가격 폭도 자연히 좁으므로 기준을 기간에 맞춰 조절 (40일 기준)
    k = math.sqrt(W / 40)

    # 1) 횡보
    mean_c = C.mean(axis=0)
    range_pct = (H.max(axis=0) - L.min(axis=0)) / mean_c
    logc = np.log(np.where(C > 0, C, np.nan))
    slope_c, _ = lin_slope_corr(np.nan_to_num(logc))
    trend_pct = np.expm1(slope_c * (W - 1))          # 추세선으로 본 기간 전체 변화율
    flat_range = clip01((0.30 * k - range_pct) / (0.20 * k))
    flat_trend = clip01((0.15 * k - np.abs(trend_pct)) / (0.10 * k))
    s_flat = flat_range * flat_trend

    # 2) OBV
    dirn = np.sign(np.diff(Cprev, axis=0))           # (W, N)
    obv = np.cumsum(dirn * V, axis=0)
    slope_obv, corr_obv = lin_slope_corr(obv)
    vol_sum = V.sum(axis=0)
    obv_ratio = slope_obv * (W - 1) / vol_sum         # 기간 거래량 중 '순유입'으로 쌓인 비율
    s_obv = clip01(np.nan_to_num(obv_ratio) / 0.25) * clip01(corr_obv)

    # 3) 자금흐름: CMF + 상승일/하락일 거래량
    hl = H - L
    mfm = np.where(hl > 0, ((C - L) - (H - C)) / np.where(hl > 0, hl, 1), 0.0)
    cmf = np.nan_to_num((mfm * V).sum(axis=0) / vol_sum)
    upv = (V * (dirn > 0)).sum(axis=0)
    dnv = (V * (dirn < 0)).sum(axis=0)
    ud = np.nan_to_num(upv / np.where(dnv > 0, dnv, np.nan), nan=2.0)
    s_mf = 0.7 * clip01(cmf / 0.15) + 0.3 * clip01((ud - 1.0) / 0.8)

    # 4) 수급: 외국인 + 기관(기관합계에 연기금 포함)
    smart = F + I
    net_sum = smart.sum(axis=0)
    net_mcap = net_sum / mcap
    pos_ratio = (smart > 0).mean(axis=0)
    s_flow = 0.6 * clip01(np.nan_to_num(net_mcap) / (0.01 * W / 40)) + 0.4 * clip01((pos_ratio - 0.5) / 0.25)

    total = 25 * s_flat + 25 * s_obv + 20 * s_mf + 30 * s_flow
    # 이미 크게 오르거나 내린 종목은 다른 점수가 높아도 후보에서 뺌
    flat_ok = (range_pct <= MAX_RANGE * k) & (np.abs(trend_pct) <= MAX_TREND * k)
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
    return df[ok]


def ranked(sc):
    return sc.dropna(subset=["score"]).sort_values("score", ascending=False)


def tracking(piv, info, t, W, today):
    """어제 순위, 신규 진입 여부, 상위권 연속 일수."""
    today_top = list(today.index[:TRACK_TOP])
    prev_rank = {}
    streak = {tk: 1 for tk in today_top}
    alive = set(today_top)
    for back in range(1, TRACK_DAYS + 1):
        tt = t - back
        if tt < W or not alive:
            break
        r = ranked(score_at(piv, info, tt, W))
        if back == 1:
            prev_rank = {tk: i + 1 for i, tk in enumerate(r.index)}
        top = set(r.index[:TRACK_TOP])
        still = alive & top
        for tk in still:
            streak[tk] += 1
        alive = still
    return prev_rank, streak


def backtest(piv, info, dates, W):
    """과거 시점마다 같은 규칙으로 상위 종목을 뽑고, 이후 수익률을 시장 중앙값과 비교.
    점수 구간별로도 '시장보다 얼마나 나았는지'를 모아 봄."""
    close = piv["close"]
    n = len(dates)
    rows = []
    bucket_acc = {h: {b[2]: [] for b in BUCKETS} for h in BT_HOLD}
    fac_acc = {h: {f[0]: {"top": [], "bot": [], "ic": []} for f in FACTORS} for h in BT_HOLD}
    for t in range(W, n - min(BT_HOLD), BT_STEP):
        sc = score_at(piv, info, t, W)
        if sc is None:
            continue
        r = ranked(sc)
        if r.empty:
            continue
        top = r.head(BT_TOP).index
        row = {"date": dates[t]}
        base = close.iloc[t]
        for h in BT_HOLD:
            if t + h >= n:
                row[f"r{h}"] = row[f"m{h}"] = None
                continue
            fwd = close.iloc[t + h] / base - 1
            univ = fwd.reindex(sc.index).dropna()   # 비교 기준: 거래 조건을 통과한 전체 종목
            mkt = float(univ.median()) if len(univ) else 0.0
            pick = fwd.reindex(top).dropna()
            row[f"r{h}"] = round(float(pick.mean() * 100), 2) if len(pick) else None
            row[f"m{h}"] = round(mkt * 100, 2)
            fr = fwd.reindex(r.index)
            for lo, hi, label in BUCKETS:
                sel = fr[(r["score"] >= lo) & (r["score"] < hi)].dropna()
                if len(sel):
                    bucket_acc[h][label].append((float(sel.mean() - mkt) * 100, len(sel)))
            # 횡보 통과 종목 안에서 지표 하나만 보고 상위 20% / 하위 20% 를 나눠 비교
            ex = (fr - mkt).dropna()
            if len(ex) >= 25:
                for col, _, higher in FACTORS:
                    v = r[col].reindex(ex.index)
                    if not higher:
                        v = -v
                    rk = v.rank(method="first", pct=True)
                    hi_ex, lo_ex = ex[rk > 1 - FACTOR_Q], ex[rk <= FACTOR_Q]
                    if len(hi_ex) and len(lo_ex):
                        fac_acc[h][col]["top"].append(float(hi_ex.mean()) * 100)
                        fac_acc[h][col]["bot"].append(float(lo_ex.mean()) * 100)
                    ic = v.rank().corr(ex.rank())
                    if not np.isnan(ic):
                        fac_acc[h][col]["ic"].append(float(ic))
        rows.append(row)

    summary = {}
    for h in BT_HOLD:
        valid = [x for x in rows if x.get(f"r{h}") is not None]
        if valid:
            summary[str(h)] = {
                "times": len(valid),
                "avg_pick": round(float(np.mean([x[f"r{h}"] for x in valid])), 2),
                "avg_market": round(float(np.mean([x[f"m{h}"] for x in valid])), 2),
                "beat_ratio": round(float(np.mean([x[f"r{h}"] > x[f"m{h}"] for x in valid]) * 100), 1),
            }
    buckets = {}
    for h in BT_HOLD:
        buckets[str(h)] = []
        for _, _, label in BUCKETS:
            acc = bucket_acc[h][label]
            if acc:
                buckets[str(h)].append({
                    "label": label,
                    "excess": round(float(np.mean([a for a, _ in acc])), 2),
                    "beat": round(float(np.mean([a > 0 for a, _ in acc]) * 100), 1),
                    "times": len(acc),
                    "avg_count": round(float(np.mean([c for _, c in acc])), 1),
                })
    factors = {}
    for h in BT_HOLD:
        lst = []
        for col, label, higher in FACTORS:
            a = fac_acc[h][col]
            if not a["top"]:
                continue
            spread = np.array(a["top"]) - np.array(a["bot"])
            ic = np.array(a["ic"]) if a["ic"] else np.array([0.0])
            # 시점끼리 기간이 겹쳐 실제 독립 표본은 더 적으므로, t 값은 보유 기간만큼 줄여서 계산
            n_eff = max(1.0, len(spread) * BT_STEP / h)
            sd = spread.std(ddof=1) if len(spread) > 1 else 0
            t = float(spread.mean() / (sd / math.sqrt(n_eff))) if sd > 0 else 0.0
            lst.append({
                "key": col, "label": label,
                "top": round(float(np.mean(a["top"])), 2),
                "bot": round(float(np.mean(a["bot"])), 2),
                "spread": round(float(spread.mean()), 2),
                "win": round(float((spread > 0).mean() * 100), 1),
                "ic": round(float(ic.mean()), 3),
                "t": round(t, 2),
                "times": int(len(spread)),
            })
        factors[str(h)] = lst
    first = rows[0]["date"] if rows else None
    return {"rows": rows[-40:], "summary": summary, "buckets": buckets, "factors": factors, "factor_q": FACTOR_Q,
            "top": BT_TOP, "hold": BT_HOLD, "from": first}


def sector_summary(sc, meta, W):
    """업종별로 '횡보 통과 + 60점 이상' 종목 수와 외국인·기관 순매수 합계."""
    if meta.empty or "sector" not in meta:
        return []
    df = sc.join(meta[["sector"]], how="left")
    df["sector"] = df["sector"].fillna("기타")
    df["smart"] = df["net_frgn"] + df["net_inst"]
    out = []
    for sec, g in df.groupby("sector"):
        passed = g.dropna(subset=["score"])
        strong = passed[passed["score"] >= 60]
        mc = g["mcap"].sum()
        out.append({
            "sector": sec,
            "stocks": int(len(g)),
            "flat": int(len(passed)),
            "strong": int(len(strong)),
            "avg_score": round(float(passed["score"].mean()), 1) if len(passed) else None,
            "net_eok": round(float(g["smart"].sum()) / 1e8, 0),
            "net_mcap_pct": round(float(g["smart"].sum() / mc * 100), 3) if mc else None,
            "names": [str(x) for x in strong.sort_values("score", ascending=False).index[:6]],
        })
    out.sort(key=lambda x: (x["strong"], x["net_mcap_pct"] or -9), reverse=True)
    return out


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
            "obv": [float(x) for x in obv],
            "net": [round(float(x) / 1e8, 2) for x in net],  # 억 원
        }
    return [str(x) for x in idx], out


def num(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd if abs(x) < 100 else 0)


FIELDS = ["score", "s_flat", "s_obv", "s_mf", "s_flow", "range_pct", "trend_pct", "obv_ratio",
          "cmf", "ud_ratio", "net_frgn", "net_inst", "net_pension", "net_mcap_pct", "pos_ratio",
          "close", "mcap", "avg_value"]


def main():
    piv, info, dates, tickers = load_panel()
    meta = load_meta()
    t = len(dates) - 1
    if t < min(WINDOWS):
        raise SystemExit(f"자료가 {len(dates)}일치뿐입니다.")

    windows = {}
    chart_set = []
    for W in WINDOWS:
        if t < W:
            continue
        allsc = score_at(piv, info, t, W)
        r = ranked(allsc)
        prev_rank, streak = tracking(piv, info, t, W, r)
        cands = []
        for i, (tk, row) in enumerate(r.iterrows()):
            item = {"ticker": tk, "name": str(info.at[tk, "name"]), "market": str(info.at[tk, "market"]),
                    "rank": i + 1, "prev": prev_rank.get(tk), "streak": streak.get(tk, 0)}
            for f in FIELDS:
                item[f] = num(row[f])
            if not meta.empty and tk in meta.index:
                m = meta.loc[tk]
                item["sector"] = None if pd.isna(m.get("sector")) else str(m.get("sector"))
                for f in ["per", "pbr", "div"]:
                    v = num(m.get(f), 2)
                    item[f] = v if v else None   # KRX 는 적자·자료없음을 0 으로 줌
            cands.append(item)
        for tk in r.index[:TOP_N]:
            if tk not in chart_set:
                chart_set.append(tk)
        windows[str(W)] = {
            "universe": int(len(allsc)),
            "flat_count": int(len(r)),
            "candidates": cands,
            "sectors": sector_summary(allsc, meta, W),
            "backtest": backtest(piv, info, dates, W),
        }
        print(f"{W}일 기준: 대상 {len(allsc)}종목, 횡보 통과 {len(r)}종목")

    chart_dates, series = series_for(piv, chart_set, t)

    # 가상 매매 일지용: 마지막 날 전 종목 종가와 이름
    last_close = piv["close"].iloc[t]
    prices = {tk: [str(info.at[tk, "name"]), float(c)] for tk, c in last_close.items()
              if not pd.isna(c) and c > 0}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "date": dates[t],
        "generated": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "days_loaded": len(dates),
        "first_date": dates[0],
        "default_window": DEFAULT_WINDOW,
        "params": {"min_mcap": MIN_MCAP, "min_avg_value": MIN_AVG_VALUE,
                   "max_range": MAX_RANGE, "max_trend": MAX_TREND, "weights": WEIGHTS,
                   "track_top": TRACK_TOP},
        "meta_date": None if meta.empty else str(meta["date"].iloc[0]),
        "windows": windows,
        "chart_dates": chart_dates,
        "series": series,
        "prices": prices,
    }
    (OUT_DIR / "latest.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    hist = OUT_DIR / "history"
    hist.mkdir(exist_ok=True)
    dw = windows.get(str(DEFAULT_WINDOW)) or next(iter(windows.values()))
    (hist / f"{dates[t]}.json").write_text(json.dumps(
        [{"ticker": c["ticker"], "name": c["name"], "score": c["score"], "close": c["close"]}
         for c in dw["candidates"][:30]], ensure_ascii=False), encoding="utf-8")
    print(f"{dates[t]} 분석 완료")


if __name__ == "__main__":
    main()

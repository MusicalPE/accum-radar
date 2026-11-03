"""
매집 레이더 - 데이터 수집

한국거래소(KRX) 데이터를 pykrx 로 받아 하루에 파일 하나씩 저장합니다.
  data/daily/YYYYMMDD.csv.gz  (하루치 시세·수급)
  data/meta.csv               (가장 최근 날의 업종·PER·PBR·배당)

받는 것 (코스피 + 코스닥 전 종목)
  - 시가/고가/저가/종가/거래량/거래대금
  - 시가총액
  - 외국인 순매수 거래대금, 기관합계 순매수 거래대금, 연기금 순매수 거래대금

KRX 정보데이터시스템이 회원제로 바뀌어 로그인이 필요합니다.
환경변수 KRX_ID, KRX_PW 를 설정해 두면 pykrx 가 알아서 로그인합니다.
(GitHub 에서는 저장소 Settings > Secrets 에 넣어 둡니다.)

사용법
  python screener/collect.py              # 오늘(또는 가장 최근 영업일) 하루
  python screener/collect.py --days 500   # 최근 500일치(약 2년) 중 없는 날만 채우기
  python screener/collect.py --date 20261103
"""
import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"
KEEP_DAYS = 520          # 약 2년치. 이보다 오래된 일별 파일은 지워서 저장소가 커지지 않게
META_PATH = ROOT / "data" / "meta.csv"
MARKETS = ["KOSPI", "KOSDAQ"]
INVESTORS = {"외국인": "frgn_net", "기관합계": "inst_net", "연기금": "pension_net"}
KST = dt.timezone(dt.timedelta(hours=9))


def log(*a):
    print(*a, flush=True)


def retry(fn, *args, tries=3, wait=3, **kw):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:  # 네트워크 오류 등
            last = e
            log(f"  재시도 {i + 1}/{tries}: {e}")
            time.sleep(wait * (i + 1))
    raise last


def fetch_day(day: str) -> pd.DataFrame | None:
    """하루치 전 종목 데이터. 휴장일이면 None."""
    from pykrx import stock

    frames = []
    for mkt in MARKETS:
        ohlcv = retry(stock.get_market_ohlcv_by_ticker, day, mkt)
        # 휴장일은 빈 표 또는 모든 값이 0 인 표가 옴
        if ohlcv is None or ohlcv.empty or (ohlcv[["거래량", "거래대금"]] == 0).all(axis=None):
            return None
        ohlcv = ohlcv.rename(columns={
            "시가": "open", "고가": "high", "저가": "low", "종가": "close",
            "거래량": "volume", "거래대금": "value", "등락률": "chg",
        })
        ohlcv = ohlcv[["open", "high", "low", "close", "volume", "value"]]
        ohlcv["market"] = mkt

        cap = retry(stock.get_market_cap_by_ticker, day, mkt)
        if cap is not None and not cap.empty:
            ohlcv["mcap"] = cap["시가총액"].reindex(ohlcv.index)
            ohlcv["shares"] = cap["상장주식수"].reindex(ohlcv.index)

        names = {}
        for inv, col in INVESTORS.items():
            net = retry(stock.get_market_net_purchases_of_equities, day, day, mkt, inv)
            if net is None or net.empty:
                ohlcv[col] = 0
                continue
            ohlcv[col] = net["순매수거래대금"].reindex(ohlcv.index).fillna(0)
            names.update(net["종목명"].to_dict())
            time.sleep(0.5)

        # 종목명: 순매수 표에 없던 종목은 하나씩 조회 (보통 몇 개 안 됨)
        missing = [t for t in ohlcv.index if t not in names]
        for t in missing:
            try:
                names[t] = stock.get_market_ticker_name(t)
            except Exception:
                names[t] = t
        ohlcv["name"] = [names.get(t, t) for t in ohlcv.index]
        frames.append(ohlcv)
        time.sleep(1)

    if not frames:
        return None
    df = pd.concat(frames)
    df.index.name = "ticker"
    return df


def business_days_back(n: int, end: dt.date):
    d = end
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return list(reversed(out))


def fetch_meta(day: str) -> pd.DataFrame | None:
    """업종 분류와 PER·PBR·배당수익률 (가장 최근 날 하루치만)."""
    from pykrx import stock

    frames = []
    for mkt in MARKETS:
        part = pd.DataFrame()
        try:
            sec = retry(stock.get_market_sector_classifications, day, mkt)
            if sec is not None and not sec.empty:
                part = pd.DataFrame({"sector": sec["업종명"]})
        except Exception as e:
            log(f"  업종 분류를 받지 못했습니다({mkt}): {e}")
        try:
            fu = retry(stock.get_market_fundamental_by_ticker, day, mkt)
            if fu is not None and not fu.empty:
                fu = fu.rename(columns={"PER": "per", "PBR": "pbr", "DIV": "div", "EPS": "eps", "BPS": "bps"})
                part = part.join(fu[["per", "pbr", "div", "eps", "bps"]], how="outer")
        except Exception as e:
            log(f"  PER·PBR 을 받지 못했습니다({mkt}): {e}")
        if not part.empty:
            frames.append(part)
        time.sleep(1)
    if not frames:
        return None
    df = pd.concat(frames)
    df.index.name = "ticker"
    df["date"] = day
    return df


def daily_path(key: str) -> Path:
    return DAILY_DIR / f"{key}.csv.gz"


def prune():
    files = sorted(list(DAILY_DIR.glob("*.csv")) + list(DAILY_DIR.glob("*.csv.gz")), key=lambda f: f.name[:8])
    for f in files[:-KEEP_DAYS]:
        f.unlink()
        log("오래된 파일 삭제:", f.name)
    if files[:-KEEP_DAYS]:
        oldest = files[-KEEP_DAYS].name[:8]
        for h in DAILY_DIR.glob("*.holiday"):
            if h.stem < oldest:
                h.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=1, help="최근 며칠(평일 기준)을 채울지")
    p.add_argument("--date", help="특정 날짜 하루 (YYYYMMDD)")
    p.add_argument("--force", action="store_true", help="이미 있는 파일도 다시 받기")
    args = p.parse_args()

    if not os.getenv("KRX_ID") or not os.getenv("KRX_PW"):
        log("KRX_ID / KRX_PW 가 설정되지 않았습니다. 저장소 Settings > Secrets 에 넣어 주세요.")
        sys.exit(1)
    # 로그인이 안 되면 모든 날이 빈 자료로 와서 휴장일로 잘못 기록되므로, 먼저 확인하고 멈춤
    from pykrx.website.comm.auth import get_auth_session
    if get_auth_session() is None:
        log("KRX 로그인에 실패했습니다. 아이디·비밀번호를 확인해 주세요.")
        sys.exit(1)

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(KST)
    if args.date:
        days = [dt.datetime.strptime(args.date, "%Y%m%d").date()]
    else:
        end = now.date()
        # 투자자별 매매 확정치는 18시 이후에 나오므로, 그 전에는 전날까지만
        if now.hour < 18:
            end -= dt.timedelta(days=1)
        days = business_days_back(args.days, end)

    got, fails, streak = 0, 0, 0
    for d in days:
        key = d.strftime("%Y%m%d")
        path = daily_path(key)
        old_plain = DAILY_DIR / f"{key}.csv"
        holiday_mark = DAILY_DIR / f"{key}.holiday"
        if not args.force and (path.exists() or old_plain.exists() or holiday_mark.exists()):
            continue
        log(f"{key} 수집 중...")
        try:
            df = fetch_day(key)
        except Exception as e:
            # 하루 실패로 긴 수집 전체를 버리지 않도록 건너뛰고 계속. 다음 실행 때 다시 시도됨
            fails += 1
            streak += 1
            log(f"  {key}: 실패 ({e})")
            if streak >= 5:
                log("연속 5번 실패해서 멈춥니다. 지금까지 받은 자료는 저장됩니다.")
                break
            continue
        streak = 0
        if df is None:
            log(f"  {key}: 휴장일(또는 자료 없음)")
            if d < now.date():
                holiday_mark.write_text("")
            continue
        df.to_csv(path, encoding="utf-8", compression="gzip")
        if old_plain.exists():
            old_plain.unlink()
        got += 1
        log(f"  {key}: {len(df)} 종목 저장")

    # 업종·PER·PBR 은 가장 최근에 자료가 있는 날 기준으로 매번 새로 받음
    latest = sorted(f.name[:8] for f in list(DAILY_DIR.glob("*.csv.gz")) + list(DAILY_DIR.glob("*.csv")))
    if latest:
        try:
            meta = fetch_meta(latest[-1])
            if meta is not None:
                meta.to_csv(META_PATH, encoding="utf-8")
                log(f"업종·기본지표 저장: {len(meta)} 종목 ({latest[-1]})")
        except Exception as e:
            log(f"업종·기본지표를 받지 못했습니다: {e}")

    prune()
    log(f"끝. 새로 받은 날: {got}, 실패: {fails}")
    if fails and got == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
매집 레이더 - 텔레그램 알림

분석이 끝난 뒤 docs/data/latest.json 을 읽어 상위 종목과 새로 들어온 종목을 텔레그램으로 보냅니다.
환경변수 TELEGRAM_TOKEN, TELEGRAM_CHAT_ID 가 없으면 아무것도 하지 않고 끝납니다.
결과 화면 주소는 PAGE_URL 환경변수(없으면 생략)로 받습니다.
"""
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "docs" / "data" / "latest.json"
SHOW = 10


def eok(x):
    return f"{(x or 0) / 1e8:+,.0f}억"


def build_message(d):
    w = str(d.get("default_window", 40))
    win = d["windows"].get(w) or next(iter(d["windows"].values()))
    cands = win["candidates"]
    top_n = d["params"].get("track_top", 30)
    date = f'{d["date"][:4]}.{d["date"][4:6]}.{d["date"][6:]}'
    lines = [f"매집 레이더 {date} (판단 기간 {w}거래일)",
             f"대상 {win['universe']:,}종목 중 횡보 {win['flat_count']}종목", ""]

    lines.append(f"상위 {SHOW}")
    for c in cands[:SHOW]:
        mark = "신규 " if c.get("prev") is None or c["prev"] > top_n else ""
        streak = c.get("streak") or 0
        stay = " 30일 넘게" if streak > 30 else (f" {streak}일째" if streak >= 3 else "")
        sector = f" [{c['sector']}]" if c.get("sector") else ""
        smart = (c.get("net_frgn") or 0) + (c.get("net_inst") or 0)
        lines.append(f"{c['rank']}. {mark}{c['name']}{sector} {c['score']:.0f}점{stay} / 외인+기관 {eok(smart)}")

    new = [c for c in cands[:top_n] if c.get("prev") is None or c["prev"] > top_n]
    new = [c for c in new if c["rank"] > SHOW]
    if new:
        lines += ["", f"상위 {top_n}에 새로 들어온 종목"]
        lines.append(", ".join(f"{c['name']}({c['rank']}위)" for c in new[:10]))

    url = os.getenv("PAGE_URL")
    if url:
        lines += ["", url]
    lines += ["", "후보를 좁히는 참고 자료이며 매매 추천이 아닙니다."]
    return "\n".join(lines)


def main():
    token = os.getenv("TELEGRAM_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("텔레그램 설정이 없어 알림을 건너뜁니다.")
        return
    d = json.loads(LATEST.read_text(encoding="utf-8"))
    text = build_message(d)
    if "--dry" in sys.argv:
        print(text)
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text, "disable_web_page_preview": "true"}, timeout=20)
    if r.ok:
        print("텔레그램 알림 보냄")
    else:
        # 알림 실패로 전체 작업을 실패 처리하지는 않음
        print(f"텔레그램 알림 실패: {r.status_code} {r.text[:200]}")


if __name__ == "__main__":
    main()

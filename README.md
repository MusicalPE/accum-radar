# 매집 레이더

가격은 크게 오르지 않았는데 매수 흔적이 꾸준히 쌓이는 국내 주식(코스피·코스닥)을 매일 골라 보여 주는 개인용 도구입니다.

- 횡보(25점): 최근 40거래일 가격 폭이 좁고 추세가 평평한지
- OBV(25점): 누적 거래량이 꾸준히 오르는지
- 자금흐름(20점): CMF와 상승일·하락일 거래량 비교
- 수급(30점): 외국인과 기관의 누적 순매수

평일 장 마감 후 GitHub Actions가 한국거래소 자료를 받아 점수를 다시 계산하고, 결과는 GitHub Pages(docs 폴더)에서 봅니다.
같은 규칙을 과거 시점에 적용했을 때의 결과도 함께 보여 줍니다.

투자 권유가 아니며, 이 도구의 결과로 생긴 손익은 사용하는 사람에게 책임이 있습니다.

## 구성

- screener/collect.py: 하루치 전 종목 시세·시가총액·투자자별 순매수(data/daily), 최근 업종·PER·PBR(data/meta.csv) 수집
- screener/analyze.py: 점수 계산과 과거 검증 (docs/data/latest.json). 조건 숫자는 파일 위쪽에서 바꿀 수 있음
- screener/notify.py: 갱신 후 텔레그램 알림 (설정했을 때만)
- docs/index.html: 결과 화면 (판단 기간·점수 비중 바꾸기, 업종별 보기, 관심 종목, 가상 매매 일지, 과거 검증)
- .github/workflows/daily.yml: 매일 자동 실행

## 필요한 설정

저장소 Settings > Secrets and variables > Actions 에 한국거래소(KRX Data Marketplace) 계정을 넣습니다.

- KRX_ID
- KRX_PW
- TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (알림을 받을 때만)

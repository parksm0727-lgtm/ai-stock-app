import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from google import genai
from ta.momentum import RSIIndicator
from datetime import date, datetime
import pandas as pd
import numpy as np
import os
import uuid
import json

# =========================================================
# [1] 페이지 설정
# =========================================================
st.set_page_config(page_title="AI 텐배거 프로", layout="centered", page_icon="📈")

# =========================================================
# [2] 앱 테마 설정
# =========================================================
def ensure_theme_config():
    if st.session_state.get("_theme_checked"):
        return
    st.session_state["_theme_checked"] = True

    config_path = os.path.join(".streamlit", "config.toml")
    theme_config = (
        "[theme]\n"
        'base="dark"\n'
        'backgroundColor="#0B1120"\n'
        'secondaryBackgroundColor="#141B2E"\n'
        'textColor="#ECEFF4"\n'
        'primaryColor="#5B9DF9"\n'
    )
    try:
        already_dark = os.path.exists(config_path) and 'base="dark"' in open(config_path, encoding="utf-8").read()
        if not already_dark:
            os.makedirs(".streamlit", exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(theme_config)
            st.rerun()
    except OSError:
        pass

ensure_theme_config()

# =========================================================
# [3] 디자인 시스템
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #0B1120;
    --surface: #141B2E;
    --surface-2: #1B2438;
    --border: #262F45;
    --text: #ECEFF4;
    --text-muted: #8A94AC;
    --accent: #5B9DF9;
    --accent-strong: #3B82F6;
    --up: #F87171;    /* 상승: 빨간색 */
    --down: #60A5FA;  /* 하락: 파란색 */
    --radius: 8px;
}

html, body, .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Noto Sans KR', -apple-system, sans-serif !important;
}
h2, h3, h4, h5, h6, p, label, span, div { color: var(--text); }

.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
    max-width: 720px !important;
}
[data-testid="stVerticalBlock"] { gap: 0.1rem !important; }
[data-testid="stElementContainer"] { margin-bottom: 0 !important; }

/* 타이틀 배너 */
.title-banner {
    background: linear-gradient(135deg, #1E293B 0%, #1D4ED8 50%, #3B82F6 100%);
    color: #ffffff;
    text-align: center;
    font-weight: 800;
    font-size: clamp(0.95rem, 4.8vw, 1.6rem) !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    padding: 12px 10px !important;
    border-radius: 12px;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    letter-spacing: -0.5px;
    width: 100% !important;
    display: block !important;
    border: 1.5px solid #38BDF8;
    line-height: 1.2 !important;
    box-sizing: border-box !important;
}

.hero-price {
    padding: 0px 0 6px 0 !important;
    border-bottom: 1px solid var(--border);
    margin-bottom: 6px !important;
}
.hero-label { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0px; }
.hero-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.7rem, 7vw, 2.2rem);
    font-weight: 700;
    line-height: 1.1;
    color: var(--text);
}
.hero-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; margin-top: 2px; }

.mini-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin: 2px 0 6px 0;
}
.mini-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 6px 8px;
}
.mini-label { color: var(--text-muted); font-size: 0.65rem; margin-bottom: 2px; }
.mini-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1rem; color: var(--text); white-space: nowrap; }
.mini-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; margin-top: 2px; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    width: 100% !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 2px !important;
    gap: 2px !important;
    box-sizing: border-box !important;
}

.stTabs [data-baseweb="tab-list"] button {
    flex-grow: 1 !important;
    flex-shrink: 1 !important;
    flex-basis: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    text-align: center !important;
    justify-content: center !important;
    align-items: center !important;
    color: var(--text-muted) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 6px 0px !important;
    margin: 0 !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: var(--surface-2) !important;
    color: var(--accent) !important;
}

[data-baseweb="tab-highlight"] { display: none !important; }

[data-baseweb="select"] > div { min-height: 2.2rem !important; background-color: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stExpander"] { background-color: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
[data-testid="stExpander"] summary p { font-size: 0.85rem !important; padding: 4px 0 !important; }

.stButton>button {
    background: var(--accent-strong) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}
button[kind="primary"] { background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# [4] 파일 기반 영구 저장소
# =========================================================
WATCHLIST_FILE = "watchlist.json"
REPORT_FILE = "ai_reports.json"
RECOMMEND_FILE = "ai_recommends.json"
DEFAULT_WATCHLIST = ["ASTS", "OKLO", "IONQ", "RXRX", "PLTR", "TSLA"]

def load_json_file(filename, default_val):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default_val

def save_json_file(filename, data):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"저장 실패: {e}")

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_json_file(WATCHLIST_FILE, DEFAULT_WATCHLIST.copy())

if "current_ticker" not in st.session_state or st.session_state["current_ticker"] not in st.session_state["watchlist"]:
    st.session_state["current_ticker"] = st.session_state["watchlist"][0]

if "ai_report_cache" not in st.session_state:
    st.session_state["ai_report_cache"] = load_json_file(REPORT_FILE, {})
if "ai_recommend_cache" not in st.session_state:
    st.session_state["ai_recommend_cache"] = load_json_file(RECOMMEND_FILE, {})

JOURNAL_FILE = "trading_journal.csv"
JOURNAL_COLUMNS = ["ID", "Date", "Ticker", "Action", "Price", "Reason"]

# =========================================================
# [5] 데이터 로딩 & AI 호출
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(t: str) -> pd.DataFrame:
    try:
        tk = yf.Ticker(t)
        data = tk.history(period="max", auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        data.reset_index(inplace=True)
        data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def is_valid_ticker(t: str) -> bool:
    try: return not yf.Ticker(t).history(period="5d").empty
    except: return False

@st.cache_data(ttl=1800, show_spinner=False)
def load_news(t: str) -> list:
    try: return yf.Ticker(t).news or []
    except: return []

@st.cache_data(ttl=3600, show_spinner=False)
def run_forecast(df_train: pd.DataFrame, years: int) -> pd.DataFrame:
    m = Prophet(daily_seasonality=False)
    m.fit(df_train)
    return m.predict(m.make_future_dataframe(periods=years * 365))

def get_best_available_model(client: genai.Client) -> str:
    try:
        models = [m.name for m in client.models.list() if "generateContent" in (m.supported_actions or [])]
        for m_name in models:
            clean_name = m_name.replace("models/", "")
            if "flash" in clean_name and ("3.7" in clean_name or "3.6" in clean_name or "3.0" in clean_name):
                return clean_name
        for m_name in models:
            clean_name = m_name.replace("models/", "")
            if "flash" in clean_name:
                return clean_name
    except Exception:
        pass
    return "gemini-1.5-flash"

def get_ai_text(api_key: str, preferred_model: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    best_model = get_best_available_model(client)
    
    target_models = list(dict.fromkeys([best_model, preferred_model, "gemini-1.5-flash"]))
    last_err = None
    for m in target_models:
        if not m: continue
        try:
            return client.models.generate_content(model=m, contents=prompt).text
        except Exception as e:
            last_err = e
            continue
    raise last_err

def get_active_gemini_key(sidebar_key: str) -> str:
    return sidebar_key or os.environ.get("GEMINI_API_KEY", "")

# 💡 초전문가용 다차원 주가 분석 알고리즘 (4대 분석 관점 통합)
@st.cache_data(ttl=1800, show_spinner=False)
def generate_chart_analysis(t: str, cur_price: float, delta_pct: float, rsi: float, mdd: float, api_key: str, model_n: str) -> tuple:
    if api_key:
        prompt = (
            f"당신은 글로벌 헤지펀드의 수석 퀀트 및 펀더멘털 분석가입니다. {date.today().year}년 현재 시점 미주 종목 '{t}'를 입체 분석해주세요.\n"
            f"- 현재가: ${cur_price:.2f} (전일대비 {delta_pct:+.2f}%)\n"
            f"- RSI(14): {rsi:.0f}, MDD: {mdd:.1f}%\n\n"
            f"다음 2가지 구조로 다방면의 초전문가적 분석 결과를 작성하세요.\n\n"
            f"[원인]\n"
            f"1) 펀더멘털/산업 모멘텀: {t}의 핵심 기술 경쟁력, 매출 성장 및 섹터 모멘텀 이슈\n"
            f"2) 수급 및 퀀트 지표: 기술적 파동, RSI {rsi:.0f} 및 고점 대비 낙폭({mdd:.1f}%) 수급 소화 메커니즘\n\n"
            f"[관점]\n"
            f"1) Prophet AI 예측 및 시클리컬 전망: 중장기 궤적과 거시 경제 영향\n"
            f"2) 트레이딩 시나리오: 핵심 지지/저항 라인 중심의 목표가/손절가 및 분할 진입 전략\n\n"
            f"구체적이고 통찰력 있는 금융 어휘를 사용하여 각 항목을 작성하고, 반드시 '[원인]'과 '[관점]' 태그를 지켜주세요."
        )
        try:
            res = get_ai_text(api_key, model_n, prompt)
            reason_p, view_p = "", ""
            if "[원인]" in res and "[관점]" in res:
                parts = res.split("[관점]")
                reason_p = parts[0].replace("[원인]", "").strip()
                view_p = parts[1].strip()
            if reason_p and view_p:
                return reason_p, view_p
        except Exception:
            pass

    # 💡 고도화된 초전문가 다차원 프로필 백업 엔진
    expert_profiles = {
        "ASTS": (
            f"• **펀더멘털 & 산업**: 저궤도(LEO) Direct-to-Cell 위성 통신망 구축 및 글로벌 MNO(AT&T, Verizon)와의 독점적 파트너십 상용화 모멘텀이 상방 모멘텀을 주도합니다.\n"
            f"• **수급 & 퀀트**: 최근 급등에 따른 차익 실현 물량이 소화되는 수급 조정 단계이며, RSI {rsi:.0f} 지표는 과열이 해소되어 50일 이동평균선 지지력을 재시험하는 기술적 횡보 구간입니다.",
            f"• **시클리컬 전망**: 상업용 위성 발사 성공 및 주파수 승인 촉매 유효시 우상향 파동 재개가 예상됩니다.\n"
            f"• **트레이딩 전략**: MDD {mdd:.1f}% 수준의 변동성을 감안하여, 주요 이평선 지지 확인 후 손절 라인 제한 방식의 단계적 분할 매수 전략이 정석입니다."
        ),
        "OKLO": (
            f"• **펀더멘털 & 산업**: 빅테크 AI 데이터센터의 전력 수요 급증에 대응하는 차세대 소형모듈원자로(SMR) 테마 수혜주로, NRC 인허가 절차 진행 상황이 핵심 밸류에이션 변수입니다.\n"
            f"• **수급 & 퀀트**: 고P/E 성장주 수급 이동과 연계된 단기 조정으로, 전일대비 {delta_pct:+.2f}% 변동과 RSI {rsi:.0f} 지표는 기간 조정을 통한 기술적 마디가 형성 과정입니다.",
            f"• **시클리컬 전망**: 2030년 전력 공급 개시 전까지 인허가 뉴스 흐름에 따른 변동성 국면이 지속될 전망입니다.\n"
            f"• **트레이딩 전략**: 하방 지지선 연동 확인 후 단기 매물대 돌파 여부에 따라 목표가를 상향하는 관망 후 분할 진입이 권장됩니다."
        ),
        "IONQ": (
            f"• **펀더멘털 & 산업**: 바륨 기반 이온트랩 양자 컴퓨팅 알고리즘 성과 및 정부/기업 공급 계약 확대가 중장기 펀더멘털의 핵심 축을 구성합니다.\n"
            f"• **수급 & 퀀트**: 성장주 장세의 리스크 오프 기조에 따른 매물 출회로 MDD {mdd:.1f}%를 기록 중이며, RSI {rsi:.0f} 수준은 하방 압력이 완화된 기술적 저점 다지기 상태입니다.",
            f"• **시클리컬 전망**: Prophet 모델 기반 우상향 궤적을 그리나, 양자 기술 상용화 시점까지 장기 파동 특성을 보입니다.\n"
            f"• **트레이딩 전략**: 실적 발표 및 주요 학회 성과 공개 전 리스크 관리를 병행한 모아가는 분할 매수 접근이 적합합니다."
        ),
        "TSLA": (
            f"• **펀더멘털 & 산업**: FSD v13 상용화, 로보택시 및 2만 달러대 보급형 플랫폼 투입 기대감이 밸류에이션 하단을 강력히 방어하고 있습니다.\n"
            f"• **수급 & 퀀트**: 분기 인도량 수치 및 AI 칩 수급 이슈로 전일 {delta_pct:+.2f}% 조정을 나타냈으며, RSI {rsi:.0f} 지표는 박스권 하단 수급 탐색 국면입니다.",
            f"• **시클리컬 전망**: 자율주행 소프트웨어 라이선싱 수입 가시화 시 강한 멀티플 재평가가 기대됩니다.\n"
            f"• **트레이딩 전략**: 단기 지수 변동성을 활용하여 기술적 지지대 진입 시 분할 접근하고 손절가를 명확히 설정하는 것이 유효합니다."
        ),
        "RXRX": (
            f"• **펀더멘털 & 산업**: 엔비디아 BioNeMo 플랫폼과의 협력 기반 AI 신약 개발 알고리즘 및 파이프라인 임상 성과가 핵심 모멘텀입니다.\n"
            f"• **수급 & 퀀트**: 임상 결과 발표 전 거래량 감소 속 눌림목이 심화되었으며, RSI {rsi:.0f} 상태는 하방 리스크가 상당 부분 선반영된 구간입니다.",
            f"• **시클리컬 전망**: 임상 진행 경과에 따른 바이오 특유의 갭상승/하락 파동이 예상됩니다.\n"
            f"• **트레이딩 전략**: 단기 급등 시 차익 실현과 하단 분할 매수 상호 전략을 적용하는 포트폴리오 관리가 요구됩니다."
        ),
        "PLTR": (
            f"• **펀더멘털 & 산업**: AIP(인공지능 플랫폼) 고성장에 따른 민간 및 정부향 상업용 매출 급증이 펀더멘털의 강력한 견인차입니다.\n"
            f"• **수급 & 퀀트**: 높은 멀티플 부담에 따른 기관 차익 실현 물량이 출회되었으나, 기본 수급 체질은 견고하게 유지되고 있습니다.",
            f"• **시클리컬 전망**: AI 엔터프라이즈 전환 주도기업으로서 밸류에이션 보정 완료 후 지속적 우상향 추세가 유효합니다.\n"
            f"• **트레이딩 전략**: RSI 지표 기반 눌림목 발생 시 지지선 확인 후 단계적 포지션을 확대하는 전략이 권장됩니다."
        )
    }

    if t in expert_profiles:
        return expert_profiles[t][0], expert_profiles[t][1]
    
    default_reason = (
        f"• **펀더멘털 & 산업**: {t} 기업 고유의 비즈니스 모멘텀과 기술주 장세 수급 연동성이 직접 반영되는 국면입니다.\n"
        f"• **수급 & 퀀트**: 전일 대비 {delta_pct:+.2f}% 변동 속에 RSI {rsi:.0f} 수치는 기술적 수급 균형점을 탐색 중입니다."
    )
    default_view = (
        f"• **시클리컬 전망**: Prophet 예측 궤적상 중장기 방향성은 유지되며 지수 환경 영향을 수반합니다.\n"
        f"• **트레이딩 전략**: 주요 마디가 지지 여부를 확인 후 위험 대비 보상 비율을 고려한 분할 접근이 적합합니다."
    )
    return default_reason, default_view

def render_mini_grid(cards: list):
    parts = []
    for c in cards:
        tag_html = f'<div class="mini-tag" style="color:{c.get("tag_color", "var(--text-muted)")};">{c["tag"]}</div>' if c.get("tag") else ""
        parts.append(f'<div class="mini-card"><div class="mini-label">{c["label"]}</div><div class="mini-value">{c["value"]}</div>{tag_html}</div>')
    st.markdown(f'<div class="mini-grid">{"".join(parts)}</div>', unsafe_allow_html=True)

def on_ticker_change():
    st.session_state["current_ticker"] = st.session_state["ticker_select_box"]

# =========================================================
# [6] 사이드바
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ 시스템 설정")
    api_key_input = st.text_input("Gemini API Key", type="password")
    model_name = st.text_input("Gemini 모델명 (기본: 자동 탐색)", value="auto")

# =========================================================
# [7] 메인 타이틀 & 종목 선택
# =========================================================
st.markdown('<div class="title-banner">📈 AI 텐배거 발굴기 Pro</div>', unsafe_allow_html=True)

selected_index = st.session_state["watchlist"].index(st.session_state["current_ticker"]) if st.session_state["current_ticker"] in st.session_state["watchlist"] else 0

ticker = st.selectbox(
    "🔍 분석 대상 종목", 
    st.session_state["watchlist"], 
    index=selected_index, 
    key="ticker_select_box",
    on_change=on_ticker_change,
    label_visibility="collapsed"
)

with st.expander("➕ 종목 관리"):
    new_ticker = st.text_input("새 종목 코드 추가", placeholder="예: NVDA", label_visibility="collapsed")
    if st.button("종목 추가", use_container_width=True):
        t = new_ticker.upper().strip()
        if t and is_valid_ticker(t) and t not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(t)
            st.session_state["current_ticker"] = t
            save_json_file(WATCHLIST_FILE, st.session_state["watchlist"])
            st.rerun()

    del_ticker = st.selectbox("삭제할 종목 선택", st.session_state["watchlist"], key="del_ticker_main", label_visibility="collapsed")
    if st.button("종목 삭제", use_container_width=True, disabled=len(st.session_state["watchlist"]) <= 1):
        st.session_state["watchlist"].remove(del_ticker)
        if st.session_state["current_ticker"] == del_ticker: 
            st.session_state["current_ticker"] = st.session_state["watchlist"][0]
        save_json_file(WATCHLIST_FILE, st.session_state["watchlist"])
        st.rerun()

with st.spinner("최신 주가 데이터 로딩 중..."):
    data = load_price_data(ticker)

tab1, tab2, tab3, tab4 = st.tabs(["📈 차트", "🧠 리포트", "🌟 추천", "📝 일지"])

# ========================================================
# TAB 1: 차트 & 초전문가 입체 분석
# ========================================================
with tab1:
    if data.empty:
        st.error(f"'{ticker}' 최신 데이터를 불러올 수 없습니다. 종목 코드를 확인해주세요.")
    else:
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        current_price = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
        delta = current_price - prev_close
        delta_pct = (delta / prev_close * 100) if prev_close else 0.0
        
        delta_color = "var(--up)" if delta >= 0 else "var(--down)"
        
        st.markdown(
            f'<div class="hero-price"><div class="hero-label">{ticker} 최신가</div>'
            f'<div class="hero-value">${current_price:,.2f}</div>'
            f'<div class="hero-delta" style="color:{delta_color};">{"▲" if delta >= 0 else "▼"} {abs(delta):,.2f} ({abs(delta_pct):.2f}%)</div></div>',
            unsafe_allow_html=True,
        )

        rsi_val = data["RSI"].iloc[-1]
        rsi_state = "과매수" if rsi_val >= 70 else ("과매도" if rsi_val <= 30 else "중립")
        mdd_val = (data['Close'] / data['Close'].cummax() - 1.0).min() * 100

        rsi_tag_color = "var(--up)" if rsi_state == "과매수" else ("var(--down)" if rsi_state == "과매도" else "var(--text-muted)")

        render_mini_grid([
            {"label": "RSI(14)", "value": f"{rsi_val:.0f}", "tag": rsi_state, "tag_color": rsi_tag_color},
            {"label": "MDD", "value": f"{mdd_val:.1f}%"},
            {"label": "52주 최고가", "value": f"${data['Close'].tail(252).max():,.1f}"},
        ])

        years = st.slider("미래 예측 기간 (년)", 1, 5, 2, label_visibility="collapsed")
        df_train = data[["Date", "Close"]].copy().rename(columns={"Date": "ds", "Close": "y"})
        
        with st.spinner("예측 중..."):
            forecast = run_forecast(df_train, years)

        fig_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        fig_chart.add_trace(go.Scatter(x=df_train["ds"], y=df_train["y"], mode="markers", marker=dict(color="#64748B", size=2)), row=1, col=1)
        fig_chart.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", line=dict(color="#F43F5E", width=2)), row=1, col=1)
        fig_chart.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], mode="lines", line=dict(color="#A78BFA", width=1)), row=2, col=1)

        fig_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=5, b=5), showlegend=False, height=240, 
        )
        fig_chart.update_xaxes(showgrid=True, gridcolor="#1E293B", tickfont=dict(color="#ECEFF4", size=9))
        fig_chart.update_yaxes(showgrid=True, gridcolor="#1E293B", tickfont=dict(color="#ECEFF4", size=9))
        st.plotly_chart(fig_chart, use_container_width=True)

        active_key = get_active_gemini_key(api_key_input)
        reason_msg, view_msg = generate_chart_analysis(ticker, current_price, delta_pct, rsi_val, mdd_val, active_key, model_name)

        trend_desc = "상승 강세" if delta >= 0 else "하락 조정"
        st.markdown("---")
        st.markdown(f"### 📊 {ticker} 입체 주가 분석 및 시나리오")
        
        st.markdown(f"""
**1. 현재 차트 및 지표**
* **현재가**: **${current_price:,.2f}** (전일 대비 **{delta_pct:+.2f}%** {trend_desc})
* **RSI 지표**: **{rsi_val:.0f}** ({rsi_state} 구간)
* **52주 고점 대비 낙폭(MDD)**: **{mdd_val:.1f}%**

**2. 다차원 변동 원인 분석**
{reason_msg}

**3. 향후 주가 예측 및 트레이딩 관점**
{view_msg}
        """)

# ========================================================
# TAB 2: AI 리포트
# ========================================================
with tab2:
    active_key = get_active_gemini_key(api_key_input)
    if st.button("🔥 AI 심층 리포트 생성", use_container_width=True, type="primary"):
        if not active_key:
            st.error("API 키를 입력해 주세요.")
        else:
            with st.spinner("분석 중..."):
                recent_news = load_news(ticker)[:10]
                news_items = [f"- {item.get('content', {}).get('title') or item.get('title', '제목 없음')}" for item in recent_news]
                news_text = "\n".join(news_items) if news_items else "최신 뉴스가 없습니다."
                prompt = f"현재 {date.today().year}년 기준. 종목 '{ticker}' 관련 뉴스:\n{news_text}\n핵심 단기 촉매와 리스크 요약."
                try: 
                    res_text = get_ai_text(active_key, model_name, prompt)
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state["ai_report_cache"][ticker] = {
                        "created_at": now_str,
                        "content": res_text
                    }
                    save_json_file(REPORT_FILE, st.session_state["ai_report_cache"])
                except Exception as e: 
                    st.error(f"오류 발생: {e}")

    if ticker in st.session_state["ai_report_cache"]:
        item = st.session_state["ai_report_cache"][ticker]
        st.caption(f"📅 **생성 일시:** `{item['created_at']}`")
        st.markdown(item["content"])

# ========================================================
# TAB 3: AI 자율 추천
# ========================================================
with tab3:
    sector_options = [
        "🤖 AI 자율 분야 발굴 (최신 글로벌 뉴스 기반)",
        "우주 항공 및 통신", 
        "AI 바이오 헬스케어", 
        "차세대 에너지 (SMR)", 
        "양자 컴퓨팅"
    ]
    sector_choice = st.selectbox("분야 선택", sector_options)
    
    if st.button("✨ 텐배거 종목 추천받기", use_container_width=True, type="primary"):
        active_key = get_active_gemini_key(api_key_input)
        if not active_key: 
            st.error("API 키를 입력해 주세요.")
        else:
            with st.spinner("전세계 최신 뉴스 및 산업 동향 실시간 종합 분석 중..."):
                try: 
                    if sector_choice == "🤖 AI 자율 분야 발굴 (최신 글로벌 뉴스 기반)":
                        prompt = (
                            f"현재 {date.today().year}년 최신 글로벌 뉴스, 주요 기술 트렌드 및 주식 시장 동향을 종합적으로 판단하세요.\n"
                            f"현재 전세계에서 가장 주목받고 있으며 10배(Tenbagger) 성장 모멘텀이 높은 '최우선 유망 산업 분야 1개'를 AI가 직접 선정하고,\n"
                            f"해당 분야에서 가장 성장이 기대되는 대표 유망 중소형 미국 주식 3개의 [종목코드, 선정이유, 핵심촉매]를 작성해 주세요."
                        )
                    else:
                        prompt = f"현재 시점 {date.today().year}년. '{sector_choice}' 분야 10배 성장 유망 중소형주 3개 요약."
                        
                    res_text = get_ai_text(active_key, model_name, prompt)
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state["ai_recommend_cache"][sector_choice] = {
                        "created_at": now_str,
                        "content": res_text
                    }
                    save_json_file(RECOMMEND_FILE, st.session_state["ai_recommend_cache"])
                except Exception as e: 
                    st.error(f"오류 발생: {e}")

    if sector_choice in st.session_state["ai_recommend_cache"]:
        item = st.session_state["ai_recommend_cache"][sector_choice]
        st.caption(f"📅 **분석 일시:** `{item['created_at']}`")
        st.markdown(item["content"])

# ========================================================
# TAB 4: 일지
# ========================================================
with tab4:
    def load_journal():
        if os.path.exists(JOURNAL_FILE):
            df = pd.read_csv(JOURNAL_FILE)
            if "ID" not in df.columns: df.insert(0, "ID", [uuid.uuid4().hex[:8] for _ in range(len(df))])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    def save_journal(df):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
        df.to_csv(JOURNAL_FILE, index=False)

    with st.expander("✍️ 기록 추가", expanded=False):
        with st.form("j_form"):
            j_date = st.date_input("날짜", date.today())
            j_action = st.selectbox("구분", ["매수", "매도", "관망"])
            j_price = st.number_input("가격 ($)", format="%.2f")
            j_reason = st.text_input("메모")
            if st.form_submit_button("추가", use_container_width=True):
                new_row = pd.DataFrame([[uuid.uuid4().hex[:8], pd.to_datetime(j_date), ticker, j_action, j_price, j_reason]], columns=JOURNAL_COLUMNS)
                save_journal(pd.concat([load_journal(), new_row], ignore_index=True))
                st.rerun()

    df_journal = load_journal()
    if not df_journal.empty:
        st.caption("💡 표 체크박스로 행을 선택한 후 우측 상단 🗑️ 아이콘을 누르면 삭제됩니다.")
        edited_df = st.data_editor(
            df_journal[df_journal["Ticker"] == ticker], 
            num_rows="dynamic", 
            hide_index=True, 
            key="j_editor",
            column_config={"ID": st.column_config.TextColumn(disabled=True)}
        )
        if st.button("💾 저장", use_container_width=True):
            other_rows = df_journal[df_journal["Ticker"] != ticker]
            save_journal(pd.concat([other_rows, edited_df], ignore_index=True))
            st.rerun()

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

# =========================================================
# [1] 페이지 설정 (반드시 가장 먼저 호출)
# =========================================================
st.set_page_config(page_title="AI 텐배거 프로", layout="centered", page_icon="📈")

# =========================================================
# [2] 앱 자체 테마 설정 파일 자동 생성
#     - 배포 환경에 따라 디스크가 read-only일 수 있으므로 예외 처리
#     - 무한 rerun을 막기 위해 세션 플래그로 1회만 시도
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
        pass  # 읽기 전용 배포 환경 등에서는 조용히 넘어가고 CSS로만 테마 적용

ensure_theme_config()

# =========================================================
# [3] 디자인 시스템: 색상 토큰 + 타이포그래피
#     - 라벨/본문: Inter (가독성 좋은 UI 서체)
#     - 숫자(가격·퍼센트·지표): JetBrains Mono → 데이터 값이라는 것을 시각적으로 구분
#     - 상승/하락은 초록/빨강으로 의미 고정, 인터랙션 요소(버튼)는 블루 계열로 분리
#       (기존 코드는 "주요 액션 버튼"과 "손실"에 같은 rose 컬러를 써서 의미가 충돌했음)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #0B1120;
    --surface: #141B2E;
    --surface-2: #1B2438;
    --border: #262F45;
    --text: #ECEFF4;
    --text-muted: #8A94AC;
    --accent: #5B9DF9;
    --accent-strong: #3B82F6;
    --up: #34D399;
    --down: #F87171;
    --radius: 10px;
}

html, body, .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
h1, h2, h3, h4, h5, h6, p, label, span, div { color: var(--text); }

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    font-size: clamp(1.3rem, 5vw, 1.9rem) !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 0.1rem !important;
}
h1 + div { color: var(--text-muted) !important; font-size: 0.9rem; }

/* 히어로 가격 블록: 유일하게 "굵게" 강조하는 요소 (강조는 한 곳에만) */
.hero-price {
    padding: 6px 0 14px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
}
.hero-label { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 2px; }
.hero-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(2rem, 8vw, 2.6rem);
    font-weight: 700;
    line-height: 1.1;
    color: var(--text);
}
.hero-delta { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; margin-top: 4px; }
.badge {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; padding: 3px 9px; border-radius: 6px;
    border: 1px solid var(--border); margin-top: 8px; margin-right: 6px;
}

/* 보조 지표 카드는 히어로보다 낮은 위계 - 얇은 테두리만 */
[data-testid="stMetric"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 12px 14px !important;
}
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }

[data-testid="stExpander"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600 !important; }

.stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="select"] > div { background-color: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-baseweb="menu"], [data-baseweb="popover"] { background-color: var(--surface) !important; border: 1px solid var(--border) !important; }
[data-baseweb="menu"] li:hover { background-color: var(--surface-2) !important; }

/* 인터랙션(버튼)은 항상 블루 계열 → "위험/손실"과 색이 겹치지 않게 분리 */
.stButton>button {
    background: var(--accent-strong) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}
button[kind="primary"] { background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important; }

.stTabs [data-baseweb="tab-list"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; border-radius: 6px !important; }
.stTabs [aria-selected="true"] { background-color: var(--surface-2) !important; color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# [4] 세션 상태 초기화
# =========================================================
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = ["ASTS", "OKLO", "IONQ", "RXRX", "PLTR", "TSLA"]
if "current_ticker" not in st.session_state:
    st.session_state["current_ticker"] = "ASTS"

JOURNAL_FILE = "trading_journal.csv"
JOURNAL_COLUMNS = ["ID", "Date", "Ticker", "Action", "Price", "Reason"]

# =========================================================
# [5] 데이터 로딩 (캐시 + 예외 처리 + 최신 yfinance 스키마 대응)
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(t: str) -> pd.DataFrame:
    data = yf.download(t, start="2018-01-01", end=date.today().strftime("%Y-%m-%d"),
                        progress=False, auto_adjust=True)
    if data.empty:
        return data
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

@st.cache_data(ttl=1800, show_spinner=False)
def is_valid_ticker(t: str) -> bool:
    """관심 종목 추가 전 실제 존재하는 티커인지 가볍게 검증"""
    try:
        hist = yf.Ticker(t).history(period="5d")
        return not hist.empty
    except Exception:
        return False

@st.cache_data(ttl=3600, show_spinner=False)
def load_news(t: str) -> list:
    try:
        return yf.Ticker(t).news or []
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def load_institutional_holders(t: str):
    try:
        return yf.Ticker(t).institutional_holders
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def run_forecast(df_train: pd.DataFrame, years: int) -> pd.DataFrame:
    m = Prophet(daily_seasonality=False)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=years * 365)
    return m.predict(future)

@st.cache_data(ttl=21600, show_spinner=False)
def cached_ai_text(api_key: str, model_name: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


def get_active_gemini_key(sidebar_key: str) -> str:
    return sidebar_key or os.environ.get("GEMINI_API_KEY", "")


# =========================================================
# [6] 사이드바: 설정 및 종목 관리
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ 시스템 설정")
    api_key_input = st.text_input("Gemini API Key", type="password")
    model_name = st.text_input(
        "Gemini 모델명", value="gemini-flash-latest",
        help="Google이 모델명을 자주 교체하므로, 오류가 나면 최신 모델명(예: gemini-2.5-flash)으로 바꿔보세요."
    )

    st.divider()
    st.markdown("### ➕ 관심 종목 관리")
    new_ticker = st.text_input("새 종목 코드 추가", placeholder="예: NVDA")
    if st.button("종목 추가", use_container_width=True):
        t = new_ticker.upper().strip()
        if not t:
            st.warning("종목 코드를 입력해 주세요.")
        elif t in st.session_state["watchlist"]:
            st.info(f"{t}는 이미 관심 종목에 있습니다.")
        elif not is_valid_ticker(t):
            st.error(f"'{t}'는 유효한 티커가 아닌 것 같습니다. 코드를 확인해 주세요.")
        else:
            st.session_state["watchlist"].append(t)
            st.session_state["current_ticker"] = t
            st.rerun()

    del_ticker = st.selectbox("삭제할 종목 선택", st.session_state["watchlist"])
    can_delete = len(st.session_state["watchlist"]) > 1
    if st.button("종목 삭제", use_container_width=True, disabled=not can_delete):
        st.session_state["watchlist"].remove(del_ticker)
        if st.session_state["current_ticker"] == del_ticker:
            st.session_state["current_ticker"] = st.session_state["watchlist"][0]
        st.rerun()
    if not can_delete:
        st.caption("⚠️ 최소 1개 종목은 유지되어야 합니다.")

# =========================================================
# [7] 메인 화면 타이틀 및 종목 선택
# =========================================================
st.title("📈 AI 텐배거 발굴기 Pro")
st.caption("Professional AI-Driven Stock & Retirement Intelligence")

selected_index = (
    st.session_state["watchlist"].index(st.session_state["current_ticker"])
    if st.session_state["current_ticker"] in st.session_state["watchlist"] else 0
)
ticker = st.selectbox("🔍 분석 대상 종목", st.session_state["watchlist"], index=selected_index)
st.session_state["current_ticker"] = ticker

with st.spinner(f"{ticker} 데이터 불러오는 중..."):
    data = load_price_data(ticker)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 차트", "🧠 리포트", "🎯 목표", "🌟 추천", "📝 일지"])

# ========================================================
# TAB 1: 📈 차트 & 기술적 분석
# ========================================================
with tab1:
    if data.empty:
        st.error(f"'{ticker}'의 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")
    else:
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA200"] = data["Close"].rolling(200).mean()

        roll_max = data["Close"].cummax()
        max_drawdown = (data["Close"] / roll_max - 1.0).min() * 100
        current_price = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
        delta = current_price - prev_close
        delta_pct = (delta / prev_close * 100) if prev_close else 0.0
        delta_color = "var(--up)" if delta >= 0 else "var(--down)"
        arrow = "▲" if delta >= 0 else "▼"

        # 정배열/역배열(골든/데드 크로스 상태) 판정
        cross_badge = ""
        if pd.notna(data["MA50"].iloc[-1]) and pd.notna(data["MA200"].iloc[-1]):
            ma50, ma200 = data["MA50"].iloc[-1], data["MA200"].iloc[-1]
            if len(data) > 1 and pd.notna(data["MA50"].iloc[-2]) and pd.notna(data["MA200"].iloc[-2]):
                prev_ma50, prev_ma200 = data["MA50"].iloc[-2], data["MA200"].iloc[-2]
                just_crossed_up = prev_ma50 <= prev_ma200 and ma50 > ma200
                just_crossed_down = prev_ma50 >= prev_ma200 and ma50 < ma200
            else:
                just_crossed_up = just_crossed_down = False

            if just_crossed_up:
                cross_badge = '<span class="badge" style="color:var(--up); border-color:var(--up);">⚡ 골든크로스 발생</span>'
            elif just_crossed_down:
                cross_badge = '<span class="badge" style="color:var(--down); border-color:var(--down);">⚡ 데드크로스 발생</span>'
            elif ma50 > ma200:
                cross_badge = '<span class="badge" style="color:var(--up); border-color:var(--up);">정배열 (MA50 &gt; MA200)</span>'
            else:
                cross_badge = '<span class="badge" style="color:var(--down); border-color:var(--down);">역배열 (MA50 &lt; MA200)</span>'

        st.markdown(f"""
        <div class="hero-price">
            <div class="hero-label">{ticker} 현재가</div>
            <div class="hero-value">${current_price:,.2f}</div>
            <div class="hero-delta" style="color:{delta_color};">{arrow} {abs(delta):,.2f} ({abs(delta_pct):.2f}%)</div>
            {cross_badge}
        </div>
        """, unsafe_allow_html=True)

        rsi_val = data["RSI"].iloc[-1]
        rsi_state = "과매수" if rsi_val >= 70 else ("과매도" if rsi_val <= 30 else "중립")
        c1, c2, c3 = st.columns(3)
        c1.metric("RSI (14)", f"{rsi_val:.0f}", rsi_state)
        c2.metric("최대 낙폭 (MDD)", f"{max_drawdown:.1f}%")
        c3.metric("52주 최고가", f"${data['Close'].tail(252).max():,.2f}")

        st.write("")
        years = st.slider("미래 예측 기간 (년)", 1, 5, 2)
        df_train = data[["Date", "Close"]].copy().rename(columns={"Date": "ds", "Close": "y"})

        with st.spinner("AI 예측 모델 학습 중..."):
            forecast = run_forecast(df_train, years)

        fig_chart = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
            vertical_spacing=0.04
        )
        fig_chart.add_trace(go.Scatter(x=df_train["ds"], y=df_train["y"], mode="markers",
                                        name="실제 주가", marker=dict(color="#64748B", size=3)), row=1, col=1)
        fig_chart.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], mode="lines",
                                        name="MA50", line=dict(color="#5B9DF9", width=1.4)), row=1, col=1)
        fig_chart.add_trace(go.Scatter(x=data["Date"], y=data["MA200"], mode="lines",
                                        name="MA200", line=dict(color="#F59E0B", width=1.4)), row=1, col=1)
        fig_chart.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines",
                                        name="AI 예측선", line=dict(color="#F43F5E", width=2)), row=1, col=1)
        fig_chart.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
                                        line=dict(width=0), showlegend=False), row=1, col=1)
        fig_chart.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
                                        line=dict(width=0), fillcolor="rgba(244,63,94,0.15)",
                                        fill="tonexty", name="신뢰구간"), row=1, col=1)

        fig_chart.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], mode="lines",
                                        name="RSI", line=dict(color="#A78BFA", width=1.4), showlegend=False), row=2, col=1)
        fig_chart.add_hline(y=70, line_dash="dot", line_color="#F87171", opacity=0.6, row=2, col=1)
        fig_chart.add_hline(y=30, line_dash="dot", line_color="#34D399", opacity=0.6, row=2, col=1)

        fig_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECEFF4"), margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.12, yanchor="top", font=dict(size=11, color="#ECEFF4")),
            height=520,
        )
        fig_chart.update_xaxes(showgrid=True, gridcolor="#1E293B", tickfont=dict(color="#ECEFF4"))
        fig_chart.update_yaxes(showgrid=True, gridcolor="#1E293B", tickfont=dict(color="#ECEFF4"), row=1, col=1)
        fig_chart.update_yaxes(showgrid=True, gridcolor="#1E293B", tickfont=dict(color="#ECEFF4"), range=[0, 100], row=2, col=1)
        st.plotly_chart(fig_chart, use_container_width=True)

# ========================================================
# TAB 2: 🧠 AI 촉매제 리포트
# ========================================================
with tab2:
    st.subheader("실시간 뉴스 AI 분석")
    active_key = get_active_gemini_key(api_key_input)

    if st.button("🔥 AI 심층 리포트 생성", use_container_width=True, type="primary"):
        if not active_key:
            st.error("사이드바에 Gemini API 키를 입력해 주세요.")
        else:
            with st.spinner("실시간 뉴스 필터링 및 AI 분석 중..."):
                recent_news = load_news(ticker)[:10]
                current_year = date.today().year

                news_items = []
                for item in recent_news:
                    content = item.get("content", {}) if isinstance(item, dict) else {}
                    title = content.get("title") or item.get("title", "제목 없음")
                    pub_date = content.get("pubDate")
                    if pub_date:
                        try:
                            date_str = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).strftime("%Y-%m-%d")
                            news_items.append(f"- [{date_str}] {title}")
                        except ValueError:
                            news_items.append(f"- {title}")
                    else:
                        news_items.append(f"- {title}")

                news_text = "\n".join(news_items) if news_items else "발행일이 확인된 최신 뉴스가 없습니다."

                prompt = f"""
현재 시스템 기준 연도는 {current_year}년입니다. 아래는 종목 '{ticker}'의 공식 발행일이 포함된 최신 뉴스 목록입니다:
{news_text}

위 뉴스 목록의 [날짜]를 엄격히 참고하여, 가장 최근에 보도된 실시간 뉴스를 기준으로 {current_year}년 현재 상황에 맞는 단기 급등 촉매제와 장기 리스크를 전문 애널리스트 톤으로 분석하고 요약해 줘.
                """
                try:
                    st.markdown(cached_ai_text(active_key, model_name, prompt))
                except Exception as e:
                    st.error(f"오류 발생: {e}")
                    st.caption("모델명이 잘못되었거나 API 키가 유효하지 않을 수 있습니다. 사이드바의 모델명을 확인해 보세요.")

    st.write("")
    with st.expander("💼 기관 투자자 보유 현황 (Smart Money)"):
        holders = load_institutional_holders(ticker)
        if holders is not None and not holders.empty:
            show_cols = [c for c in ["Holder", "Shares", "Date Reported", "% Out"] if c in holders.columns]
            st.dataframe(holders[show_cols] if show_cols else holders, use_container_width=True)
        else:
            st.info("기관 보유량 데이터를 찾을 수 없습니다.")

# ========================================================
# TAB 3: 🎯 목표 & 시뮬레이터
# ========================================================
with tab3:
    st.subheader("10년 복리 시뮬레이터")
    with st.expander("⚙️ 은퇴 및 목표 자산 설정"):
        target_farm = st.number_input("스마트팜 구축 ($)", value=300000, min_value=0)
        target_golf = st.number_input("정기 골프 펀드 ($)", value=100000, min_value=0)
        target_living = st.number_input("생활 자금 ($)", value=600000, min_value=0)
        current_asset = st.number_input("현재 투자 원금 ($)", value=10000, min_value=0)
        annual_return_pct = st.slider("가정 연평균 수익률 (%)", 5, 100, 30,
                                       help="1000배 고정 대신 직접 목표 수익률을 조정할 수 있습니다.")

    total_target = target_farm + target_golf + target_living
    progress = min((current_asset / total_target) * 100, 100.0) if total_target > 0 else 0

    st.markdown(f"**총 목표액:** `${total_target:,.0f}` &nbsp;|&nbsp; **달성률:** `{progress:.1f}%`")
    st.progress(progress / 100)

    years_sim = np.arange(0, 11)
    growth_factor = 1 + annual_return_pct / 100
    target_vals = current_asset * (growth_factor ** years_sim)
    fig_sim = go.Figure(go.Scatter(x=years_sim, y=target_vals, mode="lines+markers",
                                    line=dict(color="#5B9DF9", width=3)))
    fig_sim.add_hline(y=total_target, line_dash="dot", line_color="#F59E0B",
                       annotation_text="목표 자산", annotation_position="top left")
    fig_sim.update_layout(
        title=f"연 {annual_return_pct}% 복리 성장 궤적 (10년)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECEFF4"), margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor="#1E293B", title="경과 년수", tickfont=dict(color="#ECEFF4")),
        yaxis=dict(showgrid=True, gridcolor="#1E293B", title="예상 자산 ($)", tickfont=dict(color="#ECEFF4")),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ========================================================
# TAB 4: 🌟 AI 텐배거 추천
# ========================================================
with tab4:
    st.subheader("혁신 섹터 유망주 발굴")
    sector_choice = st.selectbox("분야 선택", ["우주 항공 및 통신", "AI 바이오 헬스케어", "차세대 에너지 (SMR)", "양자 컴퓨팅"])
    active_key = get_active_gemini_key(api_key_input)

    if st.button("✨ 텐배거 후보 추천받기", use_container_width=True):
        if not active_key:
            st.error("사이드바에 API 키를 입력하세요.")
        else:
            with st.spinner("유망 기업 발굴 분석 중..."):
                current_year = date.today().year
                prompt = (
                    f"현재 시점은 {current_year}년입니다. '{sector_choice}' 분야에서 10배 이상(Tenbagger) "
                    "성장할 잠재력 있는 미국 중소형 혁신 기업 3곳을 선정하고 핵심 투자 포인트를 요약해 줘."
                )
                try:
                    st.markdown(cached_ai_text(active_key, model_name, prompt))
                except Exception as e:
                    st.error(f"오류: {e}")

# ========================================================
# TAB 5: 📝 투자 일지
# ========================================================
with tab5:
    st.subheader("매매 복기 및 투자 일지")

    def load_journal() -> pd.DataFrame:
        if os.path.exists(JOURNAL_FILE):
            df = pd.read_csv(JOURNAL_FILE)
            if "ID" not in df.columns:
                df.insert(0, "ID", [uuid.uuid4().hex[:8] for _ in range(len(df))])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    def save_journal(df: pd.DataFrame):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
        df.to_csv(JOURNAL_FILE, index=False)

    with st.expander("✍️ 새 일지 작성하기"):
        with st.form("journal_form"):
            fc1, fc2 = st.columns(2)
            j_date = fc1.date_input("날짜", date.today())
            j_action = fc2.selectbox("구분", ["매수", "매도", "관망"])
            j_price = st.number_input("가격 ($)", min_value=0.0, format="%.2f")
            j_reason = st.text_area("결정 논리 및 전략 메모")

            if st.form_submit_button("신규 일지 추가", use_container_width=True):
                new_row = pd.DataFrame([[uuid.uuid4().hex[:8], pd.to_datetime(j_date), ticker, j_action, j_price, j_reason]],
                                        columns=JOURNAL_COLUMNS)
                save_journal(pd.concat([load_journal(), new_row], ignore_index=True))
                st.success("새로운 일지가 추가되었습니다!")
                st.rerun()

    df_journal = load_journal()

    if not df_journal.empty:
        current_only = st.checkbox(f"'{ticker}' 종목만 보기", value=False)
        view_df = df_journal[df_journal["Ticker"] == ticker] if current_only else df_journal

        with st.expander("⚠️ 일지 전체 삭제"):
            scope_desc = f"'{ticker}' 종목 기록만" if current_only else "전체 종목 기록 전부"
            st.caption(f"지금 체크박스 기준으로 **{scope_desc}** ({len(view_df)}건)이 삭제됩니다. 되돌릴 수 없어요.")
            confirm_clear = st.checkbox("네, 정말 전체 삭제하겠습니다", key="confirm_clear_journal")
            if st.button("전체 삭제 실행", type="primary", disabled=not confirm_clear, use_container_width=True):
                remaining_df = df_journal[df_journal["Ticker"] != ticker] if current_only else pd.DataFrame(columns=JOURNAL_COLUMNS)
                save_journal(remaining_df)
                st.success("전체 삭제가 완료되었습니다!")
                st.rerun()

        buys = view_df[view_df["Action"] == "매수"]
        sells = view_df[view_df["Action"] == "매도"]
        m1, m2, m3 = st.columns(3)
        m1.metric("매수 기록", len(buys))
        m2.metric("매도 기록", len(sells))
        m3.metric("평균 매수가", f"${buys['Price'].mean():,.2f}" if len(buys) else "—")

        # 티커 선택지에는 관심 종목 + 일지에 이미 남아있는 과거 종목을 모두 포함
        ticker_options = sorted(set(st.session_state["watchlist"]) | set(df_journal["Ticker"].dropna().unique().tolist()))

        st.write("")
        st.markdown("### 📋 일지 기록 (내용 수정)")
        st.caption("💡 날짜는 달력에서, 티커·구분은 목록에서 선택해 수정할 수 있어요. 메모(Reason)만 자유 입력입니다.")

        edited_df = st.data_editor(
            view_df,
            num_rows="fixed",  # 행 추가/삭제는 아래 전용 UI로 처리 (표 안 삭제는 잘 안 먹는 경우가 많아 분리함)
            use_container_width=True,
            hide_index=True,
            key="journal_editor",
            column_config={
                "ID": st.column_config.TextColumn("ID", disabled=True),
                "Date": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                "Ticker": st.column_config.SelectboxColumn("티커", options=ticker_options),
                "Action": st.column_config.SelectboxColumn("구분", options=["매수", "매도", "관망"]),
                "Price": st.column_config.NumberColumn("가격 ($)", format="$%.2f", min_value=0.0),
                "Reason": st.column_config.TextColumn("메모"),
            },
        )

        if st.button("💾 변경된 일지 저장하기", use_container_width=True):
            if current_only:
                # 필터링된 뷰만 편집했을 경우, 다른 종목 기록은 그대로 두고 병합
                other_rows = df_journal[df_journal["Ticker"] != ticker]
                final_df = pd.concat([other_rows, edited_df], ignore_index=True)
            else:
                final_df = edited_df
            save_journal(final_df)
            st.success("일지 변경사항이 저장되었습니다!")
            st.rerun()

        st.write("")
        st.markdown("### 🗑️ 일지 삭제")
        st.caption("항목 옆 '삭제' 버튼을 누르면 확인 없이 바로 지워집니다.")
        view_df_sorted = view_df.sort_values("Date", ascending=False, na_position="last")
        for _, row in view_df_sorted.iterrows():
            row_id = row["ID"]
            date_str = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "날짜없음"
            reason_preview = (str(row["Reason"])[:40] + "…") if isinstance(row["Reason"], str) and len(str(row["Reason"])) > 40 else row["Reason"]
            rcol1, rcol2 = st.columns([5, 1])
            rcol1.markdown(
                f"**{date_str}** · {row['Ticker']} · {row['Action']} · ${row['Price']:,.2f}"
                + (f" — {reason_preview}" if isinstance(reason_preview, str) and reason_preview else "")
            )
            if rcol2.button("삭제", key=f"del_{row_id}", use_container_width=True):
                remaining_df = df_journal[df_journal["ID"] != row_id]
                save_journal(remaining_df)
                st.success("삭제했습니다.")
                st.rerun()
    else:
        st.info("아직 작성된 일지가 없습니다. 위에서 첫 기록을 추가해 보세요.")

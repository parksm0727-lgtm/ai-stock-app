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
# [3] 디자인 시스템 (탭 5개 완벽 가로 균등 분할 CSS)
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
    --up: #34D399;
    --down: #F87171;
    --radius: 8px;
}

html, body, .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Noto Sans KR', -apple-system, sans-serif !important;
}
h2, h3, h4, h5, h6, p, label, span, div { color: var(--text); }

/* 상단바 가림 방지 여백 */
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
    max-width: 720px !important;
}
[data-testid="stVerticalBlock"] { gap: 0.1rem !important; }
[data-testid="stElementContainer"] { margin-bottom: 0 !important; }

/* 폭 방향으로 꽉 차는 타이틀 배너 */
.title-banner {
    background: linear-gradient(135deg, #1E293B 0%, #1D4ED8 50%, #3B82F6 100%);
    color: #ffffff;
    text-align: center;
    font-weight: 800;
    font-size: clamp(1.3rem, 5.5vw, 1.7rem);
    padding: 10px 12px !important;
    border-radius: 12px;
    margin-bottom: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    letter-spacing: -0.5px;
    width: 100%;
    border: 1.5px solid #38BDF8;
    line-height: 1.3 !important;
    box-sizing: border-box;
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

/* 탭 5개 가로 1:1:1:1:1 강제 분할 CSS */
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
    font-size: 0.78rem !important;
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

def get_ai_text(api_key: str, model_name: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    target_models = ["gemini-3.6-flash", model_name, "gemini-2.5-flash"]
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

def render_mini_grid(cards: list):
    parts = []
    for c in cards:
        tag_html = f'<div class="mini-tag" style="color:{c.get("tag_color", "var(--text-muted)")};">{c["tag"]}</div>' if c.get("tag") else ""
        parts.append(f'<div class="mini-card"><div class="mini-label">{c["label"]}</div><div class="mini-value">{c["value"]}</div>{tag_html}</div>')
    st.markdown(f'<div class="mini-grid">{"".join(parts)}</div>', unsafe_allow_html=True)

# 💡 셀렉트박스 변경 시 한 번에 바로 즉시 동기화시키는 콜백 함수
def on_ticker_change():
    st.session_state["current_ticker"] = st.session_state["ticker_select_box"]

# =========================================================
# [6] 사이드바
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ 시스템 설정")
    api_key_input = st.text_input("Gemini API Key", type="password")
    model_name = st.text_input("Gemini 모델명", value="gemini-3.6-flash")

# =========================================================
# [7] 메인 타이틀 & 종목 선택
# =========================================================
st.markdown('<div class="title-banner">📈 AI 텐배거 발굴기 Pro</div>', unsafe_allow_html=True)

selected_index = st.session_state["watchlist"].index(st.session_state["current_ticker"]) if st.session_state["current_ticker"] in st.session_state["watchlist"] else 0

# 💡 key 및 on_change 적용으로 2번 클릭 현상 완벽 해결
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 차트", "🧠 리포트", "🎯 목표", "🌟 추천", "📝 일지"])

# ========================================================
# TAB 1: 차트
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
        
        st.markdown(
            f'<div class="hero-price"><div class="hero-label">{ticker} 최신가</div>'
            f'<div class="hero-value">${current_price:,.2f}</div>'
            f'<div class="hero-delta" style="color:{"var(--up)" if delta >= 0 else "var(--down)"};">{"▲" if delta >= 0 else "▼"} {abs(delta):,.2f} ({abs(delta_pct):.2f}%)</div></div>',
            unsafe_allow_html=True,
        )

        rsi_val = data["RSI"].iloc[-1]
        rsi_state = "과매수" if rsi_val >= 70 else ("과매도" if rsi_val <= 30 else "중립")
        render_mini_grid([
            {"label": "RSI(14)", "value": f"{rsi_val:.0f}", "tag": rsi_state, "tag_color": "var(--down)" if rsi_state == "과매수" else ("var(--up)" if rsi_state == "과매도" else "var(--text-muted)")},
            {"label": "MDD", "value": f"{(data['Close'] / data['Close'].cummax() - 1.0).min() * 100:.1f}%"},
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
# TAB 3: 목표
# ========================================================
with tab3:
    with st.expander("⚙️ 설정", expanded=True):
        c1, c2 = st.columns(2)
        target_farm = c1.number_input("스마트팜($)", value=300000)
        target_golf = c2.number_input("골프펀드($)", value=100000)
        target_living = c1.number_input("생활자금($)", value=600000)
        current_asset = c2.number_input("현재투자($)", value=10000)
        annual_return_pct = st.slider("연평균 수익률 (%)", 5, 100, 30)

    total_target = target_farm + target_golf + target_living
    progress = min((current_asset / total_target) * 100, 100.0) if total_target > 0 else 0
    st.markdown(f"**목표액:** `${total_target:,.0f}` &nbsp;|&nbsp; **달성:** `{progress:.1f}%`")
    st.progress(progress / 100)

    years_sim = np.arange(0, 11)
    target_vals = current_asset * ((1 + annual_return_pct / 100) ** years_sim)
    fig_sim = go.Figure(go.Scatter(x=years_sim, y=target_vals, mode="lines+markers", line=dict(color="#5B9DF9", width=2)))
    fig_sim.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ECEFF4"))
    st.plotly_chart(fig_sim, use_container_width=True)

# ========================================================
# TAB 4: 추천
# ========================================================
with tab4:
    sector_choice = st.selectbox("분야 선택", ["우주 항공 및 통신", "AI 바이오 헬스케어", "차세대 에너지 (SMR)", "양자 컴퓨팅"])
    if st.button("✨ 추천받기", use_container_width=True):
        active_key = get_active_gemini_key(api_key_input)
        if not active_key: 
            st.error("API 키 필요")
        else:
            with st.spinner("분석 중..."):
                try: 
                    res_text = get_ai_text(active_key, model_name, f"현재 시점 {date.today().year}년. '{sector_choice}' 분야 10배 성장 유망 중소형주 3개 요약.")
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
        st.caption(f"📅 **생성 일시:** `{item['created_at']}`")
        st.markdown(item["content"])

# ========================================================
# TAB 5: 일지
# ========================================================
with tab5:
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

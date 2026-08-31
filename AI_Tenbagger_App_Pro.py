import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
import google.generativeai as genai
from ta.momentum import RSIIndicator
from datetime import date
import pandas as pd
import numpy as np
import os
import datetime

st.markdown("""
    <style>
    /* 1. 전체 배경 (첨부 이미지 스타일의 깊은 네이비) */
    .stApp, .main, [data-testid="stSidebar"] {
        background-color: #0B1121 !important;
        color: #F8FAFC !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #F8FAFC !important;
    }
    
    /* 2. 타이틀 그라데이션 (네온 블루) */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38BDF8 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* 3. 카드형 컨테이너 */
    [data-testid="stMetric"] {
        background-color: #172033 !important;
        border: 1px solid #2D3B55 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] label {
        color: #94A3B8 !important;
    }

    /* 4. 아코디언 (Expander) */
    [data-testid="stExpander"] {
        background-color: #172033 !important;
        border: 1px solid #2D3B55 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #172033 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* 5. 기본 입력창 및 드롭다운 */
    .stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea {
        background-color: #0B1121 !important;
        color: #FFFFFF !important;
        border: 1px solid #2D3B55 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] > div {
        background-color: #0B1121 !important;
        color: #FFFFFF !important;
        border: 1px solid #2D3B55 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] span {
        color: #FFFFFF !important;
    }
    [data-baseweb="menu"] {
        background-color: #172033 !important;
        border: 1px solid #2D3B55 !important;
    }
    [data-baseweb="menu"] li {
        color: #F8FAFC !important;
        background-color: transparent !important;
    }

    /* 💡 [핵심 해결] 6. 표(Data Editor) 내부 팝업창 및 검색창 완벽 제어 */
    [data-testid="stDataFrame"] {
        background-color: #172033 !important;
    }
    /* 팝업창(Dialog/Popover) 배경 네이비 지정 */
    div[role="dialog"], div[data-baseweb="popover"], .gdg-search-container {
        background-color: #172033 !important;
        border: 1px solid #2D3B55 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
    /* 팝업창 내부 입력칸 글자색 강제 지정 */
    div[role="dialog"] input, div[data-baseweb="popover"] input, .gdg-search-input {
        background-color: #0B1121 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important; /* 사파리/모바일 크롬 대응 */
        border: 1px solid #38BDF8 !important;
    }
    div[role="dialog"] span, div[role="dialog"] p {
        color: #94A3B8 !important;
    }

    /* 7. 버튼 스타일 */
    .stButton>button {
        background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%) !important;
    }

    /* 8. 탭 (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #172033 !important;
        border: 1px solid #2D3B55 !important;
        border-radius: 10px !important;
        padding: 5px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2D3B55 !important;
        color: #38BDF8 !important;
    }
    </style>
""", unsafe_allow_html=True)
# --- [2] 세션 상태 초기화 ---
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = ['ASTS', 'OKLO', 'IONQ', 'RXRX', 'PLTR', 'TSLA']
if 'current_ticker' not in st.session_state:
    st.session_state['current_ticker'] = 'ASTS'

# --- [3] 사이드바: 설정 및 종목 관리 ---
with st.sidebar:
    st.markdown("### ⚙️ 시스템 설정")
    api_key_input = st.text_input("Gemini API Key", type="password")
    
    if api_key_input:
        genai.configure(api_key=api_key_input)
    elif "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    st.divider()
    st.markdown("### ➕ 관심 종목 관리")
    new_ticker = st.text_input("새 종목 코드 추가", placeholder="예: NVDA")
    if st.button("종목 추가", use_container_width=True):
        t = new_ticker.upper().strip()
        if t and t not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(t)
            st.session_state['current_ticker'] = t
            st.rerun()
            
    del_ticker = st.selectbox("삭제할 종목 선택", st.session_state['watchlist'])
    if st.button("종목 삭제", use_container_width=True):
        st.session_state['watchlist'].remove(del_ticker)
        if st.session_state['current_ticker'] == del_ticker and st.session_state['watchlist']:
            st.session_state['current_ticker'] = st.session_state['watchlist'][0]
        st.rerun()

# --- [4] 메인 화면 타이틀 및 종목 선택 ---
st.title("📈 AI 텐배거 프로")
st.caption("Professional AI-Driven Stock & Retirement Intelligence")
st.write("")

selected_index = st.session_state['watchlist'].index(st.session_state['current_ticker']) if st.session_state['current_ticker'] in st.session_state['watchlist'] else 0
ticker = st.selectbox("🔍 분석 대상 종목", st.session_state['watchlist'], index=selected_index)
st.session_state['current_ticker'] = ticker

@st.cache_data
def load_data(t):
    data = yf.download(t, "2018-01-01", date.today().strftime("%Y-%m-%d"))
    if not data.empty:
        data.reset_index(inplace=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
    return data

data = load_data(ticker)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 차트", "🧠 리포트", "🎯 목표", "🌟 추천", "📝 일지"])

# ========================================================
# TAB 1: 📈 차트 & 기술적 분석 (색상 리뉴얼)
# ========================================================
with tab1:
    if data.empty:
        st.error("데이터를 불러올 수 없습니다.")
    else:
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        roll_max = data['Close'].cummax()
        max_drawdown = (data['Close'] / roll_max - 1.0).min() * 100
        current_price = data['Close'].iloc[-1]
        
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("현재가", f"${current_price:.2f}")
        c2.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.0f}")
        c3.metric("최대 낙폭 (MDD)", f"{max_drawdown:.1f}%")
        st.write("")
        
        years = st.slider("미래 예측 기간 (년)", 1, 5, 2)
        df_train = data[['Date', 'Close']].copy().rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet()
        m.fit(df_train)
        forecast = m.predict(m.make_future_dataframe(periods=years * 365))
        
        fig_chart = go.Figure()
        # 네이비 배경에 어울리는 핫핑크/네온블루 라인 적용
        fig_chart.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='markers', name='실제 주가', marker=dict(color='#64748B', size=3)))
        fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='AI 예측선', line=dict(color='#F43F5E', width=2)))
        fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fillcolor='rgba(244,63,94,0.15)', fill='tonexty', name='신뢰구간'))
        
        fig_chart.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC'),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=-0.25, yanchor="top", font=dict(size=11, color='#F8FAFC')),
            xaxis=dict(showgrid=True, gridcolor='#1E293B', tickfont=dict(color='#F8FAFC')),
            yaxis=dict(showgrid=True, gridcolor='#1E293B', tickfont=dict(color='#F8FAFC'))
        )
        st.plotly_chart(fig_chart, use_container_width=True)

# ========================================================
# TAB 2: 🧠 AI 촉매제 리포트
# ========================================================
with tab2:
    st.subheader("실시간 뉴스 AI 분석")
    st.write("")
    
    if st.button("🔥 AI 심층 리포트 생성", use_container_width=True, type="primary"):
        if not api_key_input and "GEMINI_API_KEY" not in os.environ:
            st.error("사이드바에 Gemini API 키를 입력해 주세요.")
        else:
            with st.spinner("실시간 뉴스 필터링 및 AI 분석 중..."):
                stock_info = yf.Ticker(ticker)
                recent_news = stock_info.news[:10] if stock_info.news else []
                current_year = datetime.date.today().year
                
                news_items = []
                for news in recent_news:
                    title = news.get('title', '제목 없음')
                    pub_time = news.get('providerPublishTime', None)
                    if pub_time:
                        date_str = datetime.datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
                        news_items.append(f"- [{date_str}] {title}")
                    else:
                        news_items.append(f"- {title}")
                
                news_text = "\n".join(news_items) if news_items else "발행일이 확인된 최신 뉴스가 없습니다."
                
                prompt = f"""
                현재 시스템 기준 연도는 {current_year}년입니다. 아래는 종목 '{ticker}'의 공식 발행일이 포함된 최신 뉴스 목록입니다:
                {news_text}
                
                위 뉴스 목록의 [날짜]를 엄격히 참고하여, 가장 최근에 보도된 실시간 뉴스를 기준으로 {current_year}년 현재 상황에 맞는 단기 급등 촉매제와 장기 리스크를 전문 애널리스트 톤으로 분석하고 요약해 줘.
                """
                
                try:
                    model = genai.GenerativeModel('gemini-3.6-flash')
                    st.markdown(model.generate_content(prompt).text)
                except Exception as e:
                    st.error(f"오류 발생: {e}")
                    
    st.write("")
    with st.expander("💼 기관 투자자 보유 현황 (Smart Money)"):
        try:
            holders = yf.Ticker(ticker).institutional_holders
            if holders is not None and not holders.empty:
                st.dataframe(holders[['Holder', 'Shares']], use_container_width=True)
            else:
                st.info("기관 보유량 데이터를 찾을 수 없습니다.")
        except:
            st.error("데이터를 가져오는 중 오류가 발생했습니다.")

# ========================================================
# TAB 3: 🎯 목표 & 시뮬레이터 (색상 리뉴얼)
# ========================================================
with tab3:
    st.subheader("10년 복리 시뮬레이터")
    st.write("")
    with st.expander("⚙️ 은퇴 및 목표 자산 설정"):
        target_farm = st.number_input("스마트팜 구축 ($)", value=300000)
        target_golf = st.number_input("정기 골프 펀드 ($)", value=100000)
        target_living = st.number_input("생활 자금 ($)", value=600000)
        current_asset = st.number_input("현재 투자 원금 ($)", value=10000)
        
    total_target = target_farm + target_golf + target_living
    progress = min((current_asset / total_target) * 100, 100.0) if total_target > 0 else 0
    
    st.write("")
    st.markdown(f"**총 목표액:** `${total_target:,.0f}` &nbsp;|&nbsp; **달성률:** `{progress:.1f}%`")
    st.progress(progress / 100)
    st.write("")
    
    years_sim = np.arange(0, 11)
    target_vals = current_asset * (1000 ** (years_sim/10))
    fig_sim = go.Figure(go.Scatter(x=years_sim, y=target_vals, mode='lines+markers', line=dict(color='#38BDF8', width=3)))
    fig_sim.update_layout(
        title="10년 1000배 성장 궤적",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC'),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.25, yanchor="top", font=dict(color='#F8FAFC')),
        xaxis=dict(showgrid=True, gridcolor='#1E293B', title="경과 년수", tickfont=dict(color='#F8FAFC')),
        yaxis=dict(showgrid=True, gridcolor='#1E293B', title="예상 자산 ($)", tickfont=dict(color='#F8FAFC'))
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ========================================================
# TAB 4: 🌟 AI 텐배거 추천
# ========================================================
with tab4:
    st.subheader("혁신 섹터 유망주 발굴")
    st.write("")
    sector_choice = st.selectbox("분야 선택", ["우주 항공 및 통신", "AI 바이오 헬스케어", "차세대 에너지 (SMR)", "양자 컴퓨팅"])
    
    if st.button("✨ 텐배거 후보 추천받기", use_container_width=True):
        if not api_key_input and "GEMINI_API_KEY" not in os.environ:
            st.error("사이드바에 API 키를 입력하세요.")
        else:
            with st.spinner("유망 기업 발굴 분석 중..."):
                current_year = datetime.date.today().year
                prompt = f"현재 시점은 {current_year}년입니다. '{sector_choice}' 분야에서 10배 이상(Tenbagger) 성장할 잠재력 있는 미국 중소형 혁신 기업 3곳을 선정하고 핵심 투자 포인트를 요약해 줘."
                try:
                    model = genai.GenerativeModel('gemini-3.6-flash')
                    st.markdown(model.generate_content(prompt).text)
                except Exception as e:
                    st.error(f"오류: {e}")

# ========================================================
# TAB 5: 📝 투자 일지 (수정 및 삭제 기능 완벽 구현)
# ========================================================
with tab5:
    st.subheader("매매 복기 및 투자 일지")
    journal_file = "trading_journal.csv"
    
    with st.expander("✍️ 새 일지 작성하기"):
        with st.form("journal_form"):
            c1, c2 = st.columns(2)
            j_date = c1.date_input("날짜", date.today())
            j_action = c2.selectbox("구분", ["매수", "매도", "관망"])
            j_price = st.number_input("가격 ($)", min_value=0.0, format="%.2f")
            j_reason = st.text_area("결정 논리 및 전략 메모")
            
            if st.form_submit_button("신규 일지 추가", use_container_width=True):
                new_data = pd.DataFrame([[j_date, ticker, j_action, j_price, j_reason]], columns=["Date", "Ticker", "Action", "Price", "Reason"])
                if os.path.exists(journal_file):
                    df_journal = pd.concat([pd.read_csv(journal_file), new_data], ignore_index=True)
                else:
                    df_journal = new_data
                df_journal.to_csv(journal_file, index=False)
                st.success("새로운 일지가 추가되었습니다!")
                st.rerun() # 추가 후 화면 새로고침

    if os.path.exists(journal_file):
        st.write("")
        st.markdown("### 📋 일지 기록 (수정/삭제 가능)")
        st.caption("💡 표 안의 글자를 더블클릭하여 바로 수정하거나, 좌측 체크박스를 선택해 삭제(Delete)할 수 있습니다.")
        
        df_journal = pd.read_csv(journal_file)
        
        # st.data_editor를 사용하여 엑셀처럼 실시간 수정/삭제 지원
        edited_df = st.data_editor(
            df_journal, 
            num_rows="dynamic", # 이 옵션이 행 삭제/추가를 가능하게 합니다
            use_container_width=True,
            key="journal_editor"
        )
        
        # 수정된 표를 다시 파일에 저장하는 버튼
        if st.button("💾 변경된 일지 저장하기", use_container_width=True):
            edited_df.to_csv(journal_file, index=False)
            st.success("일지 변경사항이 완벽하게 저장되었습니다!")

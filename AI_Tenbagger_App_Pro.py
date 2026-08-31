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

# --- [1] 모바일 최적화 기본 설정 ---
st.set_page_config(page_title="AI 텐배거 발굴기", layout="centered", page_icon="📱")
st.title("📱 AI 주식 분석기 Pro")

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = ['ASTS', 'OKLO', 'IONQ', 'RXRX', 'PLTR', 'TSLA']
if 'current_ticker' not in st.session_state:
    st.session_state['current_ticker'] = 'ASTS'

# --- [2] 사이드바: 설정 및 종목 관리 (최소화) ---
with st.sidebar:
    st.header("⚙️ 환경 설정")
    api_key_input = st.text_input("Gemini API Key", type="password")
    
    if api_key_input:
        genai.configure(api_key=api_key_input)
    elif "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    st.divider()
    st.header("➕ 관심 종목 관리")
    new_ticker = st.text_input("새 종목 코드 추가", placeholder="예: NVDA")
    if st.button("추가하기", use_container_width=True):
        t = new_ticker.upper().strip()
        if t and t not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(t)
            st.session_state['current_ticker'] = t
            st.rerun()
            
    del_ticker = st.selectbox("삭제할 종목 선택", st.session_state['watchlist'])
    if st.button("삭제하기", use_container_width=True):
        st.session_state['watchlist'].remove(del_ticker)
        if st.session_state['current_ticker'] == del_ticker and st.session_state['watchlist']:
            st.session_state['current_ticker'] = st.session_state['watchlist'][0]
        st.rerun()

# --- [3] 메인 화면: 모바일 친화적 UI ---
selected_index = st.session_state['watchlist'].index(st.session_state['current_ticker']) if st.session_state['current_ticker'] in st.session_state['watchlist'] else 0
ticker = st.selectbox("🔍 분석할 종목을 선택하세요", st.session_state['watchlist'], index=selected_index)
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
# TAB 1: 📈 차트 & 기술적 분석
# ========================================================
with tab1:
    if data.empty:
        st.error("데이터가 없습니다.")
    else:
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        roll_max = data['Close'].cummax()
        max_drawdown = (data['Close'] / roll_max - 1.0).min() * 100
        current_price = data['Close'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("현재가", f"${current_price:.2f}")
        c2.metric("RSI", f"{data['RSI'].iloc[-1]:.0f}")
        c3.metric("최대 낙폭", f"{max_drawdown:.1f}%")
        
        years = st.slider("미래 예측 (년)", 1, 5, 2)
        df_train = data[['Date', 'Close']].copy().rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet()
        m.fit(df_train)
        forecast = m.predict(m.make_future_dataframe(periods=years * 365))
        
        fig_chart = go.Figure()
        fig_chart.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='markers', name='실제', marker=dict(color='gray', size=3)))
        fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='예측', line=dict(color='royalblue')))
        fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fillcolor='rgba(65,105,225,0.2)', fill='tonexty', name='신뢰구간'))
        
        # 수정된 범례 위치 속성 적용 (y, yanchor)
        fig_chart.update_layout(
            margin=dict(l=10, r=10, t=30, b=10), 
            legend=dict(orientation="h", y=-0.2, yanchor="top")
        )
        st.plotly_chart(fig_chart, use_container_width=True)

# ========================================================
# TAB 2: 🧠 AI 촉매제 리포트
# ========================================================
with tab2:
    st.subheader("뉴스 기반 AI 분석")
    if st.button("🔥 AI 리포트 생성 (Gemini 3.6)", use_container_width=True, type="primary"):
        if not api_key_input and "GEMINI_API_KEY" not in os.environ:
            st.error("사이드바에 API 키를 입력하세요.")
        else:
            with st.spinner("최신 뉴스 분석 중..."):
                stock_info = yf.Ticker(ticker)
                recent_news = stock_info.news[:5] if stock_info.news else []
                
                import datetime
                current_year = datetime.date.today().year  # 시스템의 현재 연도를 동적으로 가져옴
                
                # 뉴스 제목과 날짜(타임스탬프)를 함께 추출
                news_items = []
                for news in recent_news:
                    title = news.get('title', '제목 없음')
                    pub_time = news.get('providerPublishTime', None)
                    if pub_time:
                        date_str = datetime.datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
                        news_items.append(f"- [{date_str}] {title}")
                    else:
                        news_items.append(f"- {title}")
                
                news_text = "\n".join(news_items) if news_items else "뉴스 없음"
                
                prompt = f"""
                현재 시스템 기준 연도는 {current_year}년입니다. 아래는 종목 '{ticker}'의 수집된 최신 뉴스 목록(발행일 포함)입니다:
                {news_text}
                
                위 뉴스 목록의 [날짜]를 최우선으로 참고하여, 가장 최근에 보도된 실시간 뉴스를 기준으로 {current_year}년 현재 상황에 맞는 단기 급등 촉매제와 장기 리스크를 분석하고 요약해 줘.
                """
                
                try:
                    model = genai.GenerativeModel('gemini-3.6-flash')
                    st.markdown(model.generate_content(prompt).text)
                except Exception as e:
                    st.error(f"오류: {e}")
                    
    with st.expander("💼 기관 보유량 보기 (Smart Money)"):
        try:
            holders = yf.Ticker(ticker).institutional_holders
            if holders is not None and not holders.empty:
                st.dataframe(holders[['Holder', 'Shares']])
            else:
                st.info("데이터가 없습니다.")
        except:
            st.error("오류 발생")

# ========================================================
# TAB 3: 🎯 목표 & 시뮬레이터
# ========================================================
with tab3:
    st.subheader("10년 복리 시뮬레이터")
    with st.expander("⚙️ 내 은퇴 목표 금액 설정 (터치하여 열기)"):
        target_farm = st.number_input("스마트팜 구축", value=300000)
        target_golf = st.number_input("정기 골프 펀드", value=100000)
        target_living = st.number_input("생활 자금", value=600000)
        current_asset = st.number_input("현재 투자 원금", value=10000)
        
    total_target = target_farm + target_golf + target_living
    progress = min((current_asset / total_target) * 100, 100.0) if total_target > 0 else 0
    
    st.write(f"목표액: **${total_target:,.0f}** / 달성률: **{progress:.1f}%**")
    st.progress(progress / 100)
    
    years_sim = np.arange(0, 11)
    target_vals = current_asset * (1000 ** (years_sim/10))
    fig_sim = go.Figure(go.Scatter(x=years_sim, y=target_vals, mode='lines+markers', line=dict(color='gold')))
    
    # 수정된 범례 위치 속성 적용
    fig_sim.update_layout(
        title="10년 1000배 성장 궤적", 
        margin=dict(l=10, r=10, t=30, b=10), 
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2, yanchor="top")
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ========================================================
# TAB 4: 🌟 AI 텐배거 추천
# ========================================================
with tab4:
    st.subheader("혁신 섹터 유망주 발굴")
    sector_choice = st.selectbox("분야 선택", ["우주 항공 및 통신", "AI 바이오 헬스케어", "차세대 에너지 (SMR)", "양자 컴퓨팅"])
    
    if st.button("✨ 추천받기", use_container_width=True):
        if not api_key_input and "GEMINI_API_KEY" not in os.environ:
            st.error("사이드바에 API 키를 입력하세요.")
        else:
            with st.spinner("발굴 중..."):
                prompt = f"'{sector_choice}' 분야에서 10배 이상 성장할 잠재력 있는 미국 중소형 혁신 기업 3곳을 추천해 줘."
                try:
                    model = genai.GenerativeModel('gemini-3.6-flash')
                    st.markdown(model.generate_content(prompt).text)
                except Exception as e:
                    st.error(f"오류: {e}")

# ========================================================
# TAB 5: 📝 투자 일지
# ========================================================
with tab5:
    st.subheader("매매 복기 일지")
    journal_file = "trading_journal.csv"
    
    with st.expander("✍️ 새 일지 작성하기", expanded=True):
        with st.form("journal_form"):
            c1, c2 = st.columns(2)
            j_date = c1.date_input("날짜", date.today())
            j_action = c2.selectbox("구분", ["매수", "매도", "관망"])
            j_price = st.number_input("가격 ($)", min_value=0.0, format="%.2f")
            j_reason = st.text_area("결정 논리")
            
            if st.form_submit_button("저장", use_container_width=True):
                new_data = pd.DataFrame([[j_date, ticker, j_action, j_price, j_reason]], columns=["Date", "Ticker", "Action", "Price", "Reason"])
                if os.path.exists(journal_file):
                    df_journal = pd.concat([pd.read_csv(journal_file), new_data], ignore_index=True)
                else:
                    df_journal = new_data
                df_journal.to_csv(journal_file, index=False)
                st.success("저장 완료!")

    if os.path.exists(journal_file):
        st.dataframe(pd.read_csv(journal_file), use_container_width=True)

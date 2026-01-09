import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from advanced_process_reviews import AdvancedReviewProcessor, POS_ANCHOR, NEG_ANCHOR

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load Environment Variables
load_dotenv()

# Page Config
st.set_page_config(page_title="AI Review Analysis Presentation", layout="wide")

# Load Data
@st.cache_data
def load_data():
    with open('refined_reviews_advanced.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_data()
df = pd.DataFrame(data)

# === Gemini AI Logic ===
def generate_ai_report(df):
    """
    통계 데이터를 바탕으로 Gemini에게 요약 리포트를 요청합니다.
    """
    # 1. Try Streamlit Secrets (Cloud)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        # 2. Fallback to Environment Variable (Local .env)
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return None
    
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash for speed and stability
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 1. Prepare Prompt
    total_reviews = len(df)
    avg_rating = df['별점'].astype(int).mean()
    
    # Extract Tags
    all_tags = []
    for tags in df['태그_ABSA']:
        all_tags.extend(tags)
    tag_counts = pd.Series(all_tags).value_counts().head(5).to_string()
    
    prompt = f"""
    당신은 프로페셔널한 데이터 분석 컨설턴트입니다.
    아래 렌터카 리뷰 데이터를 분석하여 '경영진을 위한 요약 리포트'를 작성해주세요.

    [Data Check]
    - 총 리뷰 수: {total_reviews}건
    - 평균 평점: {avg_rating:.2f} / 5.0
    - 주요 이슈(태그 Top 5):
    {tag_counts}

    [Requirements]
    1. '📊 종합 성과', '🚨 주요 개선점', '💡 액션 아이템' 3가지 섹션으로 나누어 작성하세요.
    2. 말투는 정중하고 전문적인 '하십시오'체를 사용하세요.
    3. 불렛 포인트를 사용하여 가독성을 높이세요.
    4. 한국어로 작성하세요.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error creating report: {str(e)}"

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 대시보드", "🧩 분석 원리 (How it Works)", "🧪 실시간 체험 (Live Demo)"])

# === Tab 1: Dashboard ===
with tab1:
    st.header("종합 분석 대시보드")
    
    # --- AI Report Section ---
    st.markdown("### 🤖 AI 인사이트 리포트")
    
    report_file = "ai_report.md"
    
    # Try Secrets then Env
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.getenv("GEMINI_API_KEY")
    
    # Refresh Button
    force_refresh = st.button("🔄 리포트 최신화 (Re-generate)")
    
    markdown_content = ""
    
    # 1. Load existing report if available and not forcing refresh
    if os.path.exists(report_file) and not force_refresh:
        with open(report_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()
    
    # 2. Generate new report if missing or forced
    elif api_key and api_key != "YOUR_API_KEY_HERE":
        with st.spinner("Gemini가 데이터를 분석하고 있습니다... (약 5~10초 소요)"):
            markdown_content = generate_ai_report(df)
            if markdown_content:
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
    
    # 3. Display Result
    if markdown_content:
        st.info(markdown_content)
    elif not api_key or api_key == "YOUR_API_KEY_HERE":
        st.warning("⚠️ Streamlit Cloud의 'Secrets' 또는 로컬 `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
    else:
        st.error("리포트 생성 실패. API Key나 네트워크 상태를 확인해주세요.")

    st.markdown("---") 

    # 1. Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("총 분석 리뷰 수", f"{len(df)}건")
    avg_rating = df['별점'].astype(int).mean()
    col2.metric("평균 평점", f"{avg_rating:.2f}점")
    
    # Flatten Tags for Analysis
    all_tags = []
    for tags in df['태그_ABSA']:
        all_tags.extend(tags)
    
    tag_counts = pd.Series(all_tags).value_counts().reset_index()
    tag_counts.columns = ['Tag', 'Count']
    
    # 2. Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("AI 분석 태그 분포")
        fig_tags = px.bar(tag_counts.head(10), x='Count', y='Tag', orientation='h', color='Count', title="Top 10 AI Tags")
        st.plotly_chart(fig_tags, use_container_width=True)
        
    with col_chart2:
        st.subheader("별점 분포")
        rating_counts = df['별점'].value_counts().sort_index()
        fig_rating = px.pie(values=rating_counts.values, names=rating_counts.index, title="Rating Distribution", hole=0.4)
        st.plotly_chart(fig_rating, use_container_width=True)

    # 3. Data Table
    st.subheader("상세 리뷰 데이터 (필터링 가능)")
    st.dataframe(df)

# === Tab 2: How it Works ===
with tab2:
    st.header("SBERT Zero-Shot ABSA 원리")
    st.markdown("""
    이 시스템은 미리 학습된 **SBERT (Sentence-BERT)** 모델을 사용하여 문장의 의미를 깊이 있게 이해합니다.
    단순한 키워드 매칭이 아닌, **'의미적 거리(Semantic Distance)'**를 계산하여 분석합니다.
    """)
    
    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.info("**1단계: 속성 추출 (Aspect Extraction)**")
        st.markdown("- 리뷰에서 중요한 비즈니스 속성(청결, 비용, 응대 등)을 찾아냅니다.")
        st.markdown("- Fallback: 점수가 낮으면 사전 정의된 키워드를 참조합니다.")
        
    with col_desc2:
        st.success("**2단계: 감정 분석 (Zero-Shot Sentiment)**")
        st.markdown("- 추출된 속성에 대해 **긍정/부정 앵커 문장**과의 거리를 비교합니다.")
        st.code(f"긍정 앵커: {POS_ANCHOR}")
        st.code(f"부정 앵커: {NEG_ANCHOR}")
        st.markdown("👉 더 가까운 쪽의 감정으로 분류합니다!")

# === Tab 3: Live Demo ===
with tab3:
    st.header("🧪 실시간 AI 분석 체험")
    st.markdown("직접 리뷰를 입력하여 AI가 어떻게 분석하는지 확인해보세요.")
    
    # Initialize Processor (Cached)
    @st.cache_resource
    def get_processor():
        return AdvancedReviewProcessor()
    
    processor = get_processor()
    
    # Input State Management
    if 'review_input' not in st.session_state:
        st.session_state.review_input = ""
    
    with st.form(key='analysis_form'):
        user_input = st.text_area("리뷰 내용 입력:", key="review_input", height=100, placeholder="예시) 직원은 친절한데 차는 좀 더러웠어요.")
        submit_button = st.form_submit_button(label='AI 분석 실행')
    
    if submit_button:
        with st.spinner("SBERT 모델이 분석 중입니다..."):
            # Process manually using the processor's methods
            # 1. Mock Aspect Extraction based on keywords (Demo logic)
            # For the demo, we replicate the process() logic simply
            tags = []
            
            # Using the process logic step-by-step
            # 1. Aspects
            from advanced_process_reviews import MOCK_KEYWORDS
            found_aspects = []
            for cat, kws in MOCK_KEYWORDS.items():
                for kw in kws:
                    if kw in user_input:
                        found_aspects.append(kw)
            
            # 2. Map & Sentiment
            results = []
            for aspect in found_aspects:
                cat = processor.map_category_sbert(aspect)
                if cat == "기타": continue
                
                sentiment = processor.analyze_sentiment_sbert(user_input, cat, aspect_keyword=aspect)
                tag_str = f"{cat}({sentiment})"
                if tag_str not in tags:
                    tags.append(tag_str)
                    results.append({"속성 키워드": aspect, "타겟 문장": f"...{aspect}...", "카테고리": cat, "감정": sentiment})
            
            # PII Check
            masked_input = user_input
            import re
            phone_pattern = r'010-\d{4}-\d{4}'
            if re.search(phone_pattern, masked_input):
                 masked_input = re.sub(phone_pattern, '010-****-****', masked_input)
                 st.warning("⚠️ 개인정보(전화번호)가 감지되어 마스킹 처리되었습니다.")
            
            st.markdown("### 분석 결과")
            st.subheader(f"🏷️ 태그: {tags}")
            
            if results:
                st.table(pd.DataFrame(results))
            else:
                st.info("검출된 주요 속성이 없습니다.")
                
            st.markdown("---")
            st.markdown("**최종 저장될 텍스트 (PII 마스킹 적용)**")
            st.code(masked_input)

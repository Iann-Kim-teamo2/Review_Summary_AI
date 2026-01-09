import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math # Added for weighting calculation
from advanced_process_reviews import AdvancedReviewProcessor, POS_ANCHOR, NEG_ANCHOR

import os
import datetime # Added to fix NameError
from dotenv import load_dotenv
import google.generativeai as genai

# Load Environment Variables
load_dotenv()

# Page Config
st.set_page_config(page_title="AI Review Analysis Presentation", layout="wide")

# Load Data
# Removed @st.cache_data to ensure fresh load during development
def load_data():
    with open('refined_reviews_advanced.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_data()
df = pd.DataFrame(data)

# === Global Data Preprocessing ===
# 1. Ensure '작성일' (Date)
today = datetime.date.today()
if '작성일' not in df.columns:
    df['작성일'] = [today.strftime("%Y-%m-%d")] * len(df)
df['작성일'] = pd.to_datetime(df['작성일'], errors='coerce')

# 2. Ensure '공감수' (Likes)
if '공감수' not in df.columns:
    df['공감수'] = 0
df['공감수'] = pd.to_numeric(df['공감수'], errors='coerce').fillna(0).astype(int)

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
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 1. Prepare Prompt
    total_reviews = len(df)
    avg_rating = df['별점'].astype(int).mean()
    
    # Extract Tags
    all_tags = []
    for tags in df['태그_ABSA']:
        all_tags.extend(tags)
    tag_counts = pd.Series(all_tags).value_counts().head(5).to_string()
    
    prompt = f"""
    당신은 렌터카 서비스 리뷰 분석 AI입니다.
    아래 데이터를 바탕으로, 사용자에게 보여줄 '직관적인 요약 카드' 내용을 작성해주세요.

    [분석 데이터]
    - 총 리뷰 수: {total_reviews}건
    - 평균 평점: {avg_rating:.2f} / 5.0
    - 주요 태그(Top 5):
    {tag_counts}

    [작성 요구사항]
    아래 5가지 항목을 순서대로 작성하세요. 각 항목 사이에는 구분선(---)을 넣지 마세요.
    
    1. **한 줄 요약**: 전체적인 고객 반응을 20자 내외의 매력적인 문구로 요약 (예: "산뜻한 사용감, 트러블 걱정 없는 수분 토너!")
    2. **� 긍정 리뷰 요약**: 고객들이 만족한 점을 2~3문장으로 자연스럽게 서술 (이모지 '�'로 시작, 인용구 사용)
    3. **💭 아쉬운 점 요약**: 개선이 필요한 점을 1~2문장으로 부드럽게 서술 (색 다른 글씨체나 인용구 사용)
    4. **가장 많이 언급된 키워드**: 주요 태그 5개를 나열 (예: ` #친절 ` ` #청결 `)
    5. **종합 감성 분석**: 긍정 비율이나 전반적인 만족도를 한 문장으로 정리 (예: "긍정 리뷰 85%, 대부분의 고객들이 만족했어요!")

    [출력 포맷 예시]
    ### (한 줄 요약 내용)
    
    > 👍 (긍정 요약 내용...)
    
    > (아쉬운 점 요약 내용...)
    
    `#키워드1` `#키워드2` `#키워드3` `#키워드4` `#키워드5`
    
    **(종합 감성 분석 내용)**
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

    # === Sidebar: Analysis Settings ===
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        apply_weight = st.checkbox("최신 리뷰 가중치 적용 (Time Decay)", value=False)
        
        if apply_weight:
            half_life = st.slider("반감기 (Half-life, 일)", 10, 180, 60, help="이 기간이 지나면 리뷰의 중요도가 절반으로 줄어듭니다.")
            # Decay Constant lambda = ln(2) / half_life
            decay_lambda = 0.693 / half_life
            st.caption(f"📉 {half_life}일 전 리뷰는 50%만 반영됩니다.")

    # 1. Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("총 분석 리뷰 수", f"{len(df)}건")
    
    # Calculate Average Rating
    avg_rating = df['별점'].astype(int).mean()
    
    if apply_weight:
        # Weighting Calculation
        weights = []
        scores = df['별점'].astype(int).values
        golden_count = 0
        
        for idx, row in df.iterrows():
            # 1. Base Weight (Time Decay)
            weight = 1.0
            try:
                # row['작성일'] is already a Timestamp due to global processing
                review_date = row['작성일'].date()
                days_diff = (today - review_date).days
                # Exponential Decay
                weight = math.exp(-decay_lambda * days_diff)
            except:
                pass
            
            # 2. Golden Review Immunity (Likes >= 10 or Length >= 200)
            body_len = len(str(row.get('본문', '')))
            likes = row['공감수']
            
            if likes >= 10 or body_len >= 200:
                weight = 1.0 # Immunity Activated
                golden_count += 1
                
            weights.append(weight)
                
        # Weighted Average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else avg_rating
        
        # Display with Delta
        delta = weighted_avg - avg_rating
        col2.metric("보정 평점 (Weighted)", f"{weighted_avg:.2f}점", f"{delta:.2f} (최신 트렌드 반영)")
        
        # Display Shield Count
        if golden_count > 0:
            st.sidebar.success(f"🛡️ **{golden_count}개**의 '골든 리뷰'(고품질/인기)가 가중치 감소에서 보호되었습니다!")
            
    else:
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
    st.dataframe(
        df,
        column_config={
            "공감수": st.column_config.ProgressColumn(
                "❤️ 공감수",
                help="사용자들의 공감(좋아요) 횟수",
                format="%d",
                min_value=0,
                max_value=100,
            ),
            "작성일": st.column_config.DateColumn(
                "📅 작성일",
                format="YYYY-MM-DD",
            ),
             "별점": st.column_config.NumberColumn(
                "⭐ 별점",
                format="%d점",
            )
        },
        use_container_width=True
    )

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
        demo_rating = st.slider("이 리뷰의 별점은?", 1, 5, 3, help="별점에 따라 일관성 검사가 수행됩니다.")
        submit_button = st.form_submit_button(label='AI 분석 실행')
    
    if submit_button:
        with st.spinner("SBERT + KoNLPy 분석 중..."):
            tags = []
            results = []
            from advanced_process_reviews import MOCK_KEYWORDS
            
            # 0. 스팸 우선 탐지
            is_spam = False
            for spam_kw in MOCK_KEYWORDS.get('스팸/홍보', []):
                if spam_kw in user_input:
                    tags = ['스팸/홍보']
                    is_spam = True
                    results.append({"속성": "스팸 감지", "카테고리": "스팸/홍보", "감정": "부정", "문장": "광고성 키워드 감지됨"})
                    break
            
            if not is_spam:
                # 1. KoNLPy 자동 속성 추출
                found_aspects = processor.extract_aspects(user_input)
                
                # 2. 매핑 및 감정분석
                temp_tags = []
                for aspect in found_aspects:
                    cat = processor.map_category_sbert(aspect)
                    if cat == "기타": continue
                    if cat == "스팸/홍보": continue
                    
                    sentiment = processor.analyze_sentiment_sbert(user_input, cat, aspect_keyword=aspect)
                    tag_str = f"{cat}({sentiment})"
                    
                    if tag_str not in temp_tags:
                        temp_tags.append(tag_str)
                        results.append({"속성": aspect, "카테고리": cat, "감정": sentiment, "문장": f"...{aspect}..."})

                # 3. Consistency Check (별점 연동)
                final_tags = []
                for t in temp_tags:
                    if demo_rating == 5 and "(부정)" in t: continue
                    if demo_rating == 1 and "(긍정)" in t: continue
                    final_tags.append(t)
                tags = final_tags
            
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

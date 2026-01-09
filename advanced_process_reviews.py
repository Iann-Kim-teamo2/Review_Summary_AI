import json
import csv
import re
import math

# SBERT 임포트 시도; 없을 경우 예외 처리
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ 'sentence-transformers' 라이브러리를 찾을 수 없습니다. MOCK 모드로 실행합니다.")

# 설정 (Configuration)
# 설정 (Configuration)
INPUT_FILE = 'raw_reviews.txt'
OUTPUT_FILE_JSON = 'refined_reviews_advanced.json'

# 비즈니스 카테고리 (Target Anchors)
FIXED_CATEGORIES = ['차량 상태', '배차 및 반납', '고객 응대', '비용', '청결']

# Mock Keywords for fallback (same as before)
MOCK_KEYWORDS = {
    '차량 상태': ['깨끗', '주행거리', '브레이크', '에어컨', '기스', '스크래치', '상태', '소리', '낡'],
    '배차 및 반납': ['셔틀', '반납', '배차', '위치', '공항', '인수'],
    '고객 응대': ['직원', '친절', '응대', '말투', '설명', '싸인', '환불', '연락'],
    '비용': ['가성비', '가격', '저렴', '비용', '돈', '덤탱이', '싸'],
    '청결': ['깨끗', '청소', '냄새', '위생', '얼룩', '쓰레기', '담배', '더러', '더럽']
}

# Zero-Shot ABSA Anchors
POS_ANCHOR = "이 점은 정말 만족스럽고 훌륭해요."
NEG_ANCHOR = "이 점은 정말 실망스럽고 별로예요."

class AdvancedReviewProcessor:
    def __init__(self):
        self.model = None
        self.category_embeddings = None
        self.anchor_embeddings = None
        
        if SBERT_AVAILABLE:
            print("⏳ SBERT 모델 로딩 중 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)...")
            self.model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            self.category_embeddings = self.model.encode(FIXED_CATEGORIES, convert_to_tensor=True)
            # 앵커 임베딩 미리 계산
            self.anchor_embeddings = self.model.encode([POS_ANCHOR, NEG_ANCHOR], convert_to_tensor=True)
            print("✅ 모델 로드 및 카테고리/앵커 임베딩 완료.")

    def parse_reviews(self, input_path):
        """리뷰 파싱 로직 (공통)"""
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        blocks = re.split(r'\n\d+\.\n', content)
        if blocks[0].strip() == '': blocks = blocks[1:]
        
        parsed = []
        for block in blocks:
            lines = block.strip().split('\n')
            review = {}
            for line in lines:
                if ':' in line:
                    key, val = line.split(':', 1)
                    review[key.strip()] = val.strip()
            if review and '본문' in review:
                parsed.append(review)
        return parsed

    def extract_aspects(self, text):
        """
        1단계: 속성 추출 (Aspect Extraction)
        이상적: NER 모델 사용.
        현재: 단순 명사/키워드 추출 (Mock).
        """
        # 실제 시나리오에서는 Konlpy/Mecab을 사용하여 명사를 추출합니다.
        # 현재는 일반적인 긍정/부정 용어를 분리하여 추출을 시뮬레이션합니다.
        words = text.split()
        potential_aspects = []
        
        # 단순 휴리스틱: 의미 없는 불용어 제외, 유의미한 명사 유지
        # 이는 ABSA 추출 단계를 위한 플레이스홀더입니다.
        for word in words:
            if len(word) > 1:
                potential_aspects.append(word)
        return list(set(potential_aspects))

    def map_category_sbert(self, aspect_text):
        """2단계: SBERT를 이용한 카테고리 매핑 (Fallback 포함)"""
        # (이전 단계에서 구현된 로직 그대로 유지)
        if not SBERT_AVAILABLE:
            for cat, kws in MOCK_KEYWORDS.items():
                if any(k in aspect_text for k in kws):
                    return cat
            return "기타"

        query_embedding = self.model.encode(aspect_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.category_embeddings)[0]
        best_score_idx = int(cos_scores.argmax())
        best_score = float(cos_scores[best_score_idx])
        
        if best_score > 0.3:
            return FIXED_CATEGORIES[best_score_idx]
        else:
            for cat, kws in MOCK_KEYWORDS.items():
                if any(k in aspect_text for k in kws):
                    return cat
            return "기타"

    def analyze_sentiment_sbert(self, text, category, aspect_keyword=None):
        """
        3단계: SBERT Zero-shot Sentiment Analysis
        리뷰 텍스트와 긍정/부정 앵커 간의 유사도를 비교합니다.
        
        * 개선: 문맥 혼동을 막기 위해 aspect_keyword가 포함된 '문장'만 추출하여 분석합니다.
        """
        if not SBERT_AVAILABLE:
            # Fallback to Rule-based
            if any(x in text for x in ['좋', '최고', '친절', '깔끔', '빵빵', '편하']): return '긍정'
            if any(x in text for x in ['냄새', '불쾌', '비추', '최악', '더러', '기스']): return '부정'
            return '중립'

        target_text = text
        
        # 문장 단위로 분리하여 키워드가 포함된 문장만 분석 (Context Narrowing)
        if aspect_keyword:
            # 문장 분리 (단순 정규식 + 연결 어미 '는데', '지만' 등 포함)
            # 문맥이 섞이는 것을 방지하기 위해 연결 어미로도 나눕니다.
            split_pattern = r'(?:[.?!,\n]|는데|지만|한데|으나|하고|했는데)'
            sentences = re.split(split_pattern, text)
            for s in sentences:
                if aspect_keyword in s:
                    # 너무 짧은 조각은 문맥이 부족할 수 있으므로 3글자 이상일 때만 채택
                    if len(s.strip()) > 3:
                        target_text = s.strip()
                        break
        
        if not target_text: target_text = text # Fallback

        # 임베딩 및 유사도 계산
        query_embedding = self.model.encode(target_text, convert_to_tensor=True)
        
        # [0]=Pos, [1]=Neg
        sim_scores = util.cos_sim(query_embedding, self.anchor_embeddings)[0]
        pos_score = float(sim_scores[0])
        neg_score = float(sim_scores[1])

        # [Hybrid Scoring] SBERT 점수 보정 (키워드 가중치 적용)
        # SBERT가 짧은 문장에서 문맥을 놓치는 경우를 방지하기 위해
        # 명확한 감정 단어가 있으면 점수를 강제로 보정합니다.
        
        STRONG_POS = ['좋', '최고', '친절', '깔끔', '훌륭', '만족', '강추', '편하', '빠르', '새 차', '쾌적']
        STRONG_NEG = ['더럽', '더러', '최악', '비추', '냄새', '불쾌', '기스', '바가지', '실망', '별로', '불친절', '짜증']

        if any(k in target_text for k in STRONG_POS):
            pos_score += 0.5
            print(f"       >>> Positive Keyword Bonus Applied (+0.5)")

        if any(k in target_text for k in STRONG_NEG):
            neg_score += 0.5
            print(f"       >>> Negative Keyword Bonus Applied (+0.5)")
        
        print(f"   [Sentiment] '{category}' (Aspect: {aspect_keyword}) -> '{target_text}'\n       Pos: {pos_score:.4f} vs Neg: {neg_score:.4f}")

        if pos_score > neg_score:
            return '긍정'
        else:
            return '부정'

    def apply_guardrails(self, review):
       # (유지)
       phone_pattern = r'010-\d{4}-\d{4}'
       if '본문' in review and re.search(phone_pattern, review['본문']):
           review['본문'] = re.sub(phone_pattern, '010-****-****', review['본문'])
       if '제목' in review and re.search(phone_pattern, review['제목']):
           review['제목'] = re.sub(phone_pattern, '010-****-****', review['제목'])
       return review

    def process(self, input_path, output_path):
        reviews = self.parse_reviews(input_path)
        
        for review in reviews:
            body = review['본문']
            
            # 1. 후보 속성 추출 (Mock)
            mock_aspects_found = []
            for cat, kws in MOCK_KEYWORDS.items():
                for kw in kws:
                    if kw in body:
                        mock_aspects_found.append(kw)
            
            tags = []
            
            for aspect in mock_aspects_found:
                # 2. 카테고리 매핑
                category = self.map_category_sbert(aspect)
                if category == "기타": continue
                    
                # 3. 감정 분석 (SBERT) - 키워드 전달
                sentiment = self.analyze_sentiment_sbert(body, category, aspect_keyword=aspect)
                
                tag_str = f"{category}({sentiment})"
                if tag_str not in tags:
                    tags.append(tag_str)
                    
            review['태그_ABSA'] = tags
            self.apply_guardrails(review)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=4)
        print(f"✨ 고급 분석 완료(SBERT ABSA). 저장됨: {output_path}")

if __name__ == "__main__":
    processor = AdvancedReviewProcessor()
    processor.process(INPUT_FILE, OUTPUT_FILE_JSON)

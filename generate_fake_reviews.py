import random
import datetime

OUTPUT_FILE = '/home/teamo2/Downloads/Review_Summary/raw_reviews.txt'
COUNT = 100

# Templates
SOURCES = ['네이버 예약', '자사몰', '구글 리뷰', '인스타그램', '앱스토어', '1:1 문의', '콜센터']
TITLES_POS = [
    "최고의 여행이었습니다", "다음에 또 이용할게요", "새 차라서 너무 좋네요", "직원분들 친절해요", "가성비 갑",
    "제주도 렌트카는 무조건 여기", "깨끗하고 좋습니다", "셔틀버스가 편해요", "완전 만족합니다", "굿굿"
]
TITLES_NEG = [
    "최악의 경험", "절대 추천 안 합니다", "차가 너무 더러워요", "직원 교육 좀 시키세요", "바가지 쓰지 마세요",
    "환불해주세요", "담배 냄새 때문에 죽는 줄", "다신 이용 안 함", "사기꾼들인가요?", "별점 1점도 아깝다"
]
TITLES_MIX = [
    "나쁘지 않아요", "가격은 좋은데 차는 별로", "직원은 친절한데 차에서 냄새남", "쏘쏘합니다", "장단점이 확실하네요",
    "무난합니다", "그냥 탈만해요", "급하게 빌렸는데 괜찮네요", "아쉬운 점이 좀 있어요", "재방문 의사는 반반"
]

BODIES_POS = [
    "차량이 관리가 잘 되어 있어서 쾌적했습니다. 직원분들도 웃으면서 응대해주셔서 기분 좋았어요.",
    "가격 대비 성능이 최고네요. 에어컨도 빵빵하고 브레이크도 잘 들어요. 강추합니다.",
    "공항 셔틀이 바로 와서 대기 시간 없이 인수받았습니다. 반납도 간편하네요.",
    "아이들이랑 타는데 내부가 깨끗해서 안심이었습니다. 카시트도 상태 좋네요.",
    "이 가격에 이런 차를 빌릴 수 있다니 행운이네요. 다음 여행 때도 꼭 다시 올게요."
]
BODIES_NEG = [
    "문 열자마자 담배 냄새가 진동을 합니다. 금연차라면서요? 머리가 아파서 운전을 못하겠어요.",
    "직원 ㅌㅐ도가 진짜 최악이네요. 설명도 귀찮다는 듯이 하고 뭐 물어보면 짜증 냅니다.",
    "바퀴벌레 나왔어요. 위생 상태 진짜 토나옵니다. 환불 요청합니다.",
    "기스 났다고 돈 더 내라고 협박하네요. 내가 탈 때 찍어놓은 사진 있는데 어이가 없어서.",
    "예약한 차랑 다른 차를 주면 어떡합니까? 사과 한마디 없고 배째라 식이네요."
]
BODIES_MIX = [
    "가격은 진짜 저렴한데 차 상태는 기대하지 마세요. 그냥 굴러가는 데 의의를 둠.",
    "직원분은 정말 친절하신데 차에서 꿉꿉한 냄새가 좀 나서 아쉬웠어요.",
    "차는 거의 새 차인데 셔틀버스 기다리는 데 1시간 걸렸습니다. 체계가 좀 없는 듯.",
    "반납 처리는 빠른데 인수받을 때 너무 오래 걸려요. 그래도 차는 깨끗했습니다.",
    "가성비로 타기엔 좋은데 데이트용으로는 비추입니다. 소리가 좀 덜덜거려요."
]
BODIES_SPAM = [
    "30대 직장인 부업으로 월 500 버는 법 무료 공개합니다. 프로필 링크 확인하세요.",
    "단기간 다이어트 성공하고 싶으신 분들 연락주세요. 100% 환불 보장.",
    "재택알바 모집합니다. 하루 1시간 투자. 남녀노소 누구나 가능."
]

BODIES_LONG = [
    "처음에는 반신반의하면서 예약했는데 막상 이용해보니 기대 이상이었습니다. "
    "차량 상태가 정말 청결했고, 특히 담배 냄새가 전혀 나지 않아서 아이들과 함께 타기에 너무 좋았습니다. "
    "직원분이 카시트 설치도 직접 도와주시고, 맛집 리스트까지 문자로 보내주셔서 감동받았습니다. "
    "제주도 여행을 여러 번 왔지만 이렇게 친절한 렌터카 업체는 처음이네요. "
    "가격도 다른 곳보다 합리적이고 셔틀버스 배차 간격도 짧아서 기다리는 시간이 거의 없었습니다. "
    "다음 가족 여행 때도 무조건 여기서 예약할 생각입니다. 번창하세요!",
    
    "예전에는 대기업 렌터카만 이용했는데, 이번에 처음으로 이용해봤습니다. "
    "솔직히 걱정을 좀 했는데 차량 연식도 23년식으로 완전 새 차였고 옵션도 풀옵션이라 운전하기 너무 편했네요. "
    "블랙박스랑 내비게이션도 최신 버전으로 업데이트되어 있어서 길 찾기도 쉬웠습니다. "
    "반납할 때도 깐깐하게 트집 잡는 거 없이 쿨하게 처리해주셔서 마지막까지 기분 좋게 여행을 마무리할 수 있었습니다. "
    "지인들에게도 적극 추천하고 싶네요. 별점 5점 만점에 10점 드리고 싶습니다."
]

def generate_review(idx):
    # 60% Normal, 20% Mixed, 10% Edge(PII), 10% Spam
    rand = random.random()
    
    review = {}
    review['idx'] = idx + 1
    
    # 10% Chance for Golden Review (Long & High Quality)
    is_golden = False
    
    if rand < 0.1: # Golden Review
        review['제목'] = "강력 추천합니다 (Golden Review)"
        review['본문'] = random.choice(BODIES_LONG)
        review['별점'] = '5'
        is_golden = True
    elif rand < 0.5: # Positive
        review['제목'] = random.choice(TITLES_POS)
        review['본문'] = random.choice(BODIES_POS)
        review['별점'] = random.choice(['4', '5'])
    elif rand < 0.7: # Negative
        review['제목'] = random.choice(TITLES_NEG)
        review['본문'] = random.choice(BODIES_NEG)
        review['별점'] = random.choice(['1', '2'])
    elif rand < 0.9: # Mixed
        review['제목'] = random.choice(TITLES_MIX)
        review['본문'] = random.choice(BODIES_MIX)
        review['별점'] = random.choice(['3'])
    else: # Spam or PII
        if random.random() < 0.5: # Spam
            review['제목'] = "광고입니다"
            review['본문'] = random.choice(BODIES_SPAM)
            review['별점'] = '5'
        else: # PII
            phone = f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
            review['제목'] = f"환불 부탁드려요 ({phone})"
            review['본문'] = f"급한 사정이 생겨서 취소합니다. {phone} 여기로 꼭 연락 주세요."
            review['별점'] = '4'

    # Date generation (last 365 days for Weighting Test)
    delta = random.randint(0, 365)
    date = datetime.date.today() - datetime.timedelta(days=delta)
    review['작성일'] = date.strftime("%Y-%m-%d")
    review['출처'] = random.choice(SOURCES)
    
    # Likes generation
    if is_golden:
        review['공감수'] = random.randint(10, 100) # High likes for golden reviews
    else:
        review['공감수'] = random.randint(0, 5) # Low likes for normal stats
    
    return review

def save_reviews(reviews):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in reviews:
            f.write(f"{r['idx']}.\n")
            f.write(f"제목: {r['제목']}\n")
            f.write(f"본문: {r['본문']}\n")
            f.write(f"별점: {r['별점']}\n")
            f.write(f"작성일: {r['작성일']}\n")
            f.write(f"출처: {r['출처']}\n")
            f.write(f"공감수: {r['공감수']}\n") # New field
            f.write("\n")
    print(f"✅ {COUNT} synthetic reviews generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    data = [generate_review(i) for i in range(COUNT)]
    save_reviews(data)

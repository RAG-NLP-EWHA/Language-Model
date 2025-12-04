import os
import time
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI # OpenAI 라이브러리 추가

# --- 설정 변수 (Configuration Variables) ---
# NOTE: OpenAI API Key는 환경 변수 'OPENAI_API_KEY'에 설정되어 있어야 합니다.
BASE_DIR = "/aix23604/"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATASET_PATH = os.path.join(BASE_DIR, "hotpotqa_val_700_question.csv")

# 모델 설정
GPT_MODEL = "gpt-4.1" # 사용할 GPT 모델 지정
MAX_NEW_TOKENS = 256 # 생성할 최대 토큰 수

# 클라이언트 초기화
# 환경 변수 OPENAI_API_KEY를 자동으로 찾습니다.
try:
    client = OpenAI()
    print("OpenAI client initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client. Is OPENAI_API_KEY set? Error: {e}")
    client = None

# --- 함수 정의 (Function Definitions) ---

def generate_answer_gpt(question: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate answer using GPT"""
    if client is None:
        return "ERROR: OpenAI client not initialized."

    # ChatCompletion 프롬프트 구성
    messages = [
        {"role": "system", "content": "You are a helpful QA assistant. Answer the user's question concisely. If you don't know the answer, reply with: I don't know."},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    try:
        # API 호출
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        # 답변 텍스트 추출
        slm_answer = completion.choices[0].message.content.strip()
        
        if slm_answer.startswith("Answer:"):
            slm_answer = slm_answer[7:].strip()
            
        return slm_answer
        
    except Exception as e:
        # API 호출 중 오류 발생 시 처리
        return f"ERROR in generate_answer_gpt: {type(e).__name__}: {str(e)}"

# --- 메인 실행 로직 (Main Execution Logic) ---

print("\nLoading dataset...")
try:
    dataset = pd.read_csv(DATASET_PATH)
    # 테스트용 5개만 주석 처리 해제 가능
    # dataset = dataset.head(5)
    print(f"Loaded {len(dataset)} samples from {DATASET_PATH}")
except FileNotFoundError:
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
    exit()

results = []

print(f"\nProcessing {len(dataset)} samples using {GPT_MODEL}...")

for i, row in enumerate(tqdm(dataset.itertuples(index=False), total=len(dataset))):
    question = row.question
    gold_answer = row.answer
    
    print(f"\n{'='*60}")
    print(f"Sample {i+1}:")
    print(f"Q: {question}")
    
    start_time = time.time()
    
    try:
        # GPT API를 사용하여 답변 생성
        slm_answer = generate_answer_gpt(question)
        print(f"A: {slm_answer}")
    except Exception as e:
        # API 클라이언트 자체 오류나 기타 예상치 못한 오류 처리
        slm_answer = f"ERROR: Unhandled Exception: {type(e).__name__}: {str(e)}"
        print(slm_answer)
        
    time_v = time.time() - start_time

    result = {
        "index": i,
        "id": getattr(row, "id", ""),
        "question": question,
        "gold_answer": gold_answer,
        "slm_answer": slm_answer,
        "slm_time_sec": round(time_v, 2),
    }
    results.append(result)

# 결과 저장
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.DataFrame(results)
output_csv_path = os.path.join(OUTPUT_DIR, "gpt4.1_answers.csv")
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(f"Results saved to {output_csv_path}")
print(f"Total samples: {len(results)}")
print(f"Average time: {df['slm_time_sec'].mean():.2f} seconds")
print(f"Total time: {df['slm_time_sec'].sum():.2f} seconds")

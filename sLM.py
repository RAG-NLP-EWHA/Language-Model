import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from tqdm.auto import tqdm

BASE_DIR = "/mnt/aix23604/"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

USE_LLAMA = False 

if USE_LLAMA:
    # LLM: Meta Llama 3 8B Instruct (gated repo)
    SLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
else:
    # LLM: Phi-3.5 mini instruct (public)
    SLM_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
    
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

def load_models(device: str = DEVICE):
    """Load embedding model and Phi-3.5 SLM."""
    print(f"[1/4] Loading models on device: {device}")

    # Embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_ID, device=device)

    # SLM
    slm_config = AutoConfig.from_pretrained(SLM_MODEL_ID, trust_remote_code=True)
    max_context = getattr(slm_config, "max_position_embeddings", 4096)

    slm_tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_ID, trust_remote_code=True)
    
    if slm_tokenizer.pad_token is None:
        slm_tokenizer.pad_token = slm_tokenizer.eos_token
    
    slm_model = AutoModelForCausalLM.from_pretrained(
        SLM_MODEL_ID,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    slm_model.eval()

    print(f"   - MAX_CONTEXT: {max_context}")
    print(f"   - Model device: {slm_model.device}")
    return embed_model, slm_tokenizer, slm_model, max_context

def generate_answer(question: str, tokenizer, model, max_new_tokens=256):
    """Generate answer using SLM"""

    prompt = f"""You are a helpful QA assistant. 
Answer the user's question concisely.
If you don't know the answer, reply with: I don't know.

Question: {question}
Answer:"""
    
    # 토큰화
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_length = inputs.input_ids.shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            return_dict_in_generate=False,
        )
    
    # 디코딩
    generated_ids = outputs[0][input_length:] # 생성된 토큰만 추출 (프롬프트 제거)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # 답변 추출
    if generated_text.startswith("Answer:"):
        generated_text = generated_text[7:].strip()
    
    if "\nQuestion:" in generated_text:
        generated_text = generated_text.split("\nQuestion:")[0].strip()
        
    return generated_text

print("Loading models...")
embed_model, slm_tokenizer, slm_model, max_context = load_models(DEVICE)
print("Models loaded!")

print("\nLoading dataset...")
DATASET_PATH = os.path.join(BASE_DIR, "hotpotqa_val_700_question.csv")
dataset = pd.read_csv(DATASET_PATH)

# 테스트용 5개만
# dataset = dataset.head(5)
# print(f"Loaded {len(dataset)} samples for testing")

results = []

print(f"\nProcessing {len(dataset)} samples...")

for i, row in enumerate(tqdm(dataset.itertuples(index=False), total=len(dataset))):
    question = row.question
    gold_answer = row.answer
    
    print(f"\n{'='*60}")
    print(f"Sample {i+1}:")
    print(f"Q: {question}")
    
    start_time = time.time()
    
    try:
        slm_answer = generate_answer(question, slm_tokenizer, slm_model)
        print(f"A: {slm_answer}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        slm_answer = f"ERROR: {str(e)}"
        
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
output_csv_path = os.path.join(OUTPUT_DIR, "slm_answers.csv")
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(f"Results saved to {output_csv_path}")
print(f"Total samples: {len(results)}")
print(f"Average time: {df['slm_time_sec'].mean():.2f} seconds")
print(f"Total time: {df['slm_time_sec'].sum():.2f} seconds")
# -*- coding: utf-8 -*-
"""
Google Colab에서 koGPT2 모델을 k-fold 교차 검증으로 학습시키는 스크립트

사용 방법:
1. Google Colab에서 이 파일을 업로드하거나 내용을 복사
2. Colab 셀에서 실행
3. 각 fold마다 학습된 모델을 Google Drive에 저장하거나 다운로드
"""

import os
import time
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset

# CUDA 디버깅 활성화 (device-side assert 오류 디버깅용)
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ===== 설정 =====
MODEL_ID = "skt/kogpt2-base-v2"  # koGPT2 모델 ID
OUTPUT_DIR = "./kogpt2_finetuned"  # 학습된 모델 저장 경로
K_FOLDS = 5  # k-fold 교차 검증의 k 값 (연산량 줄이려면 3 권장)
EPOCHS = 2  # 학습 에포크 수 (연산량 줄이려면 1로 설정)
LEARNING_RATE = 5e-5  # 학습률
BATCH_SIZE = 4  # 배치 크기 (GPU 메모리에 따라 조정: 2, 4, 8, 16)
GRADIENT_ACCUMULATION_STEPS = 4  # 그래디언트 누적 스텝
MAX_DATA_SIZE = 100  # 사용할 최대 데이터 개수 (전체 사용하려면 None)

# ===== 디바이스 설정 =====
# FORCE_CPU = True로 설정하면 GPU를 사용하지 않고 CPU만 사용합니다
# 기본값은 False (GPU 전용 학습)
FORCE_CPU = False  # True로 설정하면 CPU 모드로 강제 실행

# GPU 전용 학습 모드: GPU만 사용하고 CPU로 자동 전환하지 않음
# ⚠️ CUDA 오류가 계속 발생하면 False로 변경하거나 FORCE_CPU = True로 설정하세요
GPU_ONLY = True  # True: GPU만 사용, GPU 오류 시 중단. False: GPU 오류 시 CPU로 전환


def download_kpoem_data(max_size: int = 100) -> List[Dict]:
    """
    Hugging Face에서 KPoeM 데이터셋을 다운로드합니다.
    
    Args:
        max_size: 로드할 최대 데이터 개수
    
    Returns:
        데이터 리스트 (각 항목은 {'text': 원문, 'poem': 시} 형식)
    """
    print(f"\n{'='*80}")
    print(f"[KPoeM 데이터셋 다운로드]")
    print(f"  - 소스: Hugging Face (AKS-DHLAB/KPoEM)")
    print(f"  - 최대 개수: {max_size}")
    print(f"{'='*80}\n")
    
    try:
        # KPoEM 데이터셋 로드
        print(f"[다운로드 시도] AKS-DHLAB/KPoEM...")
        dataset = load_dataset(
            "csv",
            data_files={
                "train": "hf://datasets/AKS-DHLAB/KPoEM/KPoEM_poem_dataset_v4.tsv"
            },
            delimiter="\t",
            encoding="utf-8",
            quoting=3,  # QUOTE_NONE
        )
        dataset = dataset["train"]
        print(f"✅ 데이터셋 로드 성공: {len(dataset)}개 샘플")
        
        # 데이터 형식 변환
        normalized_data = []
        for i, item in enumerate(dataset):
            if max_size and i >= max_size:
                break
            
            if 'text' in item and item['text']:
                poem_text = str(item['text']).strip()
                normalized_data.append({
                    'text': poem_text,
                    'poem': poem_text
                })
        
        print(f"✅ {len(normalized_data)}개 데이터 변환 완료")
        return normalized_data
        
    except Exception as e:
        print(f"❌ 데이터셋 다운로드 실패: {e}")
        raise


def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    """
    간단한 키워드 추출 (Colab에서는 복잡한 라이브러리 없이)
    실제로는 더 정교한 키워드 추출이 필요하지만, 여기서는 간단히 처리
    """
    # 간단한 키워드 추출: 명사 위주로 추출 (실제로는 형태소 분석 필요)
    # 여기서는 텍스트를 단어로 나누고 길이가 2 이상인 것만 선택
    words = text.split()
    keywords = [w for w in words if len(w) >= 2][:max_keywords]
    return keywords if keywords else ["시", "감정"]


def classify_emotion_simple(text: str) -> Dict[str, str]:
    """
    간단한 감정 분류 (Colab에서는 복잡한 모델 없이)
    실제로는 더 정교한 감정 분류가 필요하지만, 여기서는 간단히 처리
    """
    # 간단한 감정 분류: 긍정/부정 키워드 기반
    positive_words = ["좋", "행복", "기쁨", "사랑", "희망", "밝", "따뜻"]
    negative_words = ["슬", "우울", "아픔", "힘듦", "어둠", "차갑"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        mood = "밝은"
        emotion = "긍정"
    elif neg_count > pos_count:
        mood = "어두운"
        emotion = "부정"
    else:
        mood = "잔잔한"
        emotion = "중립"
    
    return {'mood': mood, 'emotion': emotion}


def build_prompt_kogpt2(keywords: List[str], mood: str, lines: int, original_text: str) -> str:
    """
    koGPT2용 프롬프트 생성
    """
    kw_str = ", ".join(keywords[:10])
    
    prompt = f"""Write a Korean poem (한국어 시) based on the keywords and mood below.

**CRITICAL: Language Requirement**
- You MUST write ONLY in Korean (Hangul, 한글).
- Do NOT use Chinese characters (한자), Japanese characters, English, or any other language.
- Use ONLY Korean characters (가-힣) and Korean punctuation.
- The output MUST be a Korean poem.

**Output Requirements**
- Output ONLY the poem text (no title, no explanation, no keywords, no numbering).
- The output MUST be in poem form with line breaks.
- Write EXACTLY {lines} lines (one line per verse; no empty lines).

**Content**
- Keywords: {kw_str}
- Mood: {mood}
{f'**Original Prose (Context)**\n\"\"\"{original_text.strip()}\"\"\"\n' if original_text else ''}

**Style Rules (strict)**
1) Keep each line short and lyrical.
2) Show, don't tell.
3) Avoid plain narration and diary-like tone.
4) In Korean, avoid declarative endings like "~다", "~이다", "~했다".
5) Avoid explicit subjects/time markers like "나는", "그는/그녀는", "오늘은/어제는".

Poem:
"""
    return prompt


def convert_poem_to_prose(poem: str) -> str:
    """
    시를 산문(일상 글)으로 변환합니다.
    시의 줄바꿈을 제거하고 자연스러운 문장으로 만듭니다.
    
    Args:
        poem: 시 텍스트
    
    Returns:
        산문 텍스트
    """
    if not poem:
        return ""
    
    # 줄바꿈을 공백으로 변환
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    
    # 시의 각 줄을 연결하여 산문으로 만들기
    # 시적 표현을 일상적인 표현으로 변환
    prose = " ".join(lines)
    
    # 너무 긴 텍스트는 자르기 (토큰 길이 제한을 위해)
    if len(prose) > 400:  # 대략 400자로 제한
        prose = prose[:400] + "..."
    
    # 시적 표현을 일상 표현으로 변환
    # 예: "꽃처럼" → "꽃과 같이", "별처럼" → "별과 같이"
    prose = prose.replace("처럼", "과 같이")
    prose = prose.replace("같이", "처럼")
    
    # 문장 부호 정리
    if not prose.endswith(('.', '!', '?', '다', '요', '...')):
        prose += "."
    
    return prose


def prepare_training_data(train_data: List[Dict], tokenizer) -> List[Dict]:
    """
    학습 데이터를 koGPT2 학습 형식으로 변환합니다.
    1. 시 원문만으로 학습하여 시의 형식/구조/표현 방식을 학습
    2. 산문 → 시 변환을 학습하여 산문의 의미를 이해하고 시를 생성하도록 학습
    
    Args:
        train_data: 학습 데이터 리스트 (각 항목은 {'text': 원문, 'poem': 시})
        tokenizer: koGPT2 토크나이저
    
    Returns:
        학습용 데이터셋 리스트
    """
    training_examples = []
    total = len(train_data)
    
    print(f"\n[학습 데이터 준비]")
    print(f"  - 총 {total}개 데이터 처리 중...")
    print(f"  - 학습 형식:")
    print(f"    1. 시 원문 학습 (시의 형식/구조/표현 방식 학습)")
    print(f"    2. 산문 → 시 변환 학습 (산문의 의미를 이해하고 시 생성)")
    
    poem_only_count = 0
    prose_to_poem_count = 0
    
    for idx, item in enumerate(train_data, 1):
        if idx % 10 == 0 or idx == total:
            print(f"  - 진행 중: {idx}/{total} ({idx*100//total}%)")
        
        poem = item.get('poem', '') or item.get('text', '')
        
        if not poem:
            continue
        
        # ===== 1. 시 원문만으로 학습 (시의 형식/구조/표현 방식 학습) =====
        # 시 원문을 그대로 학습하여 koGPT2가 시가 무엇인지, 시의 형식이 어떤 것인지 학습
        # "시: " 없이 시만 학습하여 "시: " 반복 생성 패턴 방지
        # 대신 산문→시 변환 학습에서만 "시: " 패턴 학습
        poem_text = poem.strip()  # "시: " 제거 - 반복 패턴 방지
        training_examples.append({
            'text': poem_text,
            'prose': '',  # 시 원문만 있는 경우
            'poem': poem.strip()
        })
        poem_only_count += 1
        
        # ===== 2. 산문 → 시 변환 학습 =====
        # 시를 산문으로 변환 (일상 글처럼 만들기)
        # KPoeM 데이터셋에는 원문이 없으므로 시를 산문으로 변환
        prose = convert_poem_to_prose(poem)
        
        if prose:
            # "산문: [산문 내용]\n시: [시 내용]" 형식으로 학습하여
            # 모델이 "산문: [내용]"을 입력받으면 그 의미를 이해하고
            # "시: [내용]"을 생성하도록 학습
            full_text = f"산문: {prose.strip()}\n시: {poem.strip()}"
            
            training_examples.append({
                'text': full_text,
                'prose': prose.strip(),
                'poem': poem.strip()
            })
            prose_to_poem_count += 1
    
    print(f"  ✅ 처리 완료: {len(training_examples)}개 학습 예제 생성")
    print(f"    - 시 원문만 학습: {poem_only_count}개 (\"시: \" 없이 시만 학습)")
    print(f"    - 산문 → 시 변환 학습: {prose_to_poem_count}개")
    print(f"  - 학습 형식 예시:")
    if training_examples:
        # 시 원문만 있는 예시
        poem_example = next((ex for ex in training_examples if not ex['prose']), None)
        if poem_example:
            print(f"    [시 원문 학습] {poem_example['text'][:60]}...")
        
        # 산문 → 시 변환 예시
        prose_example = next((ex for ex in training_examples if ex['prose']), None)
        if prose_example:
            print(f"    [산문→시 변환] 입력(산문): {prose_example['prose'][:40]}...")
            print(f"                  학습 형식: \"산문: {prose_example['prose'][:30]}...\\n시: {prose_example['poem'][:30]}...\"")
    
    return training_examples


def train_kogpt2_model(
    train_data: List[Dict],
    output_dir: str,
    epochs: int = 2,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4
) -> str:
    """
    koGPT2 모델을 학습시킵니다.
    
    Args:
        train_data: 학습 데이터 리스트
        output_dir: 학습된 모델 저장 디렉토리
        epochs: 학습 에포크 수
        learning_rate: 학습률
        batch_size: 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝
    
    Returns:
        학습된 모델 경로
    """
    print(f"\n{'='*80}")
    print(f"[koGPT2 모델 학습 시작]")
    print(f"  - 학습 데이터: {len(train_data)}개")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"{'='*80}\n")
    
    # 모델 및 토크나이저 로드
    print(f"[1/5] 모델 및 토크나이저 로딩: {MODEL_ID}")
    
    # GPU 메모리 정리 (이전 fold의 잔여 메모리 제거)
    # GPU가 오염된 경우를 대비해 안전하게 처리
    if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        try:
            # GPU 상태 확인 (간단한 연산으로)
            _ = torch.cuda.get_device_name(0)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  ⚠️ GPU 메모리 정리 중 오류: {e}")
            print(f"  💡 GPU가 오염되었습니다. CPU 모드로 자동 전환합니다.")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)  # use_fast는 기본값 사용
    
    # pad_token 설정 전 vocab 크기 확인
    initial_vocab_size = len(tokenizer)
    initial_base_vocab_size = tokenizer.vocab_size
    
    print(f"  - 초기 vocab 크기: {initial_vocab_size} (base: {initial_base_vocab_size})")
    
    # 초기 상태에서 이미 추가 토큰이 있는지 확인
    if initial_vocab_size > initial_base_vocab_size:
        diff = initial_vocab_size - initial_base_vocab_size
        print(f"  ⚠️ 토크나이저 로드 시 이미 추가 토큰 {diff}개가 포함되어 있습니다.")
        print(f"  🔧 토크나이저를 base_vocab_size에 맞춰 조정합니다...")
        
        # 추가 토큰이 있다면, pad_token 설정 시 새 토큰을 추가하지 않도록 주의
        # 토크나이저의 vocab_size를 base_vocab_size로 제한
        # (실제로는 모델과 일치시키기 위해 모델을 리사이즈하는 것이 더 안전)
    
    # pad_token 설정 (추가 토큰을 만들지 않고 eos_token 재사용)
    # koGPT2는 기본적으로 pad_token이 없으므로 eos_token을 재사용
    if tokenizer.pad_token is None:
        # eos_token을 pad_token으로 재사용 (새 토큰 추가 안 함)
        eos_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.eos_token
        
        # 방법: special_tokens_map을 먼저 수정한 후 속성 설정
        # 이렇게 하면 add_special_tokens가 호출되지 않음
        if hasattr(tokenizer, 'special_tokens_map'):
            # special_tokens_map에 pad_token을 eos_token으로 매핑
            original_map = tokenizer.special_tokens_map.copy()
            tokenizer.special_tokens_map['pad_token'] = eos_token
        
        # 속성 직접 설정 (add_special_tokens 호출 안 함)
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
        
        # vocab 크기 재확인
        after_vocab_size = len(tokenizer)
        if after_vocab_size > initial_vocab_size:
            print(f"  ⚠️ 경고: pad_token 설정 후 vocab 크기가 증가했습니다!")
            print(f"     이전: {initial_vocab_size}, 이후: {after_vocab_size}")
            print(f"  🔧 토크나이저를 다시 로드하고 더 안전한 방법으로 설정합니다...")
            
            # 토크나이저를 다시 로드
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
            
            # 가장 안전한 방법: 속성만 직접 설정 (special_tokens_map 수정 안 함)
            # 이렇게 하면 새 토큰이 추가되지 않음
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # vocab 크기 최종 확인
            final_vocab_size = len(tokenizer)
            if final_vocab_size > initial_vocab_size:
                print(f"  ⚠️ 여전히 vocab 크기가 증가했습니다: {final_vocab_size}")
                print(f"  💡 이는 토크나이저 자체에 이미 추가 토큰이 포함되어 있기 때문입니다.")
                print(f"  ✅ 모델 리사이즈로 자동 처리됩니다.")
    else:
        print(f"  ✅ pad_token이 이미 설정되어 있습니다: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # 최종 vocab 크기 확인
    actual_vocab_size = len(tokenizer)  # 실제 토크나이저 크기 (추가 토큰 포함)
    base_vocab_size = tokenizer.vocab_size  # 기본 vocab 크기
    
    print(f"  - Base vocab size: {base_vocab_size}")
    print(f"  - Actual vocab size (len(tokenizer)): {actual_vocab_size}")
    print(f"  - Pad token ID: {tokenizer.pad_token_id}")
    print(f"  - EOS token ID: {tokenizer.eos_token_id}")
    
    if actual_vocab_size > base_vocab_size:
        diff = actual_vocab_size - base_vocab_size
        print(f"  ⚠️ 추가 토큰: {diff}개 (모델 리사이즈 필요)")
    
    # 디바이스 선택 (CUDA > CPU)
    # FP16 gradient scaling 문제를 피하기 위해 float32 사용
    device = "cpu"  # 기본값
    dtype = torch.float32
    
    # GPU 전용 모드: FORCE_CPU가 True가 아니면 GPU만 사용
    if FORCE_CPU:
        print(f"  - 디바이스: CPU (FORCE_CPU=True로 강제 설정)")
        device = "cpu"
    elif not torch.cuda.is_available():
        if GPU_ONLY:
            raise RuntimeError(
                "❌ GPU가 감지되지 않습니다. GPU 전용 모드(GPU_ONLY=True)이므로 학습을 중단합니다.\n"
                "💡 해결 방법:\n"
                "   1. Colab에서 GPU 런타임 선택: 런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU\n"
                "   2. GPU 할당량이 소진되었을 수 있습니다. 몇 시간 후 다시 시도하세요.\n"
                "   3. CPU 모드로 학습하려면 코드 상단에서 FORCE_CPU = True로 설정하세요."
            )
        else:
            print(f"  - 디바이스: CPU (GPU 없음)")
            device = "cpu"
    else:
        # GPU 사용 가능
        try:
            # GPU 상태 확인 (최소한으로만)
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device = "cuda"
                print(f"  - 디바이스: CUDA")
                # GPU 정보 출력 (오류 발생 가능하므로 안전하게)
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_props = torch.cuda.get_device_properties(0)
                    print(f"  - GPU 이름: {gpu_name}")
                    print(f"  - GPU 메모리: {gpu_props.total_memory / (1024**3):.1f}GB")
                except:
                    pass  # GPU 정보 출력 실패해도 계속 진행
                print(f"  - 데이터 타입: float32 (FP16 gradient scaling 문제 방지)")
            else:
                if GPU_ONLY:
                    raise RuntimeError("GPU 디바이스가 없습니다. GPU 전용 모드이므로 학습을 중단합니다.")
                else:
                    device = "cpu"
                    print(f"  - 디바이스: CPU (GPU 디바이스 없음)")
        except Exception as gpu_error:
            if GPU_ONLY:
                raise RuntimeError(
                    f"❌ GPU 사용 불가: {gpu_error}\n"
                    "💡 GPU 전용 모드(GPU_ONLY=True)이므로 학습을 중단합니다.\n"
                    "🔧 해결 방법:\n"
                    "   1. Colab 런타임 재시작: 런타임 → 런타임 다시 시작\n"
                    "   2. CPU 모드로 학습하려면 코드 상단에서 FORCE_CPU = True로 설정하세요."
                )
            else:
                print(f"  ⚠️ GPU 사용 불가: {gpu_error}")
                print(f"  💡 CPU 모드로 전환합니다.")
                device = "cpu"
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # 모델 로딩 (CPU에서 먼저 로드하여 리사이즈 후 GPU로 이동)
    # GPU로 바로 이동하면 리사이즈 시 오류가 발생할 수 있음
    print(f"  - 모델 로딩 중 (CPU에서 먼저 로드)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )
    
    # 모델을 명시적으로 CPU에 유지 (리사이즈 전까지)
    model = model.cpu()
    
    # loss_type 경고 해결
    if hasattr(model.config, 'loss_type') and model.config.loss_type is None:
        try:
            if hasattr(model.config, '__dict__'):
                if 'loss_type' in model.config.__dict__:
                    delattr(model.config, 'loss_type')
        except:
            pass
    
    # ===== 중요: 모델과 토크나이저의 vocab_size 일치 확인 및 수정 =====
    # pad_token 설정 전에 초기 크기 확인
    model_vocab_size_initial = model.config.vocab_size
    tokenizer_vocab_size_initial = len(tokenizer)  # 초기 토크나이저 크기
    tokenizer_base_vocab_size = tokenizer.vocab_size  # 토크나이저 기본 크기
    
    print(f"\n  📊 Vocab 크기 비교 (초기):")
    print(f"     - 모델 vocab_size: {model_vocab_size_initial}")
    print(f"     - 토크나이저 base vocab_size: {tokenizer_base_vocab_size}")
    print(f"     - 토크나이저 실제 크기 (len): {tokenizer_vocab_size_initial}")
    
    # pad_token 설정 (새 토큰 추가하지 않고 eos_token 재사용)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # pad_token 설정 후 크기 확인
    tokenizer_vocab_size_after = len(tokenizer)
    
    # 모델과 토크나이저 크기 불일치 확인 및 수정 (강제)
    # 초기부터 불일치하거나 pad_token 설정 후 불일치하는 경우 모두 처리
    needs_resize = False
    target_vocab_size = tokenizer_vocab_size_after  # pad_token 설정 후 크기 사용
    
    if model_vocab_size_initial != tokenizer_vocab_size_initial:
        # 초기부터 불일치하는 경우
        print(f"  ⚠️ Vocab 크기 불일치 감지 (초기):")
        print(f"     모델: {model_vocab_size_initial}, 토크나이저: {tokenizer_vocab_size_initial}")
        needs_resize = True
        target_vocab_size = tokenizer_vocab_size_initial
    
    if tokenizer_vocab_size_after > tokenizer_vocab_size_initial:
        # pad_token 설정으로 크기가 증가한 경우
        print(f"  ⚠️ pad_token 설정으로 토크나이저 크기가 증가했습니다:")
        print(f"     이전: {tokenizer_vocab_size_initial} → 이후: {tokenizer_vocab_size_after}")
        needs_resize = True
        target_vocab_size = tokenizer_vocab_size_after
    
    if model_vocab_size_initial != target_vocab_size:
        # 최종적으로 모델과 토크나이저 크기가 다른 경우
        needs_resize = True
    
    if needs_resize:
        print(f"  🔧 모델 임베딩 레이어를 토크나이저 크기({target_vocab_size})로 리사이즈합니다...")
        
        # 리사이즈 전에 모델이 CPU에 있는지 확인
        if next(model.parameters()).is_cuda:
            print(f"  ⚠️ 모델이 GPU에 있습니다. CPU로 이동 후 리사이즈합니다...")
            model = model.cpu()
        
        # CPU에서 리사이즈 수행
        model.resize_token_embeddings(target_vocab_size)
        model_vocab_size_after = model.config.vocab_size
        print(f"  ✅ 모델 vocab_size 업데이트: {model_vocab_size_initial} → {model_vocab_size_after}")
        
        # 리사이즈 후 즉시 확인
        if model_vocab_size_after != target_vocab_size:
            raise ValueError(
                f"❌ 모델 리사이즈 실패!\n"
                f"   목표: {target_vocab_size}, 실제: {model_vocab_size_after}\n"
                f"   이 불일치가 CUDA 오류의 원인일 수 있습니다."
            )
        
        # 리사이즈 후 모델이 여전히 CPU에 있는지 확인
        if next(model.parameters()).is_cuda:
            print(f"  ⚠️ 리사이즈 후 모델이 GPU에 있습니다. CPU로 이동합니다...")
            model = model.cpu()
    else:
        model_vocab_size_after = model_vocab_size_initial
        print(f"  ✅ Vocab 크기 일치 확인 (리사이즈 불필요)")
        
        # 모델이 CPU에 있는지 확인
        if next(model.parameters()).is_cuda:
            print(f"  ⚠️ 모델이 GPU에 있습니다. CPU로 이동합니다...")
            model = model.cpu()
    
    # 최종 크기 확인
    final_model_vocab_size = model.config.vocab_size
    final_tokenizer_vocab_size = len(tokenizer)
    
    print(f"\n  📊 Vocab 크기 비교 (최종):")
    print(f"     - 모델 vocab_size: {final_model_vocab_size}")
    print(f"     - 토크나이저 실제 크기 (len): {final_tokenizer_vocab_size}")
    
    # 최종 확인: 모델과 토크나이저가 완전히 일치하는지 확인
    if final_model_vocab_size != final_tokenizer_vocab_size:
        print(f"  ❌ 최종 확인 실패: vocab_size 불일치!")
        print(f"     모델: {final_model_vocab_size}, 토크나이저: {final_tokenizer_vocab_size}")
        raise ValueError(
            f"❌ Vocab 크기 불일치를 해결할 수 없습니다!\n"
            f"   모델: {final_model_vocab_size}, 토크나이저: {final_tokenizer_vocab_size}\n"
            f"   이 불일치가 CUDA 오류의 원인일 수 있습니다."
        )
    
    # pad_token_id가 vocab_size 범위 내인지 확인
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id >= final_model_vocab_size or pad_token_id < 0:
        print(f"  ⚠️ 경고: pad_token_id({pad_token_id})가 vocab_size({final_model_vocab_size}) 범위를 벗어납니다!")
        print(f"  🔧 eos_token_id로 대체합니다...")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id >= final_model_vocab_size or pad_token_id < 0:
            raise ValueError(f"eos_token_id({pad_token_id})도 vocab_size({final_model_vocab_size}) 범위를 벗어납니다!")
    
    print(f"  ✅ Vocab 크기 완전 일치 확인!")
    print(f"  📌 최종 vocab_size: {final_model_vocab_size} (모델과 토크나이저 완전 일치)")
    print(f"  📌 pad_token_id: {pad_token_id} (유효 범위 내)\n")
    
    # 실제 사용할 vocab_size는 모델의 vocab_size (리사이즈 후)
    safe_vocab_size = final_model_vocab_size
    
    # GPU 메모리 정리 (모델 이동 전)
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass  # GPU 정리 실패해도 계속 진행
    
    # 모델을 디바이스로 이동 (리사이즈 완료 후 GPU로 이동)
    # 리사이즈는 CPU에서 완료했으므로 이제 GPU로 이동 가능
    print(f"  - 모델을 {device.upper()}로 이동 중...")
    try:
        # 모델을 GPU로 이동하기 전에 한 번 더 확인
        if device == "cuda":
            # GPU가 정상인지 확인
            if not torch.cuda.is_available():
                raise RuntimeError("GPU가 사용 불가능합니다.")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # 모델 이동
            model = model.to(device)
            
            # GPU 사용 성공 시 GPU 정보 출력
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_props = torch.cuda.get_device_properties(0)
                print(f"  - GPU 이름: {gpu_name}")
                print(f"  - GPU 메모리: {gpu_props.total_memory / (1024**3):.1f}GB")
            except:
                pass  # GPU 정보 출력 실패해도 계속 진행
        else:
            # CPU 모드
            model = model.to(device)
        
        print(f"✅ 모델 로딩 완료\n")
    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg.lower() or "device-side assert" in error_msg.lower():
            # GPU 오류 발생 시 GPU 메모리 정리 시도
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
            
            if GPU_ONLY and not FORCE_CPU:
                # GPU 오류 발생 시 더 자세한 안내
                raise RuntimeError(
                    f"❌ GPU로 모델 이동 실패: {error_msg}\n"
                    "\n"
                    "💡 CUDA란?\n"
                    "   - CUDA는 NVIDIA GPU를 사용하기 위한 플랫폼입니다\n"
                    "   - Colab에서 GPU 런타임을 선택하면 자동으로 CUDA가 사용됩니다\n"
                    "   - GPU를 사용하지 않으려면 CPU 런타임을 선택하세요\n"
                    "\n"
                    "🔧 해결 방법:\n"
                    "   방법 1 (권장): Colab 런타임 재시작\n"
                    "      런타임 → 런타임 다시 시작\n"
                    "      (GPU 상태를 초기화하여 오류를 해결)\n"
                    "\n"
                    "   방법 2: CPU 모드로 학습 (느리지만 안정적)\n"
                    "      코드 상단에서 다음을 변경:\n"
                    "      FORCE_CPU = True  # False → True로 변경\n"
                    "\n"
                    "   방법 3: GPU 전용 모드 해제\n"
                    "      코드 상단에서 다음을 변경:\n"
                    "      GPU_ONLY = False  # True → False로 변경\n"
                    "      (GPU 오류 시 자동으로 CPU로 전환)\n"
                    "\n"
                    "   방법 4: CPU 런타임 사용\n"
                    "      런타임 → 런타임 유형 변경 → 하드웨어 가속기: None\n"
                    "      (GPU를 사용하지 않으므로 CUDA 오류가 발생하지 않음)"
                )
            else:
                print(f"  ⚠️ GPU로 모델 이동 실패: {error_msg}")
                print(f"  💡 CPU 모드로 자동 전환합니다.")
                device = "cpu"
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                model = model.cpu()  # 이미 CPU에 있을 수 있지만 명시적으로 설정
                print(f"✅ 모델 로딩 완료 (CPU 모드)\n")
        else:
            raise  # 다른 오류는 그대로 전파
    
    # 학습 데이터 준비
    print(f"[2/5] 학습 데이터 준비 중...")
    training_examples = prepare_training_data(train_data, tokenizer)
    print(f"✅ {len(training_examples)}개 학습 예제 준비 완료\n")
    
    if len(training_examples) == 0:
        raise ValueError("학습 데이터가 없습니다. poem 필드가 있는 데이터가 필요합니다.")
    
    # 데이터셋 생성
    print(f"[3/5] 데이터셋 변환 중...")
    def tokenize_function(examples):
        # 토크나이즈 (padding은 DataCollator에서 처리)
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=False  # padding을 False로 변경 - DataCollator에서 처리
        )
        
        return tokenized
    
    dataset = Dataset.from_list(training_examples)
    # remove_columns에서 실제 존재하는 컬럼만 제거
    columns_to_remove = ['text', 'prose', 'poem']
    # 존재하는 컬럼만 필터링
    existing_columns = set(dataset.column_names)
    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
    
    # 토큰화된 데이터 검증 및 수정 (더 철저하게)
    print(f"  - 토큰화된 데이터 검증 중...")
    # 모델의 vocab_size를 사용 (리사이즈 후의 실제 크기)
    vocab_size = model.config.vocab_size  # 모델 기준 vocab_size 사용
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"  - Vocab size (모델 기준): {vocab_size}")
    print(f"  - Pad token ID: {pad_token_id}")
    print(f"  - EOS token ID: {tokenizer.eos_token_id}")
    
    # 안전성 확인: pad_token_id가 vocab_size 범위 내인지 확인
    if pad_token_id >= vocab_size or pad_token_id < 0:
        print(f"  ⚠️ 경고: pad_token_id({pad_token_id})가 vocab_size({vocab_size}) 범위를 벗어납니다!")
        print(f"  🔧 eos_token_id로 대체합니다...")
        pad_token_id = tokenizer.eos_token_id
        if pad_token_id >= vocab_size or pad_token_id < 0:
            raise ValueError(f"eos_token_id({pad_token_id})도 vocab_size({vocab_size}) 범위를 벗어납니다!")
    
    # 잘못된 토큰 ID 수정 함수 (더 안전하게)
    def fix_token_ids(example):
        if 'input_ids' in example:
            ids = example['input_ids']
            # 리스트를 numpy 배열로 변환
            ids_array = np.array(ids, dtype=np.int64)
            
            # 음수나 vocab_size를 초과하는 값 수정
            # pad_token_id로 대체 (더 안전)
            invalid_mask = (ids_array < 0) | (ids_array >= vocab_size)
            if np.any(invalid_mask):
                ids_array[invalid_mask] = pad_token_id
            
            example['input_ids'] = ids_array.tolist()
        
        # attention_mask도 확인
        if 'attention_mask' in example:
            mask = np.array(example['attention_mask'], dtype=np.int64)
            # attention_mask는 0 또는 1만 가능
            mask = np.clip(mask, 0, 1)
            example['attention_mask'] = mask.tolist()
        
        return example
    
    # 모든 예제에 대해 토큰 ID 검증 및 수정
    print(f"  - 토큰 ID 검증 및 수정 중...")
    tokenized_dataset = tokenized_dataset.map(fix_token_ids, desc="토큰 ID 검증")
    
    # 전체 데이터셋 검증 (모든 샘플 검사 - 100개 제한 제거)
    print(f"  - 전체 데이터셋 최종 검증 중... (총 {len(tokenized_dataset)}개 샘플)")
    invalid_count = 0
    total_invalid_tokens = 0
    
    # 모든 샘플 검사 (100개 제한 제거)
    for i in range(len(tokenized_dataset)):
        sample = tokenized_dataset[i]
        if 'input_ids' in sample:
            input_ids = sample['input_ids']
            if input_ids:
                ids_array = np.array(input_ids, dtype=np.int64)
                invalid_mask = (ids_array < 0) | (ids_array >= vocab_size)
                
                if np.any(invalid_mask):
                    invalid_count += 1
                    invalid_token_count = np.sum(invalid_mask)
                    total_invalid_tokens += invalid_token_count
                    
                    # 문제가 있는 샘플 수정
                    ids_array[invalid_mask] = pad_token_id
                    tokenized_dataset[i]['input_ids'] = ids_array.tolist()
        
        # 진행 상황 출력 (100개마다)
        if (i + 1) % 100 == 0 or (i + 1) == len(tokenized_dataset):
            print(f"    진행 중: {i + 1}/{len(tokenized_dataset)} (잘못된 샘플: {invalid_count}개)")
    
    if invalid_count > 0:
        print(f"  ⚠️ {invalid_count}개 샘플에서 잘못된 토큰 ID 발견 및 수정 (총 {total_invalid_tokens}개 토큰)")
    else:
        print(f"  ✅ 모든 샘플 검증 완료 (잘못된 토큰 없음)")
    
    # 최종 샘플 검증 (여러 샘플 확인)
    if len(tokenized_dataset) > 0:
        print(f"  - 최종 샘플 검증 중...")
        sample_count = min(10, len(tokenized_dataset))  # 최대 10개 샘플 확인
        all_valid = True
        
        for i in range(sample_count):
            sample = tokenized_dataset[i]
            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                if input_ids:
                    ids_array = np.array(input_ids, dtype=np.int64)
                    max_id = np.max(ids_array) if len(ids_array) > 0 else 0
                    min_id = np.min(ids_array) if len(ids_array) > 0 else 0
                    
                    if max_id >= vocab_size or min_id < 0:
                        all_valid = False
                        print(f"  ⚠️ 샘플 {i}: 범위 초과 (min: {min_id}, max: {max_id}, vocab_size: {vocab_size})")
                        # 강제로 수정
                        ids_array = np.clip(ids_array, 0, vocab_size - 1)
                        invalid_mask = (ids_array < 0) | (ids_array >= vocab_size)
                        if np.any(invalid_mask):
                            ids_array[invalid_mask] = pad_token_id
                        tokenized_dataset[i]['input_ids'] = ids_array.tolist()
        
        if all_valid:
            print(f"  ✅ 최종 검증 완료: 모든 샘플이 유효한 범위 내에 있습니다")
        else:
            print(f"  ⚠️ 일부 샘플을 수정했습니다. 다시 검증합니다...")
            # 한 번 더 전체 검증
            for i in range(len(tokenized_dataset)):
                sample = tokenized_dataset[i]
                if 'input_ids' in sample:
                    input_ids = sample['input_ids']
                    if input_ids:
                        ids_array = np.array(input_ids, dtype=np.int64)
                        ids_array = np.clip(ids_array, 0, vocab_size - 1)
                        invalid_mask = (ids_array < 0) | (ids_array >= vocab_size)
                        if np.any(invalid_mask):
                            ids_array[invalid_mask] = pad_token_id
                        tokenized_dataset[i]['input_ids'] = ids_array.tolist()
            print(f"  ✅ 전체 데이터셋 재검증 완료")
    
    print(f"✅ 데이터셋 변환 완료: {len(tokenized_dataset)}개\n")
    
    # Data Collator 설정 (안전한 커스텀 버전)
    class SafeDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
        """토큰 ID 검증이 포함된 안전한 Data Collator"""
        def __init__(self, tokenizer, mlm=False, model_vocab_size=None):
            super().__init__(tokenizer=tokenizer, mlm=mlm)
            # 모델의 vocab_size를 사용 (토크나이저의 vocab_size가 아닌)
            if model_vocab_size is None:
                raise ValueError("model_vocab_size는 반드시 제공되어야 합니다.")
            self.model_vocab_size = model_vocab_size
            self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            # pad_token_id가 vocab_size 범위 내인지 확인
            if self.pad_token_id >= self.model_vocab_size or self.pad_token_id < 0:
                print(f"  ⚠️ 경고: pad_token_id({self.pad_token_id})가 vocab_size({self.model_vocab_size}) 범위를 벗어납니다!")
                self.pad_token_id = tokenizer.eos_token_id
                if self.pad_token_id >= self.model_vocab_size or self.pad_token_id < 0:
                    raise ValueError(f"eos_token_id({self.pad_token_id})도 vocab_size({self.model_vocab_size}) 범위를 벗어납니다!")
        
        def __call__(self, features):
            # 기본 collator 호출 전에 토큰 ID 검증 (매우 강력하게 - 모든 토큰을 무조건 클리핑)
            vocab_size = self.model_vocab_size  # 모델 기준 vocab_size 사용
            
            for feature in features:
                if 'input_ids' in feature:
                    ids = feature['input_ids']
                    if isinstance(ids, list):
                        ids_array = np.array(ids, dtype=np.int64)
                        # 모든 토큰을 무조건 범위로 클리핑 (안전하게)
                        ids_array = np.clip(ids_array, 0, vocab_size - 1)
                        # 여전히 범위를 벗어나는 토큰이 있으면 pad_token_id로 교체
                        invalid_mask = (ids_array < 0) | (ids_array >= vocab_size)
                        if np.any(invalid_mask):
                            ids_array[invalid_mask] = self.pad_token_id
                        feature['input_ids'] = ids_array.tolist()
                    elif isinstance(ids, torch.Tensor):
                        # Tensor인 경우도 처리
                        ids_array = ids.cpu().numpy() if ids.is_cuda else ids.numpy()
                        # 모든 토큰을 무조건 범위로 클리핑 (안전하게)
                        ids_array = np.clip(ids_array, 0, vocab_size - 1)
                        # 여전히 범위를 벗어나는 토큰이 있으면 pad_token_id로 교체
                        invalid_mask = (ids_array < 0) | (ids_array >= vocab_size)
                        if np.any(invalid_mask):
                            ids_array[invalid_mask] = self.pad_token_id
                        feature['input_ids'] = torch.tensor(ids_array, dtype=ids.dtype, device=ids.device)
            
            # 기본 collator 호출
            try:
                result = super().__call__(features)
                
                # collator 호출 후에도 결과 검증 (매우 강력하게)
                if isinstance(result, dict) and 'input_ids' in result:
                    result_ids = result['input_ids']
                    if isinstance(result_ids, torch.Tensor):
                        # GPU에 있을 수 있으므로 CPU로 이동해서 검증
                        result_ids_cpu = result_ids.cpu().numpy()
                        # 모든 토큰을 먼저 범위로 클리핑 (안전하게)
                        result_ids_cpu = np.clip(result_ids_cpu, 0, vocab_size - 1)
                        # 여전히 범위를 벗어나는 토큰이 있으면 pad_token_id로 교체
                        invalid_mask = (result_ids_cpu < 0) | (result_ids_cpu >= vocab_size)
                        if np.any(invalid_mask):
                            result_ids_cpu[invalid_mask] = self.pad_token_id
                        result['input_ids'] = torch.tensor(result_ids_cpu, dtype=result_ids.dtype, device=result_ids.device)
                
                return result
            except Exception as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "device-side assert" in error_msg.lower():
                    print(f"  ⚠️ DataCollator에서 CUDA 오류 발생: {error_msg}")
                    print(f"  💡 토큰 ID 검증을 다시 수행합니다...")
                    # 모든 feature를 다시 검증
                    for feature in features:
                        if 'input_ids' in feature:
                            ids = feature['input_ids']
                            if isinstance(ids, (list, torch.Tensor)):
                                if isinstance(ids, torch.Tensor):
                                    ids = ids.cpu().numpy()
                                else:
                                    ids = np.array(ids)
                                # 강제로 모든 토큰을 범위 내로 클리핑
                                ids = np.clip(ids, 0, vocab_size - 1)
                                if isinstance(feature['input_ids'], torch.Tensor):
                                    feature['input_ids'] = torch.tensor(ids, dtype=feature['input_ids'].dtype, device=feature['input_ids'].device)
                                else:
                                    feature['input_ids'] = ids.tolist()
                    # 다시 시도
                    return super().__call__(features)
                else:
                    raise
    
    data_collator = SafeDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM이므로 False
        model_vocab_size=model.config.vocab_size  # 모델의 vocab_size 전달
    )
    
    # 학습 스텝 수 계산
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = len(tokenized_dataset) // effective_batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    total_steps = steps_per_epoch * epochs
    
    if len(tokenized_dataset) == 0:
        raise ValueError("❌ 학습 데이터셋이 비어있습니다.")
    
    if len(tokenized_dataset) < effective_batch_size:
        print(f"\n⚠️ 경고: 데이터셋 크기({len(tokenized_dataset)})가 유효 배치 크기({effective_batch_size})보다 작습니다.")
    
    # 학습 인자 설정
    # output_dir이 None이거나 빈 문자열인지 확인
    if not output_dir:
        raise ValueError(f"output_dir이 유효하지 않습니다: {output_dir}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_output_dir = f"{output_dir}_{timestamp}"
    
    # 경로 유효성 확인
    if not model_output_dir:
        raise ValueError(f"model_output_dir이 생성되지 않았습니다. output_dir={output_dir}")
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=min(100, max(1, total_steps // 10)),
        logging_steps=1,
        save_steps=max(10, total_steps // 5),
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        # FP16 관련 설정 수정: gradient scaling 문제 해결
        fp16=False,  # FP16 비활성화 (안정성 우선)
        bf16=False,  # bfloat16도 비활성화
        dataloader_pin_memory=(device == "cuda"),
        # gradient clipping 완전히 비활성화 (FP16 문제 해결)
        max_grad_norm=None,  # None으로 설정하여 gradient clipping 비활성화
        report_to="none",
        # Accelerate 설정 명시적으로 지정
        dataloader_num_workers=0,  # 멀티프로세싱 비활성화 (안정성)
    )
    
    print(f"[학습 설정]")
    print(f"  - 디바이스: {device}")
    print(f"  - 데이터셋 크기: {len(tokenized_dataset)}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 그래디언트 누적: {gradient_accumulation_steps}")
    print(f"  - 에포크: {epochs}")
    print(f"  - 유효 배치 크기: {effective_batch_size}")
    print(f"  - 스텝/에포크: {steps_per_epoch}")
    print(f"  - 예상 총 스텝: {total_steps}")
    
    # 학습 시작 전 최종 데이터 검증 (CUDA 오류 방지)
    print(f"\n[4/5] Trainer 설정 및 최종 검증 중...")
    
    # 최종 데이터 검증: 모든 샘플의 토큰 ID가 유효한지 확인
    print(f"  - 학습 시작 전 최종 데이터 검증 중...")
    final_invalid_count = 0
    vocab_size = model.config.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # pad_token_id가 유효한지 확인
    if pad_token_id >= vocab_size or pad_token_id < 0:
        pad_token_id = tokenizer.eos_token_id
        if pad_token_id >= vocab_size or pad_token_id < 0:
            raise ValueError(f"pad_token_id와 eos_token_id 모두 vocab_size 범위를 벗어납니다!")
    
    for i in range(len(tokenized_dataset)):
        sample = tokenized_dataset[i]
        if 'input_ids' in sample:
            input_ids = sample['input_ids']
            if input_ids:
                ids_array = np.array(input_ids, dtype=np.int64)
                # 강제로 모든 토큰을 범위 내로 클리핑
                ids_array = np.clip(ids_array, 0, vocab_size - 1)
                invalid_mask = (ids_array < 0) | (ids_array >= vocab_size)
                if np.any(invalid_mask):
                    final_invalid_count += 1
                    ids_array[invalid_mask] = pad_token_id
                tokenized_dataset[i]['input_ids'] = ids_array.tolist()
    
    if final_invalid_count > 0:
        print(f"  ⚠️ 최종 검증에서 {final_invalid_count}개 샘플 수정")
    else:
        print(f"  ✅ 최종 검증 완료: 모든 샘플이 유효합니다")
    
    # Accelerate mixed precision 명시적으로 비활성화 (FP16 문제 방지)
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    print(f"✅ Trainer 설정 완료")
    print(f"  - FP16: {training_args.fp16}")
    print(f"  - Max Grad Norm: {training_args.max_grad_norm}")
    print(f"  - Vocab Size: {vocab_size}")
    print(f"  - Pad Token ID: {pad_token_id}")
    print()
    
    # 학습 시작
    print(f"[5/5] 학습 시작...")
    print(f"{'='*80}")
    print(f"📊 학습 정보:")
    print(f"   - 데이터셋: {len(tokenized_dataset)}개")
    print(f"   - 총 스텝: {total_steps} (에포크 {epochs} × 스텝/에포크 {steps_per_epoch})")
    print(f"   - 디바이스: {device}")
    print(f"{'='*80}\n")
    
    train_result = None
    try:
        train_result = trainer.train()
        print(f"{'='*80}\n")
    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*80}")
        print(f"❌ 학습 중 오류 발생: {error_msg}")
        print(f"{'='*80}")
        
        # CUDA 오류인지 확인
        is_cuda_error = "CUDA" in error_msg or "cuda" in error_msg.lower() or "device-side assert" in error_msg.lower()
        
        if is_cuda_error:
            if GPU_ONLY and not FORCE_CPU:
                # GPU 전용 모드에서는 오류 발생 시 중단
                raise RuntimeError(
                    f"❌ CUDA 오류가 발생했습니다: {error_msg}\n"
                    "💡 GPU 전용 모드(GPU_ONLY=True)이므로 학습을 중단합니다.\n"
                    "🔧 해결 방법:\n"
                    "   1. Colab 런타임 재시작 (가장 확실한 방법):\n"
                    "      런타임 → 런타임 다시 시작\n"
                    "   2. CPU 모드로 학습하려면 코드 상단에서 FORCE_CPU = True로 설정하세요."
                )
            else:
                print(f"\n💡 CUDA 오류가 발생했습니다. GPU가 오염되었을 수 있습니다.")
                print(f"🔧 해결 방법:")
                print(f"   1. Colab 런타임 재시작 (가장 확실한 방법):")
                print(f"      런타임 → 런타임 다시 시작")
                print(f"   2. 코드 상단에서 FORCE_CPU = True로 설정 후 재실행")
                print(f"      (학습은 느리지만 안정적으로 진행됩니다)")
                print(f"   3. 다음 fold부터 CPU 모드로 자동 전환됩니다 (매우 느림)")
                print(f"\n⚠️ GPU를 사용하지 않도록 설정합니다...")
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        import traceback
        traceback.print_exc()
        
        # GPU 메모리 정리 - 안전하게 처리 (오류 무시)
        if torch.cuda.is_available():
            try:
                if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as cleanup_error:
                # GPU 정리 오류는 무시하고 계속 진행
                pass
        
        # 학습 실패 시 None 반환
        return None
    
    # 모델 저장 (학습 성공 시에만)
    if train_result is None:
        print(f"⚠️ 학습이 실패하여 모델을 저장하지 않습니다.")
        return None
    
    if not model_output_dir:
        print(f"⚠️ 모델 저장 경로가 설정되지 않았습니다.")
        return None
    
    # 경로 최종 확인
    if not model_output_dir or not isinstance(model_output_dir, (str, Path)):
        error_msg = f"모델 저장 경로가 유효하지 않습니다: {model_output_dir} (type: {type(model_output_dir)})"
        print(f"❌ {error_msg}")
        return None
    
    # 경로를 문자열로 변환 (Path 객체인 경우)
    model_output_dir_str = str(model_output_dir)
    
    try:
        print(f"모델 저장 중: {model_output_dir_str}")
        
        # trainer.save_model()에 명시적으로 경로 지정 (안전하게)
        # training_args.output_dir이 None이어도 명시적 경로로 저장 가능
        trainer.save_model(output_dir=model_output_dir_str)
        
        # tokenizer 저장
        tokenizer.save_pretrained(model_output_dir_str)
        
        print(f"✅ 모델 저장 완료: {model_output_dir_str}\n")
    except Exception as save_error:
        error_msg = str(save_error)
        print(f"⚠️ 모델 저장 중 오류: {error_msg}")
        
        # 경로 관련 오류인지 확인
        if "NoneType" in error_msg or "PathLike" in error_msg:
            print(f"  ❌ 경로 오류 상세:")
            print(f"     - model_output_dir: {model_output_dir} (type: {type(model_output_dir)})")
            print(f"     - model_output_dir_str: {model_output_dir_str}")
            print(f"     - training_args.output_dir: {getattr(training_args, 'output_dir', 'N/A')}")
            print(f"     - trainer.args.output_dir: {getattr(trainer.args, 'output_dir', 'N/A') if hasattr(trainer, 'args') else 'N/A'}")
        
        # 저장 실패해도 경로는 반환 (부분 저장 가능)
        if model_output_dir_str and os.path.exists(model_output_dir_str):
            print(f"  💡 일부 파일은 저장되었을 수 있습니다: {model_output_dir_str}")
        else:
            print(f"  ❌ 모델 저장 실패: 경로가 유효하지 않습니다.")
            return None
    
    # 학습 결과 출력
    print(f"[학습 완료]")
    if train_result:
        print(f"  - 학습 손실: {train_result.training_loss:.4f}")
        print(f"  - 총 학습 시간: {train_result.metrics.get('train_runtime', 0):.2f}초")
    print(f"  - 모델 저장 위치: {model_output_dir}")
    
    return model_output_dir


def run_kfold_training(data: List[Dict], k_folds: int = 5) -> List[str]:
    """
    k-fold 교차 검증으로 모델을 학습시킵니다.
    
    Args:
        data: 학습 데이터 리스트
        k_folds: fold 개수
    
    Returns:
        각 fold의 학습된 모델 경로 리스트
    """
    print(f"\n{'='*80}")
    print(f"[k-fold 교차 검증 시작]")
    print(f"  - 데이터 개수: {len(data)}")
    print(f"  - Fold 개수: {k_folds}")
    print(f"{'='*80}\n")
    
    if len(data) < k_folds:
        raise ValueError(f"데이터 개수({len(data)})가 fold 개수({k_folds})보다 적습니다.")
    
    # 초기 GPU 상태 확인 (Colab 환경 고려)
    # GPU 우선 사용, 오류 발생 시에만 CPU로 전환
    gpu_available = False
    gpu_quota_exceeded = False
    
    # CPU 모드로 강제 실행하는 경우만 CPU 사용
    if FORCE_CPU:
        print(f"\n{'='*80}")
        print(f"💡 CPU 모드로 강제 실행합니다 (FORCE_CPU=True)")
        print(f"{'='*80}\n")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gpu_available = False
    elif not torch.cuda.is_available():
        # GPU가 감지되지 않는 경우
        if GPU_ONLY:
            raise RuntimeError(
                "❌ GPU가 감지되지 않습니다. GPU 전용 모드(GPU_ONLY=True)이므로 학습을 중단합니다.\n"
                "💡 해결 방법:\n"
                "   1. Colab에서 GPU 런타임 선택: 런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU\n"
                "   2. GPU 할당량이 소진되었을 수 있습니다. 몇 시간 후 다시 시도하세요.\n"
                "   3. CPU 모드로 학습하려면 코드 상단에서 FORCE_CPU = True로 설정하세요."
            )
        else:
            print(f"\n⚠️ GPU가 감지되지 않습니다.")
            print(f"💡 CPU 모드로 자동 전환합니다...\n")
            gpu_available = False
            gpu_quota_exceeded = True
    else:
        # GPU 사용 가능
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                print(f"✅ GPU 사용 가능 (디바이스 {device_count}개 감지)")
                print(f"💡 GPU 모드로 학습을 진행합니다.")
                gpu_available = True
            else:
                if GPU_ONLY:
                    raise RuntimeError("GPU 디바이스가 없습니다. GPU 전용 모드이므로 학습을 중단합니다.")
                else:
                    gpu_available = False
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
        except Exception as e:
            if GPU_ONLY:
                raise RuntimeError(
                    f"❌ GPU 확인 실패: {e}\n"
                    "💡 GPU 전용 모드(GPU_ONLY=True)이므로 학습을 중단합니다.\n"
                    "🔧 해결 방법:\n"
                    "   1. Colab 런타임 재시작: 런타임 → 런타임 다시 시작\n"
                    "   2. CPU 모드로 학습하려면 코드 상단에서 FORCE_CPU = True로 설정하세요."
                )
            else:
                print(f"⚠️ GPU 확인 실패: {e}")
                print(f"💡 CPU 모드로 자동 전환합니다.\n")
                gpu_available = False
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    model_paths = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(data), 1):
        print(f"\n{'='*80}")
        print(f"[Fold {fold_idx}/{k_folds}]")
        print(f"  - Train: {len(train_indices)}개")
        print(f"  - Test: {len(test_indices)}개")
        print(f"{'='*80}\n")
        
        fold_start_time = time.time()
        
        # 각 fold 전에 GPU 상태 재확인 (GPU 전용 모드)
        if gpu_available and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            if not torch.cuda.is_available():
                if GPU_ONLY:
                    raise RuntimeError(
                        f"❌ Fold {fold_idx}: GPU가 더 이상 사용 불가능합니다.\n"
                        "💡 GPU 전용 모드(GPU_ONLY=True)이므로 학습을 중단합니다.\n"
                        "🔧 해결 방법: Colab 런타임 재시작"
                    )
                else:
                    print(f"\n⚠️ Fold {fold_idx}: GPU가 더 이상 사용 불가능합니다. CPU 모드로 전환합니다.")
                    gpu_available = False
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
            else:
                # GPU 메모리만 정리 (간단한 연산은 피함)
                try:
                    torch.cuda.empty_cache()
                except:
                    pass  # GPU 정리 실패해도 계속 진행
        
        # Train/Test 데이터 분할
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        
        # 모델 학습
        try:
            fold_output_dir = f"{OUTPUT_DIR}_fold{fold_idx}"
            model_path = train_kogpt2_model(
                train_data=train_data,
                output_dir=fold_output_dir,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
            )
            model_paths.append(model_path)
            
            fold_time = time.time() - fold_start_time
            print(f"\n✅ [Fold {fold_idx} 완료]")
            print(f"  - 모델 저장 위치: {model_path}")
            print(f"  - 소요 시간: {fold_time:.2f}초")
            
            # GPU 메모리 정리 (다음 fold를 위해)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"  - GPU 메모리 정리 완료")
                except Exception as e:
                    print(f"  ⚠️ GPU 메모리 정리 실패 (무시): {e}")
        
        except Exception as e:
            print(f"\n❌ [Fold {fold_idx} 실패]: {e}")
            import traceback
            traceback.print_exc()
            model_paths.append(None)
            
            # GPU 메모리 정리 (오류 후에도) - 안전하게 처리
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # GPU 상태 초기화 시도
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except:
                        pass
                except Exception as cleanup_error:
                    print(f"  ⚠️ GPU 정리 중 오류 (무시): {cleanup_error}")
                    print(f"  💡 GPU가 오염되었습니다. 다음 fold부터 CPU 모드로 자동 전환합니다.")
                    print(f"     💡 더 빠른 학습을 원하시면 Colab 런타임을 재시작하세요:")
                    print(f"        런타임 → 런타임 다시 시작")
                    # 다음 fold부터 GPU 사용 안 하도록 설정
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    gpu_available = False
    
    return model_paths


def main():
    """메인 함수"""
    print(f"\n{'='*80}")
    print("Google Colab에서 koGPT2 모델 k-fold 교차 검증 학습")
    print(f"{'='*80}\n")
    
    # GPU 확인 (Colab 환경) - GPU 우선 사용
    if FORCE_CPU:
        print("💡 CPU 모드로 강제 실행합니다 (FORCE_CPU=True)")
        print("   CUDA 오류를 피하고 안정적으로 학습할 수 있습니다.")
        print("   ⚠️ CPU 모드는 매우 느리지만 안정적입니다.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 사용 가능: {gpu_name}")
            print("💡 GPU 모드로 학습을 진행합니다.")
            print("   CUDA 오류가 발생하면 런타임을 재시작하세요: 런타임 → 런타임 다시 시작")
        except Exception as e:
            print(f"⚠️ GPU 확인 중 오류: {e}")
            print("💡 GPU가 오염되었거나 할당량이 소진되었을 수 있습니다.")
            print("   CPU 모드로 자동 전환합니다.")
            print("   💡 GPU를 사용하려면 런타임을 재시작하세요.")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        print("⚠️ GPU가 감지되지 않았습니다.")
        print("\n💡 GPU를 사용하려면:")
        print("   런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU")
        print("\n⚠️ CPU 모드로 실행됩니다 (매우 느림).")
        print("💡 가능한 원인:")
        print("   1. Colab GPU 할당량 소진 (무료 버전은 시간 제한이 있음)")
        print("   2. GPU 런타임이 선택되지 않음")
        print("\n🔧 해결 방법:")
        print("   1. 몇 시간 후 다시 시도 (할당량 복구 대기)")
        print("   2. Colab Pro/Pro+ 구독")
        print("   3. CPU 모드로 계속 진행 (매우 느림)")
    
    # 데이터 다운로드
    try:
        data = download_kpoem_data(max_size=MAX_DATA_SIZE)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    if len(data) == 0:
        print("❌ 데이터가 없습니다.")
        return
    
    # k-fold 교차 검증 학습
    try:
        model_paths = run_kfold_training(data, k_folds=K_FOLDS)
        
        print(f"\n{'='*80}")
        print("✅ k-fold 교차 검증 학습 완료!")
        print(f"{'='*80}")
        print(f"\n📦 학습된 모델 위치:")
        for fold_idx, model_path in enumerate(model_paths, 1):
            if model_path:
                print(f"  - Fold {fold_idx}: {model_path}")
            else:
                print(f"  - Fold {fold_idx}: 실패")
        
        print(f"\n💡 다음 단계:")
        print(f"1. Google Drive에 업로드:")
        print(f"   - 각 fold의 모델 폴더를 Google Drive에 업로드")
        print(f"2. 또는 로컬로 다운로드:")
        print(f"   - Colab에서 파일 다운로드 기능 사용")
        print(f"   - 또는 zip으로 압축 후 다운로드")
        print(f"\n📝 로컬에서 사용 방법:")
        print(f"   - 학습된 모델을 'backend/trained_models/' 폴더에 복사")
        print(f"   - kfold_poem_generation.py에서 사용")
        print(f"\n💡 모든 fold 모델을 한 번에 다운로드:")
        print(f"   - 아래 코드를 실행하여 zip으로 압축:")
        print(f"   ```python")
        print(f"   import shutil")
        print(f"   from google.colab import files")
        print(f"   shutil.make_archive('all_folds_models', 'zip', '{OUTPUT_DIR}')")
        print(f"   files.download('all_folds_models.zip')")
        print(f"   ```")
        
    except Exception as e:
        print(f"❌ 학습 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


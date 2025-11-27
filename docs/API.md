# API & 환경 설정 가이드

시옷 백엔드/프론트엔드가 정상 동작하려면 API 엔드포인트와 외부 서비스(Google Cloud Translation, Gemini 등) 설정을 정확히 맞춰야 합니다. 이 문서는 README에 흩어져 있던 환경 변수와 API 관련 내용을 한 곳에 정리한 것입니다.

## 1. 백엔드 환경 변수 (.env)

`backend/.env` 파일을 생성하고 다음 항목을 필요에 따라 채웁니다.

```bash
# 모델 선택 (미설정 시 GPU 감지로 자동 선택)
POEM_MODEL_TYPE=kogpt2   # 또는 solar

# Google Cloud Translation
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
# 또는
GOOGLE_TRANSLATION_API_KEY=your-translation-api-key

# Gemini (감정 스토리/시 개선)
GEMINI_API_KEY=your-gemini-api-key

# Colab에서 ngrok 토큰을 사용할 경우
NGROK_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxx
```

> 💡 로컬 개발 시 CPU만 사용할 계획이라면 `POEM_MODEL_TYPE=kogpt2`를 미리 지정해 두면 매번 자동 감지를 기다리지 않아도 됩니다.

## 2. 프론트엔드 환경 변수 (`frontend/.env`)

```bash
# 로컬 FastAPI 백엔드
VITE_API_URL=http://localhost:8000/api/poem/generate

# SOLAR (Colab) 백엔드 URL
VITE_COLAB_API_URL=https://<your-ngrok>.ngrok-free.dev
```

프론트엔드 `PoemGeneration` 페이지는 모델 선택에 따라 위 값 중 하나를 사용합니다. SOLAR를 선택하면 **반드시** ngrok URL이 필요하며, 로컬 URL을 넣으면 거부합니다.

## 3. Google Cloud Translation API 설정

1. [Google Cloud Console](https://console.cloud.google.com/) 접속 → 프로젝트 생성/선택  
2. “APIs & Services → Library”에서 **Cloud Translation API v3**를 활성화  
3. 인증 방식 선택
   - **ADC**: 로컬 터미널에서 `gcloud auth application-default login` 실행 후 `GOOGLE_CLOUD_PROJECT_ID` 설정
   - **서비스 계정**: 키 JSON을 만들고 `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` 지정
4. (선택) API 키 발급 → `GOOGLE_TRANSLATION_API_KEY`에 넣으면 간단한 테스트가 가능  
5. Colab에서는 키 파일을 `/content/key.json`에 업로드한 뒤 환경 변수로 연결합니다.

번역 설정이 누락되면 한국어가 아닌 시를 생성해도 번역 단계가 건너뛰어지며, 로그에 경고가 표시됩니다.

## 4. Gemini API (감정 스토리, 시 개선)

1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 발급  
2. `GEMINI_API_KEY` 환경 변수에 저장  
3. FastAPI 서버에서 감정 요약(`analyze_emotions_cutely`)이나 Gemini 시 개선(`improve_poem_with_gemini`) 호출 시 자동 사용  

> 무료 티어라도 일일 호출 제한이 있으므로, 다량 테스트 시 quota에 유의하세요.

## 5. API 엔드포인트 요약

| 엔드포인트 | 메서드 | 설명 | 비고 |
|------------|--------|------|------|
| `/health` | GET | 서버/모델 상태 확인 | 모델 ID, GPU 여부, has_trained_model 표시 |
| `/api/poem/generate` | POST | 시 생성 요청 | `PoemRequest` (text, mood, lines, model_type 등) |
| `/api/emotion/analyze-cute` | POST | 감정 데이터 요약 | Gemini 기반 감정 스토리 생성 |

주요 요청 예시는 README “API 문서” 섹션의 cURL 스니펫을 참고하세요.

## 6. Colab + ngrok 연동 체크리스트

1. `GPU_backend.ipynb`를 실행해 `/backend` 디렉토리에서 `uvicorn`을 띄웁니다.  
2. ngrok 토큰 설정 → `ngrok.connect(8000)`으로 public URL 획득  
3. 프론트엔드 `.env`의 `VITE_COLAB_API_URL`을 해당 URL로 업데이트하고 `npm run dev` 재시작  
4. health 체크 (`curl <ngrok-url>/health`) 후 프론트엔드에서 SOLAR 모델을 선택합니다.

## 7. API 테스트 명령어 모음

### Health 체크
```bash
curl -H "ngrok-skip-browser-warning: true" https://<ngrok-url>/health
```

### 시 생성 (로컬 예시)
```bash
curl -X POST http://localhost:8000/api/poem/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "오늘 하루는 힘들었지만 친구 덕분에 웃을 수 있었다.",
        "lines": 4,
        "mood": "잔잔한",
        "model_type": "kogpt2"
      }'
```

### 감정 스토리 생성
```bash
curl -X POST http://localhost:8000/api/emotion/analyze-cute \
  -H "Content-Type: application/json" \
  -d '{
        "poems": [
          {"emotion": "기쁨", "createdAt": "2024-01-15T10:30:00Z"},
          {"emotion": "슬픔", "createdAt": "2024-01-16T14:20:00Z"}
        ]
      }'
```

## 8. 문제 해결 팁

- **SOLAR 요청이 실패하는 경우**  
  - ngrok URL이 만료되었는지, 브라우저에서 직접 접속해 “Visit site”를 눌렀는지 확인  
  - Colab 세션이 잠들면 uvicorn을 재실행해야 함

- **koGPT2 로컬 추론이 번역 없이 끝나는 경우**  
  - Google Translation 환경 변수가 정확히 설정되었는지 확인  
  - Colab에서는 키 파일을 `/content/key.json` 경로로 업로드했는지 체크

이 문서는 API/환경 설정과 외부 서비스 연동을 한 번에 볼 수 있도록 유지보수하며, README에는 간단한 링크만 남겨 가독성을 확보합니다.



## 상세 API 스펙 (README에서 이동)


### 시 생성 API

**엔드포인트:** `POST /api/poem/generate`

**설명:** 사용자의 일상글을 받아 키워드 추출, 감정 분석, 시 생성을 수행합니다.

**요청 본문:**

```json
{
  "text": "오늘 하루 정말 힘들었어. 하지만 친구들이 많이 응원해줘서 기분이 좋아졌다.",
  "lines": 4,
  "mood": "잔잔한",
  "required_keywords": ["친구", "응원"],
  "banned_words": ["힘들"],
  "use_rhyme": false,
  "acrostic": null,
  "model_type": "solar",
  "use_trained_model": false,
  "use_gemini_improvement": true
}
```

**요청 파라미터:**

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `text` | string | ✅ | - | 시로 변환할 일상글 |
| `lines` | integer | ❌ | 4 | 생성할 시의 줄 수 |
| `mood` | string | ❌ | 자동 감지 | 시의 분위기 (잔잔한/담담한/쓸쓸한) |
| `required_keywords` | array | ❌ | [] | 시에 반드시 포함할 키워드 |
| `banned_words` | array | ❌ | [] | 시에서 사용하지 않을 단어 |
| `use_rhyme` | boolean | ❌ | false | 운율 사용 여부 |
| `acrostic` | string | ❌ | null | 아크로스틱 (예: "사랑해") |
| `model_type` | string | ❌ | 자동 선택 | 사용할 모델 ("solar" 또는 "kogpt2") |
| `use_trained_model` | boolean | ❌ | false | 학습된 모델 사용 여부 |
| `use_gemini_improvement` | boolean | ❌ | true | Gemini로 시 개선 여부 |

**응답 예시:**

```json
{
  "keywords": ["친구", "응원", "기분", "하루"],
  "emotion": "기쁨",
  "emotion_confidence": 0.85,
  "poem": "친구들의 따뜻한 응원\n하루의 힘듦을 잊게 하네\n기분이 좋아지는 순간\n함께하는 소중함 느껴",
  "success": true,
  "message": "시가 성공적으로 생성되었습니다."
}
```

**응답 필드:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `keywords` | array | 추출된 키워드 목록 |
| `emotion` | string | 감정 분류 결과 (기쁨/슬픔/중립 등) |
| `emotion_confidence` | float | 감정 분류 신뢰도 (0.0 ~ 1.0) |
| `poem` | string | 생성된 시 |
| `success` | boolean | 성공 여부 |
| `message` | string | 응답 메시지 |

**에러 응답:**

```json
{
  "detail": "텍스트가 비어있습니다."
}
```

**cURL 예시:**

```bash
curl -X POST "http://localhost:8000/api/poem/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "오늘 하루 정말 힘들었어",
    "lines": 4,
    "mood": "쓸쓸한"
  }'
```

### 감정 분석 API

**엔드포인트:** `POST /api/emotion/analyze-cute`

**설명:** 생성된 시들의 감정 데이터를 받아 Gemini API로 사용자 친화적인 스토리로 변환합니다.

**요청 본문:**

```json
{
  "poems": [
    {
      "emotion": "기쁨",
      "createdAt": "2024-01-15T10:30:00Z"
    },
    {
      "emotion": "슬픔",
      "createdAt": "2024-01-16T14:20:00Z"
    }
  ]
}
```

**응답 예시:**

```json
{
  "story": "이번 주는 감정 변화가 다양했습니다. 월요일에는 기쁨이 많이 나타났고, 화요일에는 슬픔이 증가했습니다...",
  "summary": "전체적으로 기쁨과 슬픔이 번갈아 나타나는 패턴을 보입니다.",
  "emoji": "😊",
  "message": "오늘도 수고하셨어요!",
  "success": true
}
```

### 헬스 체크 API

**엔드포인트:** `GET /health`

**설명:** 서버 상태 및 모델 정보를 확인합니다.

**응답 예시:**

```json
{
  "ok": true,
  "service": "poem",
  "model_type": "kogpt2",
  "model_id": "skt/kogpt2-base-v2",
  "device": "cpu",
  "has_gpu": false,
  "model": "KOGPT2 (CPU)"
}
```

**인터랙티브 API 문서:**

서버 실행 후 다음 URL에서 Swagger UI를 통해 API를 테스트할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
# Train Recipes 🚀

다양한 프레임워크와 기법을 활용한 대규모 언어 모델(LLM) 파인튜닝 레시피 모음입니다.

## 개요

이 저장소는 다음 기능을 갖춘 LLM 파인튜닝 스크립트와 설정을 제공합니다:
- **LoRA/QLoRA**를 활용한 효율적인 파인튜닝
- **추론 + 대화형** 혼합 데이터셋 학습
- **Unsloth**를 통한 최적화된 학습 속도
- **Wandb** 연동으로 실험 추적

## 디렉토리 구조

```
train-recipes/
├── samples/
│   └── unsloth/
│       └── qwen3/                    # Qwen3-14B 파인튜닝
│           ├── qwen3_(14b)_reasoning_conversational.py  # 학습 모듈
│           ├── chat_console.py       # 대화형 채팅 콘솔
│           ├── train.sh              # 학습 실행 스크립트
│           ├── chat.sh               # 채팅 실행 스크립트
│           └── env_local             # 설정 파일 (git 제외)
├── reasoning/                        # 추론 중심 레시피
└── README.md
```

## 빠른 시작

### 1. 사전 요구사항

- Python 3.10+
- CUDA 11.8+ 및 호환 GPU (24GB+ VRAM 권장)
- [uv](https://docs.astral.sh/uv/) (없으면 자동 설치)

### 2. 환경 설정

```bash
cd samples/unsloth/qwen3/

# setup 스크립트 실행 (uv venv 생성 및 패키지 설치)
./setup.sh

# 또는 특정 Python 버전 지정
./setup.sh -p 3.11
```

### 3. 설정 구성

레시피 디렉토리에 `env_local` 파일 생성:

```bash
cp env_local.example env_local  # 또는 직접 생성
```

`env_local` 파일을 편집하여 설정:

```bash
# API 토큰
HF_TOKEN="your_huggingface_token"
WNB_API_KEY="your_wandb_api_key"

# 모델 설정
MODEL_NAME="unsloth/Qwen3-14B"
MAX_SEQ_LENGTH="32768"
LOAD_IN_4BIT="false"

# LoRA 설정
LORA_R="32"
LORA_ALPHA="32"

# 학습 설정
TRAIN_GPU_IDS="0,1"
TRAIN_BATCH_SIZE="2"
TRAIN_LEARNING_RATE="2e-4"
TRAIN_MAX_STEPS="30"

# 추론 설정
INFER_TEMPERATURE="0.7"
INFER_TOP_P="0.8"
INFER_MAX_TOKENS="16384"
```

### 4. 학습 실행

```bash
# 현재 설정 확인
./train.sh -v

# env_local 설정으로 학습 시작
./train.sh

# 명령줄에서 설정 오버라이드
./train.sh -g 0 -b 4 -s 100 -r 1e-4
```

### 5. 모델 테스트

```bash
# 사용 가능한 학습된 모델 목록
./chat.sh -l

# 대화형 채팅 시작 (최신 모델 자동 탐지)
./chat.sh

# thinking 모드로 시작
./chat.sh -T

# 특정 모델 경로 지정
./chat.sh -m ./logs/qwen3-14b_reasoning-conversational_20260111_143052/lora_model
```

## 학습 스크립트

### `train.sh` 옵션

| 옵션 | 설명 |
|------|------|
| `-g, --gpu IDS` | GPU 장치 ID (예: `0,1`) |
| `-b, --batch N` | 배치 크기 |
| `-r, --lr RATE` | 학습률 |
| `-s, --steps N` | 최대 학습 스텝 |
| `--lora-r N` | LoRA 랭크 |
| `--no-venv` | 가상환경 활성화 건너뛰기 |
| `-l, --logs` | 최근 학습 로그 목록 |
| `-c, --clean` | 오래된 로그 디렉토리 정리 |
| `-v, --vars` | 현재 설정 표시 |

### `chat.sh` 옵션

| 옵션 | 설명 |
|------|------|
| `-m, --model PATH` | LoRA 모델 경로 |
| `-g, --gpu ID` | GPU 장치 ID |
| `-t, --tokens N` | 최대 생성 토큰 수 |
| `-T, --thinking` | thinking 모드 활성화 |
| `--no-venv` | 가상환경 활성화 건너뛰기 |
| `-l, --list` | 사용 가능한 모델 목록 |
| `-v, --vars` | 추론 설정 표시 |

## 채팅 콘솔 명령어

대화형 콘솔 내에서 사용 가능한 명령어:

| 명령어 | 설명 |
|--------|------|
| `/help` | 사용 가능한 명령어 표시 |
| `/thinking` | thinking 모드 토글 |
| `/clear` | 대화 히스토리 초기화 |
| `/mode` | 현재 설정 표시 |
| `/tokens N` | 최대 토큰 수 설정 |
| `/single` | 단일 턴 모드 |
| `/multi` | 멀티 턴 모드 |
| `/exit` | 콘솔 종료 |

## 출력 구조

학습 결과물은 고유한 타임스탬프와 함께 저장됩니다:

```
logs/
└── qwen3-14b_reasoning-conversational_20260111_143052/
    ├── train/
    │   ├── train.log           # 학습 로그
    │   ├── config.txt          # 설정 스냅샷
    │   ├── training_stats.txt  # 최종 통계
    │   └── tensorboard/        # TensorBoard 로그
    ├── eval/
    │   └── dataset_info.txt    # 데이터셋 통계
    ├── checkpoints/            # 학습 체크포인트
    └── lora_model/             # 최종 LoRA 어댑터
```

## 사용 가능한 레시피

### Qwen3-14B 추론 + 대화형

Qwen3-14B를 다음 데이터 혼합으로 파인튜닝:
- **추론 데이터 (75%)**: [OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)
- **대화형 데이터 (25%)**: [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)

특징:
- thinking 및 non-thinking 추론 모드 모두 지원
- 각 모드에 최적화된 생성 파라미터
- 완전한 로깅 및 실험 추적

## 설정 우선순위

설정은 다음 순서로 로드됩니다 (높은 우선순위 먼저):

1. **명령줄 인자**
2. **환경 변수** (`export TRAIN_BATCH_SIZE=4`)
3. **env_local 파일**
4. **코드 내 기본값**

## 라이선스

이 프로젝트는 LGPL-3.0 라이선스 하에 배포됩니다.

## 감사의 글

- [Unsloth](https://github.com/unslothai/unsloth) - 최적화된 학습 제공
- [Hugging Face](https://huggingface.co/) - 모델 호스팅
- [Weights & Biases](https://wandb.ai/) - 실험 추적


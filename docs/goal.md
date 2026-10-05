# 보이스 파인튜닝 프로젝트 목표

> 작성일: 2026-10-04

## 1. 한 줄 요약

**내 목소리(또는 동의받은 화자의 목소리)로 아무 문장이나 읽어주는 TTS 모델을 만든다.**
공개된 TTS 모델을 가져와, 수 분~수십 분 분량의 음성 데이터로 추가 학습(파인튜닝)한다.

## 2. 기본 개념

| 용어 | 뜻 |
|---|---|
| **TTS** (Text-to-Speech) | 텍스트를 넣으면 음성을 만들어주는 모델 |
| **제로샷 클로닝** | 학습 없이 3~10초짜리 참고 음성만 주고 목소리를 흉내 내는 방식. 빠르지만 닮은 정도에 한계가 있음 |
| **파인튜닝** | 이미 학습된 모델을 특정 화자 데이터로 추가 학습하는 방식. 시간이 들지만 음색·말투가 훨씬 비슷해짐 |
| **LoRA** | 모델 전체가 아니라 작은 어댑터만 학습하는 방법. 메모리와 시간이 적게 듦 |

**이 프로젝트의 핵심:** 제로샷 결과를 기준선으로 삼고, 파인튜닝으로 그보다 나은 품질을 얻는 것.

## 3. 후보 모델 (한국어 지원)

| 모델 | 특징 | 필요한 데이터 | 라이선스 |
|---|---|---|---|
| **Qwen3-TTS** (0.6B / 1.7B Base) | 2026년 1월 공개. 공식 파인튜닝 스크립트를 제공하고, MLX로 Mac에서도 LoRA 학습 가능 | 수십 분 권장 | Apache-2.0 |
| **GPT-SoVITS** | 웹 UI 하나로 자르기 → 자막 만들기 → 학습까지 진행. 입문용으로 가장 쉬움 | 1분부터 가능, 5~10분 권장 | MIT |
| XTTS-v2 | 다국어 제로샷 품질이 좋음 | 6초(제로샷) | CPML(비상업) |
| F5-TTS | 제로샷이 가볍고 간단함 | 5~15초(제로샷) | 모델은 CC-BY-NC |

**추천 순서:** GPT-SoVITS로 전체 흐름을 먼저 익힌 뒤 Qwen3-TTS 파인튜닝에 도전한다.

## 4. 하드웨어 권장 사양

> VRAM은 GPU 메모리를 뜻한다. Mac은 CPU와 GPU가 메모리를 함께 쓰므로(통합 메모리) 전체 메모리 용량이 곧 VRAM 한도가 된다.
> Qwen3-TTS의 LoRA와 0.6B 수치는 GPU 대여 업체의 계산기가 낸 **추정치**이고, 1.7B 전체 학습 수치는 실제 학습 사례(sabari245/qwen3-tts-audiobook-finetune)에서 가져왔다. 실제 사용량은 배치 크기와 문장 길이에 따라 달라지니 여유분을 둔다.

### 4-1. 학습(파인튜닝)에 필요한 하드웨어

| 모델 / 방식 | 최소 VRAM | 권장 VRAM | 권장 GPU 예시 |
|---|---|---|---|
| GPT-SoVITS (batch 1) | 6GB | 8~12GB | RTX 3060 12GB, RTX 4060 Ti 16GB |
| Qwen3-TTS 0.6B LoRA | ~3GB | 8GB 이상 | RTX 3060 / T4(Colab 무료) |
| Qwen3-TTS 0.6B 전체 학습 | ~8GB | 12~16GB | RTX 4060 Ti 16GB / T4 |
| Qwen3-TTS 1.7B LoRA | ~6.3GB | 12GB 이상 | RTX 3060 12GB / L4 |
| Qwen3-TTS 1.7B 전체 학습 | 24GB (bf16, batch 2) | 40GB 이상 (fp32) | RTX 3090·4090 / A100·H100 |

공통 권장 사양
- **시스템 RAM:** 최소 16GB, 권장 32GB
- **디스크:** SSD 여유 공간 50GB 이상 (환경·모델 약 10~20GB, 체크포인트 10~30GB, 데이터)
- **학습 시간 예시 (GPT-SoVITS):** 음성 3분 기준 RTX 2070에서 약 20~40분, 음성 5분 기준 RTX 3090에서 약 15~30분

### 4-2. 실행(추론)에 필요한 하드웨어

| 모델 | VRAM (FP16) | 8bit 양자화 | 비고 |
|---|---|---|---|
| GPT-SoVITS | 4GB 이상 (권장 6GB 이상) | - | Mac에서는 CPU로 동작하며 실시간보다 약 2배 빠름 |
| Qwen3-TTS 0.6B | ~2GB | ~1GB | Mac에서는 `mlx-audio`로 실행 가능 |
| Qwen3-TTS 1.7B | ~4.2GB | ~2.1GB | Mac 16GB에서도 실행 가능 |

추론은 학습보다 훨씬 가볍다. **학습은 GPU에서 하고 추론은 Mac에서 하는** 구성이 현실적이다.

### 4-3. GPU 등급별로 할 수 있는 것

GPU를 고를 때는 **VRAM 용량이 가장 중요**하다. 성능(속도)은 그다음이다.
학습 도구 대부분이 CUDA 기반이라 **NVIDIA GPU가 사실상 필수**다. AMD와 Intel GPU는 설정이 까다롭고 지원도 불안정하다.

| VRAM | GPU 예시 | 학습 | 실행 |
|---|---|---|---|
| 4~6GB | GTX 1660, RTX 3050 | 사실상 불가 (GPT-SoVITS batch 1이 한계) | 0.6B·1.7B 모두 가능 |
| 8GB | RTX 3060 Ti, 4060, 3070 | GPT-SoVITS, Qwen3-TTS 0.6B LoRA | 원활 |
| 12GB | RTX 3060 12GB, 4070 | 위 항목 + 1.7B LoRA, 0.6B 전체 학습 | 원활 |
| 16GB | RTX 4060 Ti 16GB, 4080, T4 | 1.7B 전체 학습을 뺀 나머지 | 원활 |
| 24GB | RTX 3090, 4090, L4, A10G | 1.7B 전체 학습까지 (bf16, 작은 batch) | 원활 |
| 40GB 이상 | A100, H100 (클라우드) | 1.7B 전체 학습(fp32)을 공식 품질로 | 원활 |

**권장 구성**
- **입문 / 가성비:** RTX 3060 12GB. 중고로 저렴하고 VRAM이 넉넉하다.
- **편하게:** RTX 3090 또는 4090 (24GB). 1.7B 전체 학습까지 할 수 있다.
- **구매 없이:** 클라우드 GPU를 빌린다. Colab 무료 T4(16GB)로 0.6B는 학습할 수 있고, 1.7B는 L4나 A10G(24GB)를 시간당 약 0.5~1달러에 빌리면 된다.

### 4-4. 스펙이 부족하면 어떻게 되나

| 상황 | 증상 | 해결책 |
|---|---|---|
| **VRAM 부족 (학습)** | 학습 시작 직후나 도중에 `CUDA out of memory` 오류가 나며 중단됨 | ① batch size를 1로 줄이기 ② 전체 학습 대신 LoRA 사용 ③ 긴 클립(15초 초과) 제거 ④ gradient checkpointing 켜기 ⑤ 더 작은 모델(0.6B) 사용 |
| **VRAM 부족 (실행)** | 모델 로딩 실패 또는 오류 발생 | 8bit/4bit 양자화 모델 사용(메모리 1/2~1/4, 품질은 약간 저하), 더 작은 모델 사용 |
| **시스템 RAM 부족** | 데이터 전처리 중 멈춤, 스왑 때문에 극단적으로 느려짐, 프로세스 강제 종료 | 데이터를 나눠 처리, 다른 프로그램 종료, RAM 32GB 권장 |
| **GPU가 느림 (VRAM은 충분)** | 오류 없이 동작하지만 학습 시간이 몇 배로 늘어남 | 그대로 기다리면 됨. 품질에는 영향이 없음 |
| **NVIDIA GPU 없음 (CPU만)** | 실행은 가능하지만 느림. 학습은 수십 배 느려져서 사실상 불가 | 클라우드 GPU로 학습하고 실행만 로컬에서 |
| **Mac (Apple Silicon)** | CUDA 전용 스크립트는 실행되지 않음. PyTorch MPS 경로는 오류가 나거나 느림 | MLX 기반 도구(`mlx-tune`, `mlx-audio`) 사용. 통합 메모리 용량 안에서 작은 모델만 |
| **디스크 부족** | 체크포인트 저장에 실패해 학습 결과를 잃음 | SSD 여유 공간 50GB 이상 확보, 오래된 체크포인트 삭제 |

**정리:** VRAM이 부족하면 대개 **품질이 떨어지는 게 아니라 아예 실행되지 않는다(OOM)**. 반대로 GPU 성능이 부족하면 **느려질 뿐 결과는 같다**. 그러니 VRAM 용량을 먼저 확보한다.

### 4-5. 현재 머신에 적용하면

- 현재 머신: **Apple M2 / 16GB 통합 메모리 / NVIDIA GPU 없음**
- 추론: Qwen3-TTS 0.6B와 1.7B 모두 **가능**
- 학습
  - Mac에서 학습: `mlx-tune`으로 Qwen3-TTS 0.6B LoRA 학습 (batch 1, 시도해 볼 만함)
  - 공식 스크립트(CUDA 전용)와 1.7B 전체 학습: **클라우드 GPU가 필요함** (Colab T4/L4, RunPod 같은 서비스에서 시간당 약 0.5~1달러)

### 4-6. 파인튜닝에 걸리는 시간

**학습 자체는 생각보다 짧다. 시간이 오래 걸리는 쪽은 데이터 준비다.**

#### 학습 시간 (실제 사례)

| 모델 | 데이터 | GPU | 학습 시간 |
|---|---|---|---|
| GPT-SoVITS | 3분 | RTX 2070 | 약 20~40분 |
| GPT-SoVITS | 5분 | RTX 3090 | 약 15~30분 |
| Qwen3-TTS 1.7B | 1.5시간 (800클립, 8 epoch) | H100 | 약 1시간 |
| Qwen3-TTS 1.7B | 8.5시간 (2,300클립, 5 epoch) | A100 80GB | 약 13분 (전체 비용 약 4달러) |

#### 예상 시간 (추정)

| 환경 | 데이터 30분 기준 |
|---|---|
| GPT-SoVITS + RTX 3060급 | 약 1시간 이내 |
| Qwen3-TTS 0.6B LoRA + 클라우드 T4/L4 | 약 30분~2시간 |
| Qwen3-TTS 1.7B 전체 학습 + RTX 3090/4090 | 수 시간 (작은 batch라서 느림) |
| Mac M2 16GB + mlx-tune (0.6B LoRA) | 수 시간~하루 (직접 측정 필요) |
| CPU만 사용 | 사실상 불가 |

학습 시간은 **데이터 양 × epoch 수**에 비례한다. GPU 성능이 절반이면 대략 2배 걸린다.

#### 전체 프로젝트 시간 (처음 하는 경우)

| 단계 | 예상 시간 |
|---|---|
| 환경 설치 | 반나절 |
| 녹음 (Craig 등) | 원하는 분량의 2~3배 (쓸 수 없는 구간을 감안) |
| 자르기와 자동 전사 | 1시간 이내 |
| **전사 직접 교정** | **녹음 30분당 2~3시간** (가장 오래 걸림) |
| 학습 | 위 표 참고 |
| 평가와 재학습 | 1~2회 반복, 각 반나절 |

**참고:** Baseten 실험에서는 1.5시간 데이터로 파인튜닝한 Qwen3-TTS가 유사도와 음질 면에서 **제로샷보다 확실히 낫지는 않았다**. 자연스러움과 표현력은 일부 좋아졌고, 데이터가 수 시간 이상으로 늘어날수록 효과가 커진다고 한다. 그러니 제로샷 기준선을 먼저 만들어 두는 단계(6장)를 꼭 거친다.

## 5. 필요한 데이터

### 5-1. 분량

| 목적 | 분량 | 기대 품질 |
|---|---|---|
| 제로샷 참고 음성 | 3~10초 1개 | 목소리 느낌만 비슷함 |
| GPT-SoVITS 파인튜닝 | 최소 1분, **권장 5~30분** | 음색이 꽤 비슷해짐 |
| Qwen3-TTS 파인튜닝 | **권장 30분~1시간 이상** | 음색과 말투가 안정적으로 재현됨 |

- 클립 하나는 **3~15초**로 맞춘다. 30분 분량이면 대략 200~500개가 된다.
- 양보다 **질**이 중요하다. 깨끗한 10분이 잡음 섞인 1시간보다 낫다.
- 평서문, 의문문, 감탄문, 숫자, 외래어처럼 **다양한 문장**을 섞어야 학습하지 않은 문장도 잘 읽는다.

### 5-2. 오디오 형식

| 항목 | 권장 |
|---|---|
| 파일 형식 | WAV (무압축). mp3는 품질이 손실되므로 피함 |
| 샘플레이트 | 24kHz 이상 (원본은 44.1/48kHz로 녹음하고 필요할 때 변환) |
| 채널 | 모노 |
| 환경 | 한 사람의 목소리만, 배경음악·잡음·울림 없이, 일정한 마이크 거리 유지 |
| 음량 | 클리핑(소리 깨짐) 없이 일정하게 |

### 5-3. 텍스트(전사) 형식

각 오디오 클립마다 **실제로 말한 내용과 정확히 일치하는 텍스트**가 있어야 한다.

**Qwen3-TTS:** JSONL, 한 줄에 클립 하나
```json
{"audio": "./data/wavs/0001.wav", "text": "오늘 날씨가 정말 좋네요.", "ref_audio": "./data/ref.wav"}
{"audio": "./data/wavs/0002.wav", "text": "회의는 세 시에 시작합니다.", "ref_audio": "./data/ref.wav"}
```
- `ref_audio`는 모든 줄에 **같은 파일**을 쓰는 것이 권장된다 (화자 일관성이 좋아짐).

**GPT-SoVITS:** `.list` 파일, `경로|화자명|언어|텍스트`
```
/data/wavs/0001.wav|me|ko|오늘 날씨가 정말 좋네요.
/data/wavs/0002.wav|me|ko|회의는 세 시에 시작합니다.
```
- 웹 UI의 자르기 → ASR 기능으로 자동 생성한 뒤 직접 교정한다.

### 5-4. 폴더 구조 예시

```
data/
├── raw/          # 원본 녹음 (긴 파일)
├── wavs/         # 3~15초로 자른 클립
├── ref.wav       # 대표 참고 음성 (5~10초)
├── train.jsonl   # 학습용 (약 90%)
└── test.jsonl    # 평가용 (약 10%, 학습에 쓰지 않음)
```

## 6. 진행 단계

1. **데이터 녹음·수집**: 위 5장의 기준에 맞춰 녹음한다.
2. **전처리**: 녹음을 3~15초 단위로 자르고, 무음과 잡음을 제거한다.
3. **전사(자막 달기)**: Whisper 같은 ASR로 각 조각의 텍스트를 뽑은 뒤 사람이 직접 교정한다. 품질에 가장 큰 영향을 주는 단계다.
4. **기준선 만들기**: 제로샷 클로닝으로 샘플 음성을 생성해 둔다.
5. **파인튜닝**: 학습 데이터는 `{audio, text, ref_audio}` 형식의 JSONL로 만든다(Qwen3-TTS 기준).
6. **평가**: 학습에 쓰지 않은 문장으로 기준선과 들어서 비교한다(유사도, 발음, 자연스러움).

## 7. 성공 기준

- [ ] 학습에 쓰지 않은 한국어 문장 10개를 생성했을 때, 듣는 사람이 원래 화자의 목소리라고 느낀다
- [ ] 발음 오류나 끊김이 제로샷 기준선보다 적다
- [ ] 학습부터 추론까지 다시 실행할 수 있도록 스크립트와 문서가 남아 있다

## 8. 주의사항

- **본인 또는 명시적으로 동의한 사람의 목소리만** 사용한다. 타인의 목소리를 무단으로 복제하면 초상권·개인정보 관련 법에 저촉될 수 있다.
- 모델마다 라이선스가 다르다. 상업적으로 쓰려면 Apache-2.0이나 MIT 모델을 고른다.

## 참고 자료

- [Qwen3-TTS (GitHub)](https://github.com/QwenLM/Qwen3-TTS) · [파인튜닝 가이드](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [GPT-SoVITS 음성 클로닝 가이드](https://localaimaster.com/blog/gpt-sovits-voice-cloning-guide)
- [mlx-tune: Apple Silicon에서의 TTS 파인튜닝](https://github.com/ARahim3/mlx-tune)
- [mlx-audio 학습 지원 논의](https://github.com/Blaizzy/mlx-audio/discussions/619)
- [2026 오픈소스 TTS 모델 비교 (BentoML)](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [오픈소스 TTS 모델 비교 (Modal)](https://modal.com/blog/open-source-tts)
- [TTS 모델 비교 (CodeSOTA)](https://www.codesota.com/guides/tts-models)
- [Qwen3-TTS 0.6B VRAM 추정 (Spheron)](https://www.spheron.network/tools/gpu-recommender/Qwen/Qwen3-TTS-12Hz-0.6B-Base/) · [1.7B](https://www.spheron.network/tools/gpu-recommender/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/)
- [GPT-SoVITS 시스템 요구사항](https://gpt-sovits.org/gpt-sovits-system-requirements-complete-setup/)
- [GPT-SoVITS 학습 가이드 (AI Hub)](https://docs.aihub.gg/tts/gpt-sovits/)
- [Fine-tuning Qwen3-TTS (Baseten)](https://www.baseten.co/blog/fine-tuning-qwen3-tts-for-high-quality-voice-cloning/)
- [qwen3-tts-audiobook-finetune (실제 학습 사례)](https://github.com/sabari245/qwen3-tts-audiobook-finetune)
- 데이터 수집: [craig.md](./craig.md) · 추천 작업 순서: [workflow.md](./workflow.md)

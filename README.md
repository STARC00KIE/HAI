# 🚗 HAI Project (Refactoring in Progress)

> 앙상블 모델 기반 차량 이미지 분류 프로젝트  
> 현재 `.ipynb` 중심 코드를 `.py` 모듈 기반 구조로 리팩토링 중

---

## 📌 리팩토링 배경

- Jupyter 기반으로 개발하면서 파일이 점점 길어지고 복잡해짐
- 기능별 분리가 되지 않아 유지보수 및 재사용 어려움
- 추론 및 제출 코드까지 혼재되어 있어 구조화 필요성 느낌

---

## 🎯 리팩토링 목표

- ✅ 학습 / 추론 / 제출 기능을 독립적인 `.py` 모듈로 분리
- ✅ 모델, Trainer, Dataset, Utils 등을 재사용 가능한 구조로 재정리
- ✅ Jupyter 외 환경(CLI 또는 배치 실행)에서도 문제없이 실행되도록 변경
- ✅ 향후 Soft Voting, 앙상블, 하이퍼파라미터 튜닝 등 추가 기능 확장 용이성 확보
- ✅ 리팩토링은 단계별로 진행하며 정상 동작을 확인하면서 전환 예정

---

# 📁 디렉토리 구조

```
HAI/
│
├── config.py                   # 설정값 모음 (argparse 기반 덮어쓰기 가능)
├── train.py                   # 학습 실행 스크립트
├── submit.py                  # 추론 후 submission.csv 저장
├── inference.py               # 추론 모듈
│
├── data/
│   ├── dataset.py             # CustomImageDataset 정의
│   ├── transforms.py          # Albumentations 기반 transform 정의
│   ├── train/                 # 학습용 이미지 (클래스별 하위 폴더)
│   └── test/                  # 테스트 이미지 (단일 폴더)
│
├── models/
│   └── base_model.py          # 모델 정의 (timm 사용)
│
├── engine/
│   ├── trainer.py             # Trainer 클래스
│   └── utils.py               # seed 고정, 디바이스 설정 등 유틸 함수
│
├── logs/
│   ├── all_logs.csv           # 전체 실험 로그 누적
│   └── log_resnet18_img224_bs64_20250529_1853.csv  # 개별 실험 로그
│
├── result/
│   ├── sample_submission.csv  # 예시 제출 포맷
│   ├── submission_*.csv       # 실제 제출 파일
│   └── *.pth                  # 학습된 모델 저장 파일
│
└── .gitignore                 # 캐시/모델 등 제외
```

---

# ⚙️ 설정 항목 변경 사항

- `config.py`에는 `DEFAULT_CFG`만 유지
- argparse 사용으로 실행 시 하이퍼파라미터 수정 가능
- `FILE_NAME` → 로그/모델 저장용 (접두사 없음)
- `CSV_NAME` → 제출용 csv 파일명 (`submission_` 접두사 포함)

---

# 📊 학습 후 로그 및 제출 자동 저장

- `train.py` 실행 시 자동으로 `logs/`, `pth/`, `result/` 폴더에 파일 저장
- `--no-submit` 인자 사용 시 자동 제출 생략 가능

```bash
python train.py --model_name resnet18 --img_size 384 --no-submit
```

---

# ✅ 기타
- `.gitignore` 설정 포함 (e.g. `__pycache__`, `*.pth`, `logs/`, `result/` 등 제외)
- `.git/index.lock` 오류 시 수동 삭제 후 재시도 (`.git/index.lock`)

---

# 🚀 사용 방법 (Usage)

```bash
python train.py --model_name resnet18 --img_size 224 --batch_size 64 --epochs 30
```

- 옵션 설명

| 인자 이름          | 설명                  | 기본값      |
| -------------- | ------------------- | -------- |
| `--model_name` | 사용할 모델 이름 (timm 기반) | resnet18 |
| `--img_size`   | 입력 이미지 크기           | 384      |
| `--batch_size` | 배치 사이즈              | 64       |
| `--epochs`     | 학습 반복 횟수            | 30       |
| `--lr`         | 학습률                 | 1e-4     |
| `--patience`   | 조기 종료 기준            | 5        |
| `--no-submit`  | 학습 후 자동 제출 생략       | -        |

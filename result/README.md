# 🚗 중고차 이미지 분류 - 제출 파일 기록

이 디렉토리는 Hecto AI Challenge - 중고차 이미지 기반 차종 분류 모델 개발 실험의 결과 제출 파일을 보관합니다.  
각 파일은 실험 세팅과 logloss 기준 성능을 기반으로 관리됩니다.

---

## 📁 제출 파일 목록

| 파일명 | 생성일 | 모델 | 주요 설정 | LogLoss | 비고 |
|--------|--------|-------|-------------|----------|------|
| `submission_01.csv` | 2025-05-20 | ResNet18 | ✅ 데이터 증강<br>(Albumentations: ColorJitter, Flip, ShiftScaleRotate 등) | **0.3580** | 현재까지 최고 성능 |
| `submission_02.csv` | 2025-05-21 | ResNet18 | ✅ EarlyStopping(patience=5)<br>❌ 데이터 증강 미포함 | 0.3811 | 성능 하락 (Baseline보다 낮음) |

---

## 🧪 향후 제출 예정 항목

- `submission_03.csv`  
  - ✅ ConvNeXt-Tiny 적용  
  - ✅ 데이터 증강 + EarlyStopping 병합  
  - ✅ 고해상도 (384x384)

- `submission_04.csv`  
  - ✅ ConvNeXt-Base or EfficientNetV2 실험  
  - ✅ 학습 전략: AMP, Label Smoothing, Scheduler 포함

- `ensemble_top3.csv`  
  - ✅ Top-k 모델 앙상블 (soft voting)

---

## 🔐 백업 및 관리 안내

- `.csv` 파일은 모두 보관 대상입니다.  
- 백업: 외부 스토리지 또는 Git LFS 권장 (파일 크기 50MB 초과 가능)  
- 각 제출 파일은 실험 당시 `seed`, `img_size`, `lr`, `augmentation` 등 설정을 기록할 것

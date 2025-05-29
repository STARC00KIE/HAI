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

## 🏗️ 계획된 디렉토리 구조

```plaintext
HAI/
├── config.py                  ← 공통 설정값 (CFG)
├── train.py                   ← 학습 실행 스크립트
├── inference.py               ← 추론 실행 스크립트
├── submit.py                  ← 결과 저장/제출 스크립트
│
├── data/                      ← 데이터 저장 경로
│   ├── train/                 ← 학습 데이터
│   └── test/                  ← 평가 데이터
│
├── models/
│   ├── __init__.py
│   └── base_model.py          ← BaseModel 클래스
│
├── engine/
│   ├── __init__.py
│   ├── trainer.py             ← Trainer 클래스
│   └── utils.py               ← load_model, seed_everything 등
│
├── data/
│   ├── __init__.py
│   ├── dataset.py             ← CustomDataset 클래스
│   └── transforms.py          ← Transform 구성
│
├── pth/                       ← 모델 저장 경로
├── result/                    ← 예측 결과 저장
└── ensemble.ipynb             ← 기존 노트북 (백업용)

---

## 🕓 버전 히스토리 (Change Log)

| 날짜       | 버전 | 내용 요약                                        |
|------------|------|--------------------------------------------------|
| 2025-05-29 | v0.1 | 프로젝트 리팩토링 구조 설계 (`README.md` 작성)    |
|            |      | 모델 구조 분리: `models/base_model.py` 생성       |
|            |      | `.gitignore` 작성 예정                             |
|            |      | `train.py`, `inference.py` 분리 작업 예정          |
|            |      | `.ipynb` → `.py` 기반 전환 시작                   |
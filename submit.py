# submit.py

import pandas as pd
from inference import run_inference_probs
from config import CFG
from datetime import datetime

def save_submission(cfg):
    """
    예측 확률을 sample_submission 형식에 맞춰 저장하는 함수.
    저장 파일명은 모델명, 이미지 사이즈, 배치사이즈, 타임스탬프를 포함하여 고유하게 생성된다.
    """
    # 1. 예측 확률 DataFrame 얻기
    pred = run_inference_probs(cfg)

    # 2. 샘플 제출 양식 불러오기
    submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')

    # 3. 클래스 컬럼 정렬 일치
    class_columns = submission.columns[1:]
    pred = pred[class_columns]
    submission[class_columns] = pred.values

    # 4. 파일명은 config.py에서 불러옴
    file_path = f"./result/{cfg['FILE_NAME']}.csv"

    # 5. 저장
    submission.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {file_path}")

if __name__ == "__main__":
    save_submission(CFG)
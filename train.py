# 경고 무시하기
import warnings
warnings.filterwarnings("ignore")

import argparse
from datetime import datetime
import torch
import os
from config import CFG as DEFAULT_CFG
from engine.utils import seed_everything, get_device
from data.dataset import CustomImageDataset
from data.transforms import get_train_transform, get_val_transform
from engine.trainer import Trainer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from submit import save_submission

def parse_args():
    """
    커맨드라인 인자들을 정의하고 파싱하는 함수
    """
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument('--model_name', type=str, default=DEFAULT_CFG['MODEL_NAME'], help='사용할 Timm 모델 이름')
    parser.add_argument('--img_size', type=int, default=DEFAULT_CFG['IMG_SIZE'], help='입력 이미지 크기')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CFG['BATCH_SIZE'], help='배치 사이즈')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CFG['EPOCHS'], help='에폭 수')
    parser.add_argument('--lr', type=float, default=DEFAULT_CFG['LEARNING_RATE'], help='학습률')
    parser.add_argument('--patience', type=int, default=DEFAULT_CFG['PATIENCE'], help='Early stopping 기준')
    
    parser.add_argument('--submit', dest='submit', action='store_true', help='학습 후 자동 제출 (기본값: True)')
    parser.add_argument('--no-submit', dest='submit', action='store_false', help='학습 후 제출 생략')
    parser.set_defaults(submit=True)

    return parser.parse_args()

def run_training(cfg):
    """
    전체 학습 과정을 수행하는 함수
    - 데이터 로딩
    - Trainer 객체 생성
    - 학습 실행
    """
    # 재현성을 위한 시드 고정
    seed_everything(cfg['SEED'])

    # 디바이스 설정 (GPU 또는 CPU)
    device = get_device()

    # 클래스 목록 추출 (알파벳 순 정렬)
    class_names = sorted(os.listdir(cfg['TRAIN_DIR']))

    # 전체 학습 데이터셋 로딩 (라벨 포함)
    full_dataset = CustomImageDataset(root_dir=cfg['TRAIN_DIR'], transform=None, is_test=False)

    # 라벨 리스트 구성 (stratify에 사용)
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    # 훈련/검증 데이터 분리 (Stratified Split)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=cfg['SEED']
    )

    # Subset으로 각각 구성
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # 데이터 변환 적용 (transform은 dataset.dataset에 적용해야 함)
    train_dataset.dataset.transform = get_train_transform(cfg['IMG_SIZE'])
    val_dataset.dataset.transform = get_val_transform(cfg['IMG_SIZE'])

    # DataLoader 구성
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=cfg['NUM_WORKERS']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=cfg['NUM_WORKERS']
    )

    # Trainer 객체 생성 및 학습 시작
    trainer = Trainer(
        model_name=cfg['MODEL_NAME'],
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg
    )

    trainer.train()

if __name__ == "__main__":
    # 커맨드라인 인자 파싱
    args = parse_args()

    # config 복사 및 인자값으로 덮어쓰기
    CFG = DEFAULT_CFG.copy()
    CFG['MODEL_NAME'] = args.model_name
    CFG['IMG_SIZE'] = args.img_size
    CFG['BATCH_SIZE'] = args.batch_size
    CFG['EPOCHS'] = args.epochs
    CFG['LEARNING_RATE'] = args.lr
    CFG['PATIENCE'] = args.patience

    # 고유한 파일명 생성 (모델명, 이미지 크기, 배치사이즈, 타임스탬프 포함)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    unique_name = f"{CFG['MODEL_NAME']}_img{CFG['IMG_SIZE']}_bs{CFG['BATCH_SIZE']}_{timestamp}"
    
    CFG['FILE_NAME'] = unique_name                     # log, ckpt 등에 사용
    CFG['CSV_NAME'] = f"submission_{unique_name}"      # 제출 파일(csv) 전용 이름
    CFG['CKPT_NAME'] = f"best_{unique_name}.pth"

    # 학습 실행
    run_training(CFG)

    if args.submit:
        save_submission(CFG)

    # 학습 후 GPU 메모리 정리
    torch.cuda.empty_cache()
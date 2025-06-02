
import torch


# 필요한 경우, yaml 기반 설정(Hydra or argparse)으로 확장 가능하지만  
#현재 프로젝트 규모엔 Python dict 기반이 가장 실용적

CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-4,
    'PATIENCE': 5,
    'SEED': 42,
    'MODEL_NAME': 'resnet18',
    'TRAIN_DIR': './data/train',
    'TEST_DIR': './data/test',
    'NUM_WORKERS': 4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}


"""
# config.py
from datetime import datetime
import torch

def get_default_cfg():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    unique_name = f"resnet18_img224_bs64_{timestamp}"

    return {
        'IMG_SIZE': 224,
        'BATCH_SIZE': 64,
        'EPOCHS': 30,
        'LEARNING_RATE': 1e-4,
        'PATIENCE': 5,
        'SEED': 42,
        'MODEL_NAME': 'resnet18',
        'FILE_NAME': unique_name,
        'CSV_NAME': f"submission_{unique_name}",
        'CKPT_NAME': f"best_{unique_name}.pth",
        'TRAIN_DIR': './data/train',
        'TEST_DIR': './data/test',
        'NUM_WORKERS': 4,
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def generate_unique_names(cfg):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    unique_name = f"{cfg['MODEL_NAME']}_img{cfg['IMG_SIZE']}_bs{cfg['BATCH_SIZE']}_{timestamp}"
    cfg['FILE_NAME'] = unique_name
    cfg['CSV_NAME'] = f"submission_{unique_name}"
    cfg['CKPT_NAME'] = f"best_{unique_name}.pth"
    return cfg
"""
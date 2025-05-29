import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    재현성을 위한 seed 고정 함수.
    무작위성 요소를 제어하여 동일한 결과를 얻을 수 있도록 한다.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

def get_device():
    """
    GPU 사용 가능 여부에 따라 디바이스를 설정한다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    return device

def make_dir(path: str):
    """
    디렉토리 없으면 생성하는 함수
    """
    os.makedirs(path, exist_ok=True)
    print(f"📁 Directory ensured: {path}")

import os
import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader
from config import CFG
from engine.utils import get_device
from models.base_model import BaseModel
from data.dataset import CustomImageDataset
from data.transforms import get_val_transform

def run_inference_probs():
    """
    테스트 데이터에 대한 softmax 확률 출력
    결과는 클래스별 확률 분포 DataFrame으로 반환
    """
    device = get_device()
    class_names = sorted(os.listdir(CFG['TRAIN_DIR']))
    num_classes = len(class_names)

    # 테스트셋 로딩
    test_dataset = CustomImageDataset(
        root_dir=CFG['TEST_DIR'],
        transform=get_val_transform(CFG['IMG_SIZE']),
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=False
    )

    # 모델 로드
    model = BaseModel(model_name=CFG['MODEL_NAME'], num_classes=num_classes)
    model.load_state_dict(torch.load(f"./pth/{CFG['CKPT_NAME']}", map_location=CFG['DEVICE']))
    model.to(CFG['DEVICE'])
    model.eval()

    # 추론 결과 확률 저장
    results = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            for prob in probs.cpu():
                result = {
                    class_names[i]: prob[i].item()
                    for i in range(num_classes)
                }
                results.append(result)

    df = pd.DataFrame(results)
    return df

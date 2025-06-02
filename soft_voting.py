"""
# soft_voting.py
import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from models.base_model import BaseModel
from data.dataset import CustomImageDataset
from data.transforms import get_val_transform
from torch.utils.data import DataLoader
from config import get_default_cfg, generate_unique_names

def load_model(pth_path, model_name, num_classes, device):
    model = BaseModel(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def soft_voting_inference(pth_list, model_names, cfg):
    device = cfg['DEVICE']
    class_names = sorted(os.listdir(cfg['TRAIN_DIR']))
    num_classes = len(class_names)

    test_dataset = CustomImageDataset(cfg['TEST_DIR'], transform=get_val_transform(cfg['IMG_SIZE']), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False)

    all_probs = []

    for pth, name in zip(pth_list, model_names):
        print(f"🔍 Inference with: {name}")
        model = load_model(pth, name, num_classes, device)
        probs = []

        with torch.no_grad():
            for images in tqdm(test_loader, desc=f"{name} predicting"):
                images = images.to(device)
                outputs = model(images)
                probs.append(F.softmax(outputs, dim=1).cpu())

        all_probs.append(torch.cat(probs, dim=0))

    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    final_preds = torch.argmax(avg_probs, dim=1)

    submission = pd.read_csv('./sample_submission.csv')
    submission['label'] = final_preds.numpy()
    output_path = f"./result/softvote_submission.csv"
    submission.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Soft voting 결과 저장 완료 → {output_path}")

if __name__ == "__main__":
    cfg = get_default_cfg()
    cfg['IMG_SIZE'] = 224
    cfg['BATCH_SIZE'] = 64
    cfg['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 필요한 모델명과 경로
    model_names = ['resnet18', 'efficientnet_b0', 'convnext_tiny', 'mobilenetv2_100']
    pth_list = [
        './pth/best_resnet18_img224_bs64_20250529_2051.pth',
        './pth/best_efficientnet_b0_img224_bs64_20250529_2240.pth',
        './pth/best_convnext_tiny_img224_bs32_20250530_0719.pth',
        './pth/best_mobilenetv2_100_img224_bs64_20250530_0313.pth'
    ]

    soft_voting_inference(pth_list, model_names, cfg)
"""

# soft_voting.py
import os
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import CustomImageDataset
from data.transforms import get_val_transform
from models.base_model import BaseModel
from config import get_default_cfg

# 1. 기본 config 불러오기
cfg = get_default_cfg()
device = cfg["DEVICE"]
class_names = sorted(os.listdir(cfg["TRAIN_DIR"]))
num_classes = len(class_names)

# 2. 앙상블 대상 모델 리스트 (모델명과 pth 파일 매핑)
ensemble_models = [
    {"name": "resnet18", "pth": "best_resnet18_img224_bs64_20250529_2051.pth"},
    {"name": "efficientnet_b0", "pth": "best_efficientnet_b0_img224_bs64_20250529_2240.pth"},
    {"name": "convnext_tiny", "pth": "best_convnext_tiny_img224_bs32_20250530_0719.pth"},
    {"name": "mobilenetv2_100", "pth": "best_mobilenetv2_100_img224_bs64_20250530_0313.pth"},
    # 필요 시 추가 가능
]

# 3. 테스트셋 준비
print("📦 Preparing test dataset...")
test_dataset = CustomImageDataset(
    root_dir=cfg["TEST_DIR"],
    transform=get_val_transform(cfg["IMG_SIZE"]),
    is_test=True,
)
test_loader = DataLoader(test_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False)

# 4. 예측 확률 계산 (소프트 보팅)
print("🚀 Starting inference...")
probs_list = []

for model_info in ensemble_models:
    model_name = model_info["name"]
    model_path = model_info["pth"]

    print(f"\n🔍 Loading {model_name} from {model_path}")
    model = BaseModel(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(f"./pth/{model_path}", map_location=device))
    model.to(device)
    model.eval()

    probs = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc=f"Inferencing {model_path}"):
            images = images.to(device)
            outputs = model(images)
            prob = F.softmax(outputs, dim=1)
            probs.append(prob.cpu())
    probs_list.append(torch.cat(probs, dim=0))

# 5. 평균 확률 → 최종 예측
avg_probs = torch.stack(probs_list).mean(dim=0)
preds = avg_probs.argmax(dim=1).numpy()

# 6. 제출 파일 저장
submission = pd.read_csv("./sample_submission.csv")
submission["label"] = preds
submission.to_csv(f"./result/soft_voting_submission.csv", index=False)

print("\n✅ soft voting 예측 결과 저장 완료 → ./result/soft_voting_submission.csv")

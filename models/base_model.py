import torch
import torch.nn as nn
import timm

class BaseModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        try:
            in_features = self.backbone.num_features
        except AttributeError:
            raise ValueError(f"❌ 모델 {model_name}에서 in_features를 찾을 수 없습니다.")

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if self._needs_pooling() else None
        self.classifier = nn.Linear(in_features, num_classes)

    def _needs_pooling(self):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = self.backbone(x)
        return y.dim() == 4  # (B, C, H, W)인지 확인

    def forward(self, x):
        x = self.backbone(x)
        if self.pool:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

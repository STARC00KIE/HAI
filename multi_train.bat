@echo off
REM [0] 아나콘다 환경 활성화
call conda activate HAI

REM [1] ResNet18
python train.py --model_name resnet18 --img_size 224 --batch_size 64 || echo ❌ ResNet18 실패

REM [2] EfficientNet
python train.py --model_name efficientnet_b0 --img_size 224 --batch_size 64 || echo ❌ EfficientNet 실패

REM [3] MobileNet
python train.py --model_name mobilenetv2_100 --img_size 224 --batch_size 64 || echo ❌ MobileNet 실패

REM [4] ConvNeXt Tiny (배치 낮게)
python train.py --model_name convnext_tiny --img_size 224 --batch_size 32 || echo ❌ ConvNeXt Tiny 실패

REM [5] ViT Tiny (주의: 메모리 부족 가능성 있음)
python train.py --model_name vit_tiny_patch16_224 --img_size 224 --batch_size 16 || echo ❌ ViT Tiny 실패

pause
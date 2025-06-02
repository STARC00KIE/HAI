@echo off
REM [0] 아나콘다 환경 활성화
call conda activate HAI

REM [1] ResNet18 (384 이미지에 적당한 배치 크기: 32~64)
python train.py --model_name resnet18 --img_size 384 --batch_size 32 || echo ❌ ResNet18 실패

REM [2] EfficientNet-B0 (384 이미지 + 적당한 배치 크기: 16~32)
python train.py --model_name efficientnet_b0 --img_size 384 --batch_size 16 || echo ❌ EfficientNet 실패

REM [3] MobileNetV2 (경량 모델, 384 이미지에서도 32 가능)
python train.py --model_name mobilenetv2_100 --img_size 384 --batch_size 32 || echo ❌ MobileNet 실패

REM [4] ConvNeXt Tiny (384 이미지에선 메모리 사용량 큼: 16 이하 추천)
python train.py --model_name convnext_tiny --img_size 384 --batch_size 16 || echo ❌ ConvNeXt Tiny 실패

REM [5] ViT Tiny (384 이미지에서는 매우 높은 메모리 요구: 8 이하 추천)
python train.py --model_name vit_tiny_patch16_384 --img_size 384 --batch_size 8 || echo ❌ ViT Tiny 실패

pause
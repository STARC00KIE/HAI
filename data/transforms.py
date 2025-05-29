import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(img_size):
    """
    학습 데이터용 데이터 증강 파이프라인
    - 조명, 방향, 위치, 그림자 등 다양한 상황에 대응
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # 좌우 반전: 차량 방향 다양화
        A.HorizontalFlip(p=0.5),

        # 밝기/대비 랜덤 조정
        A.RandomBrightnessContrast(p=0.2),

        # 색상(Hue), 채도(Saturation), 명도(Value)
        A.HueSaturationValue(p=0.2),

        # 이동, 확대/축소, 회전
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.05,
                           rotate_limit=15,
                           p=0.5),

        # 그림자 추가
        A.RandomShadow(p=0.2),

        # RGB 색상 채널 변형
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),

        # 일부분 무작위 제거 (가림 상황)
        A.CoarseDropout(max_holes=2,
                        max_height=16,
                        max_width=16,
                        p=0.3),

        # 정규화 (ImageNet mean/std)
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),

        ToTensorV2()
    ])

def get_val_transform(img_size):
    """
    검증 및 테스트 데이터는 증강 없이 리사이즈 + 정규화만 적용
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_test_transform(img_size):
    """
    테스트셋도 검증셋과 동일한 transform 사용
    테스트 데이터는 여기서 진행 안하지만 혹시나 하는 마음에 제작
    """
    return get_val_transform(img_size)
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """
    PyTorch용 커스텀 이미지 데이터셋 클래스.
    - 학습/검증/테스트 데이터 로딩을 담당하며, albumentations 기반 transform을 적용함.
    - 학습 데이터는 클래스별 폴더 구조, 테스트 데이터는 단일 폴더 구조를 가정.
    """

    def __init__(self, root_dir, transform=None, is_test=False):
        """
        데이터셋 초기화

        Args:
            root_dir (str): 이미지가 저장된 루트 폴더 경로
                - 학습/검증: root/class_name/image.jpg 구조
                - 테스트: root/image.jpg 구조
            transform (callable, optional): 이미지 전처리(transform) 함수
            is_test (bool): 테스트셋 여부 (True일 경우 라벨 없이 이미지만 불러옴)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            # 테스트셋: 라벨 없이 이미지 경로만 저장
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith('.jpg'):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            # 학습셋: 클래스별 폴더 구조에서 라벨 추출
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith('.jpg'):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        """
        전체 데이터 수 반환
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        하나의 샘플(image, label 또는 image) 반환

        Returns:
            - (image, label) → 학습/검증 데이터일 때
            - image → 테스트 데이터일 때
        """
        if self.is_test:
            img_path = self.samples[idx][0]
            image = np.array(Image.open(img_path).convert('RGB'))
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        else:
            img_path, label = self.samples[idx]
            image = np.array(Image.open(img_path).convert('RGB'))
            if self.transform:
                image = self.transform(image=image)['image']
            return image, label

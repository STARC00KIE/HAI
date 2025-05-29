import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss
from models.base_model import BaseModel # 커스텀 모델 클래스
from datetime import datetime

"""
특징:

-BaseModel 직접 호출    : 모델 이름과 클래스 수 기반으로 유연한 생성
- 얼리스토핑 내장   :log_loss 기준, PATIENCE 횟수 이상 개선 없으면 중단
- 평가 지표 다양    : val loss, accuracy, log_loss 모두 출력
- best model 저장   : ./pth/ 디렉토리에 자동 저장
- epoch당 train log 저장: ./log/ 디렉토리에 개별 모델, 전체 모델 저장   
"""

class Trainer:
    def __init__(self, model_name, class_names, train_loader, val_loader, device, cfg):
        """
        Trainer 객체 초기화

        Args:
            model_name (str): 사용할 모델 이름 (timm 기반)
            class_names (List[str]): 분류할 클래스 이름 리스트
            train_loader (DataLoader): 학습용 데이터 로더
            val_loader (DataLoader): 검증용 데이터 로더
            device (torch.device): 학습에 사용할 디바이스
            cfg (dict): 학습 설정이 담긴 설정 딕셔너리
        """
        self.model_name = model_name
        self.class_names = class_names
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.num_classes = len(class_names)

        # 모델 정의 및 손실함수, 옵티마이저 설정
        self.model = BaseModel(model_name=self.model_name, num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['LEARNING_RATE'])

        # Early stopping을 위한 최적 logloss 기록
        self.best_logloss = float('inf')
        self.history = []  # 로그 기록용 리스트

        # 체크포인트, 로그 디렉토리 생성
        os.makedirs('./pth', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)

    def train(self):
        """
        전체 학습 루프 수행 함수
        - 각 epoch마다 학습 → 검증
        - logloss 향상 시 모델 저장
        - 성능 로그 기록 및 저장
        - early stopping 적용
        """
        patience = self.cfg.get('PATIENCE', 5)
        counter = 0

        for epoch in range(self.cfg['EPOCHS']):
            self.model.train()
            total_train_loss = 0

            for images, labels in tqdm(self.train_loader, desc=f"[{self.model_name}] Epoch {epoch+1}/{self.cfg['EPOCHS']} - Training"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)

            # 검증 수행 및 성능 평가
            avg_val_loss, val_accuracy, val_logloss = self.validate(epoch)

            # 성능 기록
            self.history.append({
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M'),
                'model': self.model_name,
                'img_size': self.cfg['IMG_SIZE'],
                'batch_size': self.cfg['BATCH_SIZE'],
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_acc': val_accuracy,
                'val_logloss': val_logloss
            })

            # 모델 성능 개선 시 저장
            if val_logloss < self.best_logloss:
                self.best_logloss = val_logloss
                save_path = f"./pth/{self.cfg['CKPT_NAME']}"
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ Best model saved: {save_path} (logloss: {val_logloss:.4f})")
                counter = 0
            else:
                counter += 1
                print(f"⚠️ No improvement for {counter} epoch(s)")
                if counter >= patience:
                    print(f"⏹ Early stopping {self.model_name} at epoch {epoch+1}")
                    break

        # 로그 저장 (개별 로그)
        log_df = pd.DataFrame(self.history)
        log_path = f"./logs/log_{self.cfg['FILE_NAME']}.csv"
        log_df.to_csv(log_path, index=False)
        print(f"📄 개별 학습 로그 저장 완료: {log_path}")

        # 로그 저장 (통합 로그)
        all_log_path = './logs/all_logs.csv'
        log_df.to_csv(all_log_path, mode='a', header=not os.path.exists(all_log_path), index=False)
        print(f"📄 전체 통합 로그 업데이트 완료: {all_log_path}")

    def validate(self, epoch):
        """
        검증 루프 수행 함수

        Args:
            epoch (int): 현재 epoch 번호

        Returns:
            avg_val_loss (float): 평균 검증 손실
            val_accuracy (float): 검증 정확도 (%)
            val_logloss (float): log-loss 점수
        """
        self.model.eval()
        val_loss = 0
        correct, total = 0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"[{self.model_name}] Epoch {epoch+1}/{self.cfg['EPOCHS']} - Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total
        val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(self.class_names))))
        print(f"✅ [Val] Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%, LogLoss: {val_logloss:.4f}")
        return avg_val_loss, val_accuracy, val_logloss


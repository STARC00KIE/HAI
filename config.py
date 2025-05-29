"""
필요한 경우, yaml 기반 설정(Hydra or argparse)으로 확장 가능하지만 \
현재 프로젝트 규모엔 Python dict 기반이 가장 실용적
"""

CFG = {
    'IMG_SIZE': 384,
    'BATCH_SIZE': 64,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-4,
    'PATIENCE': 5,
    'SEED': 42,
    'MODEL_NAME': 'resnet18',
    'TRAIN_DIR': './data/train',
    'TEST_DIR': './data/test',
    'NUM_WORKERS': 4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}
## 7. 학습 실행 스크립트 (scripts/train.py)


import sys
sys.path.append('..')
sys.path.append('.')

import torch
from src.data.dataset import StructuredAudioDataset
from src.models.structure_encoder import StructureEncoder
from src.models.structure_dit import StructureConditionedDiT
from src.models.pipeline import StructureAwareStableAudioPipeline
from src.training.trainer import StructureAudioTrainer
import yaml

def main():
    # 설정 로드
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 생성
    print("Loading datasets...")
    train_dataset = StructuredAudioDataset(
        csv_path=config['data']['train_csv'],
        sample_rate=config['data']['sample_rate'],
        segment_duration=config['data']['segment_duration'],
        hop_duration=config['data']['hop_duration']
    )
    
    val_dataset = StructuredAudioDataset(
        csv_path=config['data']['val_csv'],
        sample_rate=config['data']['sample_rate'],
        segment_duration=config['data']['segment_duration'],
        hop_duration=config['data']['segment_duration']  # No overlap for validation
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 모델 생성
    print("Creating model...")
    
    # Structure encoder
    structure_encoder = StructureEncoder(
        embedding_dim=config['model']['structure_encoder']['embedding_dim'],
        hidden_dim=config['model']['structure_encoder']['hidden_dim'],
        num_layers=config['model']['structure_encoder']['num_layers']
    )
    
    # DiT
    dit = StructureConditionedDiT(
        base_dit_config=config['model']['dit'],
        structure_encoder=structure_encoder,
        conditioning_method=config['model']['conditioning_method']
    )
    
    # Pipeline
    model = StructureAwareStableAudioPipeline(
        vae_model_name=config['model']['vae_model'],
        text_encoder_name=config['model']['text_encoder'],
        structure_dit=dit
    )
    
    # 트레이너
    trainer = StructureAudioTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config['training']
    )
    
    # 학습 시작
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()



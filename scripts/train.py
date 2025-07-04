# scripts/train.py
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
import argparse
from pathlib import Path

def load_config(config_path: str) -> dict:
    """설정 파일 로드 및 검증"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 필수 섹션 확인
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config file must contain '{section}' section")
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Train Structure-Aware Stable Audio')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 설정 로드
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Resume 설정 오버라이드
    if args.resume:
        config['training']['resume_from_checkpoint'] = args.resume
    
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
    encoder_config = config['model']['structure_encoder']
    structure_encoder = StructureEncoder(
        embedding_dim=encoder_config['embedding_dim'],
        hidden_dim=encoder_config['hidden_dim'],
        num_layers=encoder_config['num_layers'],
        num_heads=encoder_config.get('num_heads', 8),
        dropout=encoder_config.get('dropout', 0.1),
        max_structures=encoder_config.get('max_structures', 50)
    )
    
    # 기존 Stable Audio transformer 설정 가져오기
    base_dit_config = {
        'hidden_size': 1024,
        'num_heads': 16,
        'depth': 24,
        'mlp_ratio': 4.0,
        'in_channels': 64,
    }
    
    # 설정 파일에서 오버라이드
    if 'dit' in config['model']:
        base_dit_config.update(config['model']['dit'])
    
    # Structure-conditioned DiT
    structure_dit = StructureConditionedDiT(
        base_dit_config=base_dit_config,
        structure_encoder=structure_encoder,
        conditioning_method=config['model']['conditioning_method']
    )
    
    # Pipeline
    model = StructureAwareStableAudioPipeline(
        model_id=config['model']['model_id'],
        structure_dit=structure_dit,
        device=config.get('device', 'cuda')
    )
    
    print(f"Model created with {sum(p.numel() for p in model.transformer.parameters() if p.requires_grad):,} trainable parameters")
    
    # 트레이너
    trainer = StructureAudioTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config['training']
    )
    
    # 체크포인트 로드 (있는 경우)
    if config['training'].get('resume_from_checkpoint'):
        checkpoint_path = config['training']['resume_from_checkpoint']
        if Path(checkpoint_path).exists():
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")
    
    # 학습 시작
    print("\nStarting training...")
    print("=" * 50)
    trainer.train()


if __name__ == "__main__":
    main()
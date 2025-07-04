# scripts/train.py - 개선된 버전
import sys
sys.path.append('..')
sys.path.append('.')

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.data.dataset import StructuredAudioDataset
from src.models.structure_encoder import StructureEncoder
from src.models.structure_dit import StructureConditionedDiT
from src.models.pipeline import StructureAwareStableAudioPipeline
from src.training.trainer import StructureAudioTrainer
import yaml
import argparse
from pathlib import Path
import json

def load_config(config_path: str) -> dict:
    """설정 파일 로드 및 검증"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 필수 섹션 확인
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config file must contain '{section}' section")
    
    return config

def validate_data_files(config: dict):
    """데이터 파일 존재 확인"""
    train_csv = Path(config['data']['train_csv'])
    val_csv = Path(config['data']['val_csv'])
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Training data not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation data not found: {val_csv}")
        
    # 데이터 샘플 수 확인
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    print(f"Data validation:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    
    if len(train_df) == 0:
        raise ValueError("Training dataset is empty!")
    if len(val_df) == 0:
        print("Warning: Validation dataset is empty!")
        
    return len(train_df), len(val_df)

def create_model(config: dict, device: str):
    """모델 생성"""
    print("Creating model components...")
    
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
    
    # 기본 DiT 설정
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
    
    print(f"DiT config: {base_dit_config}")
    
    # Structure-conditioned DiT
    structure_dit = StructureConditionedDiT(
        base_dit_config=base_dit_config,
        structure_encoder=structure_encoder,
        conditioning_method=config['model'].get('conditioning_method', 'cross_attention')
    )
    
    # Pipeline
    model = StructureAwareStableAudioPipeline(
        model_id=config['model'].get('model_id', 'stabilityai/stable-audio-open-1.0'),
        structure_dit=structure_dit,
        device=device
    )
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Structure-Aware Stable Audio')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test setup without actual training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detect if not specified')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 설정 로드
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # 디바이스 설정 오버라이드
    config['device'] = device
    
    # Resume 설정 오버라이드
    if args.resume:
        config['training']['resume_from_checkpoint'] = args.resume
    
    # 데이터 파일 검증
    print("Validating data files...")
    try:
        train_samples, val_samples = validate_data_files(config)
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        print("\nPlease run data preparation first:")
        print("python scripts/prepare_salami_data.py --salami_root <path> --audio_root <path>")
        return
    
    # 배치 크기 조정 (작은 데이터셋의 경우)
    original_batch_size = config['training']['batch_size']
    if train_samples < 100:
        config['training']['batch_size'] = min(original_batch_size, 2)
        print(f"Small dataset detected: reducing batch size to {config['training']['batch_size']}")
    
    # 데이터셋 생성
    print("Loading datasets...")
    try:
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
        
        print(f"✅ Datasets loaded:")
        print(f"  Train segments: {len(train_dataset)}")
        print(f"  Val segments: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # 샘플 데이터 테스트
    print("Testing sample data loading...")
    try:
        sample = train_dataset[0]
        print(f"✅ Sample data shape: {sample['audio'].shape}")
        print(f"  Structure sequence length: {len(sample['structure_sequence'])}")
        
    except Exception as e:
        print(f"❌ Sample data loading failed: {e}")
        return
    
    # 모델 생성
    try:
        model = create_model(config, device)
        print("✅ Model created successfully")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print("This might be due to:")
        print("- Missing Stable Audio model")
        print("- Insufficient VRAM")
        print("- Network connection issues")
        return
    
    # Dry run 모드
    if args.dry_run:
        print("\n🧪 Dry run mode - testing training step...")
        try:
            # 작은 배치로 테스트
            from torch.utils.data import DataLoader
            test_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=train_dataset.get_collate_fn()
            )
            
            batch = next(iter(test_loader))
            print(f"Test batch shape: {batch['audio'].shape}")
            
            # Forward pass 테스트
            model.train()
            with torch.cuda.amp.autocast() if device == 'cuda' else torch.no_grad():
                outputs = model.training_step(batch)
                
            print(f"✅ Training step test passed, loss: {outputs['loss'].item():.4f}")
            print("Setup is ready for training!")
            
        except Exception as e:
            print(f"❌ Dry run failed: {e}")
            print("Please check the error above and fix before training")
            return
        
        print("\n✅ All tests passed! Remove --dry-run to start actual training.")
        return
    
    # 트레이너 생성
    print("Setting up trainer...")
    try:
        trainer = StructureAudioTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config['training']
        )
        print("✅ Trainer created successfully")
        
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return
    
    # 체크포인트 로드 (있는 경우)
    resume_path = config['training'].get('resume_from_checkpoint')
    if resume_path and Path(resume_path).exists():
        try:
            trainer.load_checkpoint(resume_path)
            print(f"✅ Resumed from checkpoint: {resume_path}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
    elif resume_path:
        print(f"⚠️  Checkpoint not found: {resume_path}")
        print("Starting training from scratch...")
    
    # 설정 요약 출력
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Model: Structure-Aware Stable Audio")
    print(f"Conditioning: {config['model'].get('conditioning_method', 'cross_attention')}")
    print(f"")
    print(f"Data:")
    print(f"  Train samples: {len(train_dataset)} segments")
    print(f"  Val samples: {len(val_dataset)} segments")
    print(f"  Sample rate: {config['data']['sample_rate']} Hz")
    print(f"  Segment duration: {config['data']['segment_duration']} s")
    print(f"")
    print(f"Training:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Accumulate batches: {config['training'].get('accumulate_grad_batches', 1)}")
    print(f"  Effective batch size: {config['training']['batch_size'] * config['training'].get('accumulate_grad_batches', 1)}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Mixed precision: {config['training'].get('use_amp', False)}")
    print(f"")
    print(f"Hardware:")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)
    
    # 메모리 사용량 추정
    if device == "cuda":
        print(f"\n💾 Memory Estimation:")
        batch_size = config['training']['batch_size']
        segment_duration = config['data']['segment_duration']
        sample_rate = config['data']['sample_rate']
        
        # 대략적인 메모리 사용량 계산
        audio_size = batch_size * 2 * segment_duration * sample_rate * 4  # float32
        estimated_total = (audio_size + audio_size * 0.1) / 1e9  # 10% 오버헤드
        
        print(f"  Estimated per-batch: ~{estimated_total:.1f} GB")
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if estimated_total > available_memory * 0.8:
            print(f"  ⚠️  High memory usage expected!")
            print(f"  Consider reducing batch size or segment duration")
    
    # 학습 시작 확인
    print(f"\n🚀 Ready to start training!")
    print(f"Logs will be saved to: {config['training'].get('log_dir', './logs')}")
    print(f"Checkpoints will be saved to: {config['training'].get('checkpoint_dir', './checkpoints')}")
    
    if config['training'].get('use_wandb', False):
        print(f"Weights & Biases project: {config['training'].get('wandb_project', 'structure-audio')}")
    
    try:
        # 학습 시작
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        print("Saving checkpoint...")
        try:
            trainer.save_checkpoint()
            print("✅ Checkpoint saved successfully")
        except:
            print("❌ Failed to save checkpoint")
            
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        
        # 긴급 체크포인트 저장 시도
        try:
            print("Attempting emergency checkpoint save...")
            trainer.save_checkpoint()
            print("✅ Emergency checkpoint saved")
        except:
            print("❌ Emergency checkpoint save failed")
    
    finally:
        print("\n" + "="*60)
        print("TRAINING SESSION ENDED")
        print("="*60)
        
        # 최종 통계
        if hasattr(trainer, 'best_val_loss') and trainer.best_val_loss < float('inf'):
            print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
        print("Check the following directories:")
        print(f"  Logs: {config['training'].get('log_dir', './logs')}")
        print(f"  Checkpoints: {config['training'].get('checkpoint_dir', './checkpoints')}")
        
        if config['training'].get('use_wandb', False):
            print(f"  Wandb: https://wandb.ai/{config['training'].get('wandb_project', 'structure-audio')}")


if __name__ == "__main__":
    main()
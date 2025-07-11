# src_sagemaker/train_sagemaker.py
"""
SageMaker 컨테이너에서 실행되는 훈련 스크립트
원본 train.py를 SageMaker 환경에 맞게 수정
"""

import os
import sys
import json
import argparse
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# SageMaker 환경 변수
SM_CHANNEL_TRAIN = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
SM_CHANNEL_VAL = os.environ.get('SM_CHANNEL_VAL', '/opt/ml/input/data/val')
SM_CHANNEL_CONFIG = os.environ.get('SM_CHANNEL_CONFIG', '/opt/ml/input/data/config')
SM_CHANNEL_CHECKPOINT = os.environ.get('SM_CHANNEL_CHECKPOINT', '/opt/ml/input/data/checkpoint')
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')

# 로컬 체크포인트 디렉토리 (S3와 동기화됨)
LOCAL_CHECKPOINT_DIR = '/opt/ml/checkpoints'

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 필요한 모듈들을 src_sagemaker 디렉토리에 복사해야 함
from data.dataset import StructuredAudioDataset
from models.structure_encoder import StructureEncoder
from models.structure_dit import StructureConditionedDiT
from models.pipeline import StructureAwareStableAudioPipeline
from training.trainer import StructureAudioTrainer


def setup_sagemaker_paths(config):
    """SageMaker 경로를 로컬 경로로 매핑"""
    # CSV 파일 경로 수정
    config['data']['train_csv'] = os.path.join(SM_CHANNEL_TRAIN, 'train.csv')
    config['data']['val_csv'] = os.path.join(SM_CHANNEL_VAL, 'val.csv')
    
    # 체크포인트 디렉토리 설정
    config['training']['checkpoint_dir'] = LOCAL_CHECKPOINT_DIR
    os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)
    
    # 로그 디렉토리
    config['training']['log_dir'] = os.path.join(SM_OUTPUT_DATA_DIR, 'logs')
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Resume 체크포인트 확인
    if SM_CHANNEL_CHECKPOINT and os.path.exists(SM_CHANNEL_CHECKPOINT):
        checkpoint_files = [f for f in os.listdir(SM_CHANNEL_CHECKPOINT) if f.endswith('.pt')]
        if checkpoint_files:
            # best_model.pt를 우선적으로 찾고, 없으면 가장 최근 체크포인트 사용
            if 'best_model.pt' in checkpoint_files:
                config['training']['resume_from_checkpoint'] = os.path.join(SM_CHANNEL_CHECKPOINT, 'best_model.pt')
            else:
                checkpoint_files.sort()
                config['training']['resume_from_checkpoint'] = os.path.join(SM_CHANNEL_CHECKPOINT, checkpoint_files[-1])
            print(f"Found checkpoint to resume: {config['training']['resume_from_checkpoint']}")
    
    return config


def save_model_artifacts(model, config):
    """모델을 SageMaker 형식으로 저장"""
    print(f"Saving model artifacts to {SM_MODEL_DIR}")
    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    
    # 최종 모델 저장
    model_path = os.path.join(SM_MODEL_DIR, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    
    # 추론을 위한 코드와 설정도 함께 저장
    import shutil
    
    # 필요한 소스 코드 복사
    code_dir = os.path.join(SM_MODEL_DIR, 'code')
    os.makedirs(code_dir, exist_ok=True)
    
    # 모델 관련 파일들 복사
    for module in ['models', 'data', 'utils']:
        src_path = os.path.join(os.path.dirname(__file__), module)
        dst_path = os.path.join(code_dir, module)
        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    
    # 추론 스크립트 생성
    inference_script = '''
import torch
import json
from models.structure_encoder import StructureEncoder
from models.structure_dit import StructureConditionedDiT
from models.pipeline import StructureAwareStableAudioPipeline

def model_fn(model_dir):
    """SageMaker 추론을 위한 모델 로드"""
    model_path = os.path.join(model_dir, 'model.pt')
    checkpoint = torch.load(model_path, map_location='cpu')
    
    config = checkpoint['config']
    
    # 모델 재생성
    structure_encoder = StructureEncoder(**config['model']['structure_encoder'])
    structure_dit = StructureConditionedDiT(
        base_dit_config=config['model'].get('dit', {}),
        structure_encoder=structure_encoder,
        conditioning_method=config['model'].get('conditioning_method', 'cross_attention')
    )
    
    model = StructureAwareStableAudioPipeline(
        model_id=config['model'].get('model_id', 'stabilityai/stable-audio-open-1.0'),
        structure_dit=structure_dit
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def input_fn(request_body, content_type='application/json'):
    """입력 데이터 처리"""
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """예측 수행"""
    with torch.no_grad():
        audio = model.generate(
            prompt=input_data['prompt'],
            structure_sequence=input_data['structure_sequence'],
            duration=input_data.get('duration'),
            num_inference_steps=input_data.get('num_inference_steps', 50),
            guidance_scale=input_data.get('guidance_scale', 7.0)
        )
    return audio

def output_fn(prediction, accept='application/json'):
    """출력 데이터 처리"""
    # 실제로는 오디오를 base64 인코딩하거나 S3에 저장하고 URL 반환
    return json.dumps({
        'status': 'success',
        'audio_shape': list(prediction.shape)
    })
'''
    
    with open(os.path.join(code_dir, 'inference.py'), 'w') as f:
        f.write(inference_script)
    
    print(f"Model artifacts saved successfully")


def create_model(config, device):
    """모델 생성 (원본 train.py와 동일)"""
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
    
    if 'dit' in config['model']:
        base_dit_config.update(config['model']['dit'])
    
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
    
    return model


def train():
    """메인 훈련 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default=os.path.join(SM_CHANNEL_CONFIG, 'train_config.json'))
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # SageMaker 경로 설정
    config = setup_sagemaker_paths(config)
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
        hop_duration=config['data']['segment_duration']
    )
    
    print(f"Train segments: {len(train_dataset)}")
    print(f"Val segments: {len(val_dataset)}")
    
    # 모델 생성
    model = create_model(config, device)
    print("Model created successfully")
    
    # 트레이너 생성
    trainer = StructureAudioTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config['training']
    )
    
    # 체크포인트 로드
    if config['training'].get('resume_from_checkpoint'):
        try:
            trainer.load_checkpoint(config['training']['resume_from_checkpoint'])
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")
    
    # 훈련 시작
    print("\nStarting training...")
    try:
        trainer.train()
        
        # 최종 모델 저장
        save_model_artifacts(model, config)
        
        # 최고 성능 체크포인트를 모델 디렉토리에도 복사
        best_checkpoint = os.path.join(LOCAL_CHECKPOINT_DIR, 'best_model.pt')
        if os.path.exists(best_checkpoint):
            import shutil
            shutil.copy(best_checkpoint, os.path.join(SM_MODEL_DIR, 'best_model.pt'))
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러가 발생해도 현재까지의 체크포인트는 저장
        try:
            trainer.save_checkpoint()
            print("Emergency checkpoint saved")
        except:
            pass
        
        raise e


if __name__ == "__main__":
    train()
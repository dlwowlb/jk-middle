# sagemaker_training_job.py
"""
AWS SageMaker Studio Training Job 실행 스크립트
스팟 인스턴스를 사용하여 비용 절약
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from datetime import datetime
import json
import os
from pathlib import Path
import argparse
import time

class SageMakerStructureAudioTrainer:
    """SageMaker를 위한 Structure-Aware Audio 훈련 관리자"""
    
    def __init__(self, 
                 role: str = None,
                 region: str = 'ap-southeast-2',
                 bucket: str = None):
        
        self.session = sagemaker.Session()
        self.region = region
        
        # IAM Role (SageMaker 실행 권한)
        if role is None:
            self.role = sagemaker.get_execution_role()
        else:
            self.role = role
            
        # S3 버킷 설정
        if bucket is None:
            self.bucket = self.session.default_bucket()
        else:
            self.bucket = bucket
            
        self.s3_prefix = 'structure-aware-audio'
        
        print(f"Region: {self.region}")
        print(f"Role: {self.role}")
        print(f"Bucket: {self.bucket}")
        
    def prepare_data(self, local_data_dir: str):
        """로컬 데이터를 S3에 업로드"""
        print("Uploading data to S3...")
        
        # 훈련 데이터 업로드
        train_data_path = self.session.upload_data(
            path=os.path.join(local_data_dir, 'train.csv'),
            bucket=self.bucket,
            key_prefix=f'{self.s3_prefix}/data'
        )
        
        val_data_path = self.session.upload_data(
            path=os.path.join(local_data_dir, 'val.csv'),
            bucket=self.bucket,
            key_prefix=f'{self.s3_prefix}/data'
        )
        
        # 오디오 파일들 업로드 (대용량일 수 있으므로 진행상황 표시)
        audio_s3_path = f's3://{self.bucket}/{self.s3_prefix}/audio/'
        
        print(f"Uploading audio files to {audio_s3_path}")
        # AWS CLI를 사용하여 병렬 업로드 (더 빠름)
        os.system(f"aws s3 sync {local_data_dir}/audio/ {audio_s3_path} --exclude '*.txt'")
        
        # SALAMI 어노테이션 업로드
        salami_s3_path = f's3://{self.bucket}/{self.s3_prefix}/salami/'
        print(f"Uploading SALAMI annotations to {salami_s3_path}")
        os.system(f"aws s3 sync {local_data_dir}/salami-data-public/ {salami_s3_path}")
        
        return {
            'train': train_data_path,
            'val': val_data_path,
            'audio': audio_s3_path,
            'salami': salami_s3_path
        }
    
    def create_training_config(self):
        """훈련 설정 생성"""
        config = {
            # 데이터 설정
            'data': {
                'train_csv': '/opt/ml/input/data/train/train.csv',
                'val_csv': '/opt/ml/input/data/val/val.csv',
                'audio_root': '/opt/ml/input/data/audio',
                'salami_root': '/opt/ml/input/data/salami',
                'sample_rate': 44100,
                'segment_duration': 20.0,
                'hop_duration': 8.0
            },
            
            # 모델 설정
            'model': {
                'model_id': 'stabilityai/stable-audio-open-1.0',
                'conditioning_method': 'cross_attention',
                'structure_encoder': {
                    'embedding_dim': 768,
                    'hidden_dim': 512,
                    'num_layers': 4,
                    'num_heads': 8,
                    'dropout': 0.1,
                    'max_structures': 50
                },
                'dit': {
                    'hidden_size': 768,
                    'num_heads': 16,
                    'depth': 24,
                    'mlp_ratio': 4.0,
                    'in_channels': 64
                }
            },
            
            # 훈련 설정 (스팟 인스턴스용 최적화)
            'training': {
                'batch_size': 2,  # GPU 메모리에 맞게 조정
                'num_epochs': 50,
                'learning_rate': 5e-5,
                'min_lr': 1e-6,
                'weight_decay': 0.01,
                'gradient_clip': 1.0,
                'accumulate_grad_batches': 8,
                'use_amp': True,  # Mixed precision
                'num_workers': 4,
                'log_interval': 50,
                'val_interval': 1,
                'checkpoint_interval': 5,
                'max_checkpoints': 3,
                'checkpoint_dir': '/opt/ml/checkpoints',
                'log_dir': '/opt/ml/output/tensorboard',
                'use_wandb': False  # SageMaker CloudWatch 사용
            }
        }
        
        # S3에 설정 파일 업로드
        config_path = '/tmp/train_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        s3_config_path = self.session.upload_data(
            path=config_path,
            bucket=self.bucket,
            key_prefix=f'{self.s3_prefix}/config'
        )
        
        return s3_config_path
    
    def create_training_job(self, 
                           data_paths: dict,
                           config_path: str,
                           instance_type: str = 'ml.g4dn.xlarge',
                           spot_instance: bool = True,
                           max_runtime: int = 48*3600,  # 48시간
                           checkpoint_s3_path: str = None):
        """SageMaker 훈련 작업 생성"""
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        job_name = f'structure-audio-{timestamp}'
        
        # 체크포인트 S3 경로
        if checkpoint_s3_path is None:
            checkpoint_s3_path = f's3://{self.bucket}/{self.s3_prefix}/checkpoints/{job_name}'
        
        # 하이퍼파라미터
        hyperparameters = {
            'config': '/opt/ml/input/data/config/train_config.json',
            'resume': '/opt/ml/input/data/checkpoint/best_model.pt' if checkpoint_s3_path else ''
        }
        
        # PyTorch Estimator 생성
        estimator = PyTorch(
            entry_point='train_sagemaker.py',  # SageMaker용 진입점
            source_dir='./src_sagemaker',  # 소스 코드 디렉토리
            role=self.role,
            instance_type=instance_type,
            instance_count=1,
            framework_version='2.0.0',
            py_version='py310',
            
            # 스팟 인스턴스 설정
            use_spot_instances=spot_instance,
            max_run=max_runtime,
            max_wait=max_runtime + 600 if spot_instance else None,
            
            # 체크포인트 설정 (스팟 인스턴스 중단 시 복구용)
            checkpoint_s3_uri=checkpoint_s3_path,
            checkpoint_local_path='/opt/ml/checkpoints',
            
            # 출력 설정
            output_path=f's3://{self.bucket}/{self.s3_prefix}/output/{job_name}',
            
            # 메트릭 정의 (CloudWatch 모니터링)
            metric_definitions=[
                {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9.]+)'},
                {'Name': 'val:loss', 'Regex': 'Val Loss: ([0-9.]+)'},
                {'Name': 'lr', 'Regex': 'Learning rate: ([0-9.e-]+)'}
            ],
            
            # 환경 변수
            environment={
                'PYTHONUNBUFFERED': '1',
                'TORCH_HOME': '/opt/ml/input/data/torch_cache',
                'HF_HOME': '/opt/ml/input/data/hf_cache'
            },
            
            # 볼륨 크기 (모델 다운로드 공간 필요)
            volume_size=100,
            
            # 디버깅
            debugger_hook_config=False,  # 비용 절약을 위해 비활성화
            
            hyperparameters=hyperparameters
        )
        
        # 입력 데이터 설정
        inputs = {
            'train': TrainingInput(data_paths['train'], content_type='text/csv'),
            'val': TrainingInput(data_paths['val'], content_type='text/csv'),
            'audio': TrainingInput(data_paths['audio'], s3_data_type='S3Prefix'),
            'salami': TrainingInput(data_paths['salami'], s3_data_type='S3Prefix'),
            'config': TrainingInput(config_path, content_type='application/json')
        }
        
        # 이전 체크포인트가 있으면 추가
        if checkpoint_s3_path and self._check_s3_exists(checkpoint_s3_path):
            inputs['checkpoint'] = TrainingInput(
                checkpoint_s3_path, 
                s3_data_type='S3Prefix'
            )
        
        # 훈련 시작
        print(f"Starting training job: {job_name}")
        print(f"Instance type: {instance_type}")
        print(f"Spot instance: {spot_instance}")
        print(f"Max runtime: {max_runtime/3600:.1f} hours")
        
        estimator.fit(inputs, job_name=job_name, wait=False)
        
        return estimator, job_name
    
    def _check_s3_exists(self, s3_path: str) -> bool:
        """S3 경로 존재 확인"""
        s3 = boto3.client('s3')
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False
    
    def monitor_training(self, job_name: str):
        """훈련 작업 모니터링"""
        sm_client = boto3.client('sagemaker')
        
        print(f"\nMonitoring training job: {job_name}")
        print("Press Ctrl+C to stop monitoring (training will continue)")
        
        try:
            while True:
                response = sm_client.describe_training_job(
                    TrainingJobName=job_name
                )
                
                status = response['TrainingJobStatus']
                secondary_status = response.get('SecondaryStatus', '')
                
                print(f"\rStatus: {status} - {secondary_status}", end='')
                
                if status in ['Completed', 'Failed', 'Stopped']:
                    print(f"\nTraining job {status}")
                    
                    if status == 'Completed':
                        print(f"Model artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
                    elif status == 'Failed':
                        print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
                    
                    break
                
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\nStopped monitoring (training continues in background)")
    
    def estimate_cost(self, instance_type: str, duration_hours: float, spot: bool = True):
        """훈련 비용 추정"""
        # 대략적인 시간당 가격 (실제 가격은 리전과 시점에 따라 다름)
        on_demand_prices = {
            'ml.g4dn.xlarge': 0.736,    # 1 GPU, 16GB
            'ml.g4dn.2xlarge': 1.056,   # 1 GPU, 32GB
            'ml.g4dn.4xlarge': 1.696,   # 1 GPU, 64GB
            'ml.g5.xlarge': 1.408,       # 1 A10G GPU, 16GB
            'ml.g5.2xlarge': 1.632,      # 1 A10G GPU, 32GB
            'ml.p3.2xlarge': 3.825,      # 1 V100 GPU
        }
        
        if instance_type not in on_demand_prices:
            return None
            
        on_demand_cost = on_demand_prices[instance_type] * duration_hours
        
        # 스팟 인스턴스는 보통 70-90% 할인
        spot_cost = on_demand_cost * 0.3 if spot else on_demand_cost
        
        print(f"\nEstimated training cost for {instance_type}:")
        print(f"Duration: {duration_hours:.1f} hours")
        print(f"On-demand cost: ${on_demand_cost:.2f}")
        if spot:
            print(f"Spot cost (estimated): ${spot_cost:.2f}")
            print(f"Savings: ${on_demand_cost - spot_cost:.2f} ({(1-spot_cost/on_demand_cost)*100:.0f}%)")
        
        return spot_cost


def main():
    parser = argparse.ArgumentParser(description='Run SageMaker training job for Structure-Aware Audio')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Local directory containing processed data')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                       help='SageMaker instance type')
    parser.add_argument('--spot', action='store_true', default=True,
                       help='Use spot instances (default: True)')
    parser.add_argument('--max-runtime', type=int, default=48,
                       help='Maximum runtime in hours (default: 48)')
    parser.add_argument('--role', type=str, default=None,
                       help='IAM role ARN (auto-detect if not specified)')
    parser.add_argument('--bucket', type=str, default=None,
                       help='S3 bucket (default bucket if not specified)')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor training job after starting')
    
    args = parser.parse_args()
    
    # SageMaker 훈련 관리자 생성
    trainer = SageMakerStructureAudioTrainer(
        role=args.role,
        bucket=args.bucket
    )
    
    # 비용 추정
    trainer.estimate_cost(
        args.instance_type, 
        args.max_runtime, 
        args.spot
    )
    
    # 계속할지 확인
    response = input("\nContinue with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        return
    
    # 데이터 준비
    print("\nPreparing data...")
    data_paths = trainer.prepare_data(args.data_dir)
    
    # 설정 파일 생성
    config_path = trainer.create_training_config()
    
    # 훈련 작업 생성
    estimator, job_name = trainer.create_training_job(
        data_paths=data_paths,
        config_path=config_path,
        instance_type=args.instance_type,
        spot_instance=args.spot,
        max_runtime=args.max_runtime * 3600
    )
    
    print(f"\nTraining job started: {job_name}")
    print(f"Console URL: https://console.aws.amazon.com/sagemaker/home?region={trainer.region}#/jobs/{job_name}")
    
    # 모니터링
    if args.monitor:
        trainer.monitor_training(job_name)


if __name__ == "__main__":
    main()
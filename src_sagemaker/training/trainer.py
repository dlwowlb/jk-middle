# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Any

class StructureAudioTrainer:
    """구조 인식 오디오 생성 모델 트레이너"""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset,
                 val_dataset,
                 config: Dict[str, Any]):
        
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델을 device로 이동
        self.model.to(self.device)
        
        # 학습 설정 값들을 먼저 설정 (다른 메서드에서 사용하기 전에)
        self.num_epochs = int(config.get('num_epochs', 100))
        self.log_interval = int(config.get('log_interval', 100))
        self.val_interval = int(config.get('val_interval', 1))
        self.checkpoint_interval = int(config.get('checkpoint_interval', 5))
        self.gradient_clip = float(config.get('gradient_clip', 1.0))
        self.accumulate_grad_batches = int(config.get('accumulate_grad_batches', 1))
        
        # 데이터 로더
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(config.get('batch_size', 4)),
            shuffle=True,
            num_workers=int(config.get('num_workers', 4)),
            pin_memory=True,
            collate_fn=train_dataset.get_collate_fn()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=int(config.get('batch_size', 4)),
            shuffle=False,
            num_workers=int(config.get('num_workers', 4)),
            pin_memory=True,
            collate_fn=val_dataset.get_collate_fn()
        )
        
        # 옵티마이저 설정 (num_epochs 설정 후)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 로깅
        self.writer = SummaryWriter(config.get('log_dir', './logs'))
        if config.get('use_wandb', True):
            project_name = config.get('wandb_project', 'structure-aware-audio')
            wandb.init(project=project_name, config=config)
            
        # 체크포인트
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.current_epoch = 0
        
    def _setup_optimizer(self):
        """옵티마이저 설정"""
        # Transformer 파라미터만 학습 (VAE와 text encoder는 frozen)
        transformer_params = self.model.transformer.parameters()
        
        learning_rate = float(self.config.get('learning_rate', 1e-4))
        weight_decay = float(self.config.get('weight_decay', 0.01))
        
        optimizer = optim.AdamW(
            transformer_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """스케줄러 설정"""
        min_lr = float(self.config.get('min_lr', 1e-6))
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=min_lr
        )
        return scheduler
    
    def train_epoch(self):
        """한 에폭 학습"""
        self.model.train()
        epoch_loss = 0
        accumulation_steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model.training_step(batch)
                    loss = outputs['loss']
            else:
                outputs = self.model.training_step(batch)
                loss = outputs['loss']
            
            # Gradient accumulation으로 나누기
            loss = loss / self.accumulate_grad_batches
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_steps += 1
            
            # Gradient accumulation
            if accumulation_steps % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.transformer.parameters(),
                        self.gradient_clip
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_steps = 0
            
            # 로깅 (실제 loss 값으로)
            actual_loss = loss.item() * self.accumulate_grad_batches
            epoch_loss += actual_loss
            pbar.set_postfix({'loss': f'{actual_loss:.4f}'})
            
            if self.global_step % self.log_interval == 0:
                self.writer.add_scalar('train/loss', actual_loss, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'train/loss': actual_loss,
                        'train/lr': self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
            
            self.global_step += 1
            
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """검증"""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.training_step(batch)
                else:
                    outputs = self.model.training_step(batch)
                    
                val_loss += outputs['loss'].item()
                num_batches += 1
                
        val_loss = val_loss / num_batches if num_batches > 0 else 0
        
        self.writer.add_scalar('val/loss', val_loss, self.global_step)
        if self.config.get('use_wandb', True):
            wandb.log({'val/loss': val_loss}, step=self.global_step)
            
        return val_loss
    
    def save_checkpoint(self, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': getattr(self, 'best_val_loss', float('inf'))
        }
        
        # AMP scaler 상태도 저장
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 일반 체크포인트
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # 베스트 체크포인트
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
            
        # 최근 N개만 유지
        max_checkpoints = int(self.config.get('max_checkpoints', 5))
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > max_checkpoints:
            for ckpt in checkpoints[:-max_checkpoints]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # 다음 에폭부터 시작
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")
        print(f"Best validation loss so far: {self.best_val_loss:.4f}")
    
    def train(self):
        """전체 학습 루프"""
        self.best_val_loss = float('inf')
        
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Accumulate grad batches: {self.accumulate_grad_batches}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.accumulate_grad_batches}")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # 학습
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch}/{self.num_epochs-1} - Train Loss: {train_loss:.4f}")
            
            # 검증
            if epoch % self.val_interval == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}/{self.num_epochs-1} - Val Loss: {val_loss:.4f}")
                
                # 베스트 모델 저장
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
                
                # 체크포인트 저장
                if epoch % self.checkpoint_interval == 0 or is_best:
                    self.save_checkpoint(is_best)
            
            # 스케줄러 업데이트
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.2e}")
            
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # 학습 종료 시 wandb 종료
        if self.config.get('use_wandb', True):
            wandb.finish()
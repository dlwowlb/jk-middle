## 6. 학습 스크립트 (src/training/trainer.py)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from pathlib import Path
import json

class StructureAudioTrainer:
    """구조 인식 오디오 생성 모델 트레이너"""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset,
                 val_dataset,
                 config: dict):
        
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 데이터 로더
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            collate_fn=train_dataset.get_collate_fn()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            collate_fn=val_dataset.get_collate_fn()
        )
        
        # 옵티마이저 설정
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 로깅
        self.writer = SummaryWriter(config['log_dir'])
        if config.get('use_wandb', True):
            project = config.get('project_name', 'structure-aware-stable-audio')
            wandb.init(project=project, config=config)
            
        # 체크포인트
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.current_epoch = 0
        
    def _setup_optimizer(self):
        """옵티마이저 설정"""
        # DiT 파라미터만 학습
        dit_params = self.model.dit.parameters()
        
        optimizer = optim.AdamW(
            dit_params,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self.config['weight_decay']
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """스케줄러 설정"""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['min_lr']
        )
        return scheduler
    
    def train_epoch(self):
        """한 에폭 학습"""
        self.model.train()
        epoch_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            outputs = self.model.training_step(batch)
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.dit.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # 로깅
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if self.global_step % self.config['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                if self.config.get('use_wandb', True):
                    wandb.log({'train/loss': loss.item()}, step=self.global_step)
            
            self.global_step += 1
            
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """검증"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                outputs = self.model.training_step(batch)
                val_loss += outputs['loss'].item()
                
        val_loss /= len(self.val_loader)
        
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
            'config': self.config
        }
        
        # 일반 체크포인트
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 베스트 체크포인트
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # 최근 5개만 유지
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 5:
            for ckpt in checkpoints[:-5]:
                ckpt.unlink()
    
    def train(self):
        """전체 학습 루프"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # 학습
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # 검증
            if epoch % self.config['val_interval'] == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # 베스트 모델 저장
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    
                self.save_checkpoint(is_best)
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
        print("Training completed!")
        
        

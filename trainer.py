import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np
from datetime import datetime
from models.latent_diffusion import LatentDiffusionModel
from data.dataset import LipSyncDataset
from utils.metrics import calculate_sync_score, calculate_l2_loss
from utils.helpers import save_checkpoint, load_checkpoint, log_metrics

class Trainer:
    def __init__(self, config_path="configs/model_config.yaml"):
        # 加载配置
        self.config = yaml.safe_load(open(config_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join("logs", self.exp_name)
        self.checkpoint_dir = os.path.join("checkpoints", self.exp_name)
        
        # 创建必要目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 初始化模型
        self.model = LatentDiffusionModel(self.config).to(self.device)
        
        # 数据加载
        self._init_dataloaders()
        
        # 优化器和调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=1e-6
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device == "cuda")
        
        # 记录最佳指标
        self.best_val_score = float('inf')

    def _init_dataloaders(self):
        """初始化数据加载器"""
        self.train_dataset = LipSyncDataset(
            data_root=self.config['data']['root'],
            split="train",
            config_path="configs/model_config.yaml"
        )
        
        self.val_dataset = LipSyncDataset(
            data_root=self.config['data']['root'],
            split="val",
            config_path="configs/model_config.yaml"
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

    def train_one_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_sync_loss = 0.0
        total_temporal_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            face = batch['face'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
                # 生成随机时间步
                t = torch.randint(0, self.config['model']['diffusion_steps'], (video.shape[0],), device=self.device).float()
                
                # 前向传播
                pred = self.model(video, t, audio, face)
                
                # 计算损失
                diffusion_loss = F.l1_loss(pred, video)
                sync_loss = self.model.syncnet_loss(pred, audio)
                temporal_loss = self.model.temporal_consistency_loss(pred)
                
                # 总损失
                loss = (
                    diffusion_loss + 
                    self.config['training']['sync_loss_weight'] * sync_loss +
                    self.config['training']['temporal_loss_weight'] * temporal_loss
                )
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.config['training']['clip_grad_norm'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['clip_grad_norm']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录损失
            total_loss += diffusion_loss.item()
            total_sync_loss += sync_loss.item()
            total_temporal_loss += temporal_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "diff_loss": f"{diffusion_loss.item():.4f}",
                "sync_loss": f"{sync_loss.item():.4f}",
                "temp_loss": f"{temporal_loss.item():.4f}"
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_sync_loss = total_sync_loss / len(self.train_loader)
        avg_temporal_loss = total_temporal_loss / len(self.train_loader)
        
        return {
            "diffusion_loss": avg_loss,
            "sync_loss": avg_sync_loss,
            "temporal_loss": avg_temporal_loss,
            "total_loss": avg_loss + avg_sync_loss + avg_temporal_loss
        }

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_sync_score = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                video = batch['video'].to(self.device)
                audio = batch['audio'].to(self.device)
                face = batch['face'].to(self.device)
                
                # 生成随机时间步
                t = torch.randint(0, self.config['model']['diffusion_steps'], (video.shape[0],), device=self.device).float()
                
                # 前向传播
                pred = self.model(video, t, audio, face)
                
                # 计算损失
                loss = F.l1_loss(pred, video)
                sync_score = calculate_sync_score(pred, audio, self.model.syncnet_head)
                
                total_loss += loss.item()
                total_sync_score += sync_score.item()
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_sync_score = total_sync_score / len(self.val_loader)
        
        return {
            "val_loss": avg_loss,
            "val_sync_score": avg_sync_score
        }

    def train(self, resume_from=None):
        """完整训练流程"""
        start_epoch = 0
        
        # 加载 checkpoint
        if resume_from is not None and os.path.exists(resume_from):
            start_epoch = load_checkpoint(
                resume_from, 
                self.model, 
                self.optimizer, 
                self.scheduler
            )
            print(f"Resumed from checkpoint, starting at epoch {start_epoch+1}")
        
        # 训练循环
        for epoch in range(start_epoch, self.config['training']['epochs']):
            # 训练一个 epoch
            train_metrics = self.train_one_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印指标
            print(f"\nEpoch {epoch+1} metrics:")
            print(f"Train loss: {train_metrics['total_loss']:.4f}")
            print(f"Val loss: {val_metrics['val_loss']:.4f}")
            print(f"Sync score: {val_metrics['val_sync_score']:.4f}")
            
            # 记录指标
            log_metrics(
                {**train_metrics,** val_metrics},
                os.path.join(self.log_dir, "metrics.csv")
            )
            
            # 保存 checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            save_checkpoint(
                checkpoint_path,
                epoch,
                self.model,
                self.optimizer,
                self.scheduler,
                train_metrics,
                val_metrics
            )
            
            # 保存最佳模型
            if val_metrics['val_loss'] < self.best_val_score:
                self.best_val_score = val_metrics['val_loss']
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                    epoch,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    train_metrics,
                    val_metrics
                )
        
        print("Training completed!")
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from enum import Enum

class MotionType(Enum):
    """运动类型枚举"""
    TAICHI = "taichi"
    BOXING = "boxing" 
    DANCE = "dance"
    KARATE = "karate"
    YOGA = "yoga"
    GYMNASTICS = "gymnastics"

class MotionEncoder(nn.Module):
    """
    运动编码器 - 将不同类型的运动映射到统一的潜在空间
    
    核心功能：
    1. 提取运动特征（姿态、速度、节奏等）
    2. 学习跨模态的运动表示
    3. 支持运动相似性计算
    """
    
    def __init__(self, 
                 motion_dim: int = 256,  # 运动数据维度
                 latent_dim: int = 128,  # 潜在空间维度
                 num_motion_types: int = 6,  # 运动类型数量
                 hidden_dims: List[int] = [512, 256]):
        super().__init__()
        
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.num_motion_types = num_motion_types
        
        # 运动特征提取网络
        self.feature_extractor = self._build_feature_extractor(motion_dim, hidden_dims)
        
        # 运动类型嵌入
        self.motion_type_embedding = nn.Embedding(num_motion_types, 64)
        
        # 变分编码器 (VAE)
        self.encoder_mean = nn.Linear(hidden_dims[-1] + 64, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1] + 64, latent_dim)
        
        # 解码器
        self.decoder = self._build_decoder(latent_dim + 64, hidden_dims[::-1], motion_dim)
        
        # 运动分类器（用于辅助训练）
        self.motion_classifier = nn.Linear(latent_dim, num_motion_types)
        
    def _build_feature_extractor(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """构建特征提取网络"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _build_decoder(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
        """构建解码器网络"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def encode(self, motion_data: torch.Tensor, motion_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码运动数据到潜在空间
        
        Args:
            motion_data: (batch_size, motion_dim) 运动数据
            motion_type: (batch_size,) 运动类型索引
            
        Returns:
            mean: (batch_size, latent_dim) 潜在表示均值
            logvar: (batch_size, latent_dim) 潜在表示方差
        """
        # 提取运动特征
        motion_features = self.feature_extractor(motion_data)
        
        # 获取运动类型嵌入
        type_embedding = self.motion_type_embedding(motion_type)
        
        # 拼接特征
        combined_features = torch.cat([motion_features, type_embedding], dim=-1)
        
        # 变分编码
        mean = self.encoder_mean(combined_features)
        logvar = self.encoder_logvar(combined_features)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, latent_code: torch.Tensor, motion_type: torch.Tensor) -> torch.Tensor:
        """
        从潜在代码解码运动数据
        
        Args:
            latent_code: (batch_size, latent_dim) 潜在代码
            motion_type: (batch_size,) 运动类型索引
            
        Returns:
            reconstructed_motion: (batch_size, motion_dim) 重构的运动数据
        """
        type_embedding = self.motion_type_embedding(motion_type)
        decoder_input = torch.cat([latent_code, type_embedding], dim=-1)
        return self.decoder(decoder_input)
    
    def forward(self, motion_data: torch.Tensor, motion_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码
        mean, logvar = self.encode(motion_data, motion_type)
        latent_code = self.reparameterize(mean, logvar)
        
        # 解码
        reconstructed = self.decode(latent_code, motion_type)
        
        # 运动分类
        motion_pred = self.motion_classifier(latent_code)
        
        return {
            'latent_code': latent_code,
            'mean': mean,
            'logvar': logvar,
            'reconstructed': reconstructed,
            'motion_pred': motion_pred
        }
    
    def compute_motion_similarity(self, latent1: torch.Tensor, latent2: torch.Tensor) -> torch.Tensor:
        """计算两个运动在潜在空间中的相似度"""
        return F.cosine_similarity(latent1, latent2, dim=-1)
    
    def interpolate_motions(self, latent1: torch.Tensor, latent2: torch.Tensor, alpha: float) -> torch.Tensor:
        """在潜在空间中插值两个运动"""
        return (1 - alpha) * latent1 + alpha * latent2

class MotionCompatibilityMatrix(nn.Module):
    """
    运动兼容性矩阵 - 学习不同运动之间的兼容性
    """
    
    def __init__(self, num_motion_types: int, hidden_dim: int = 128):
        super().__init__()
        self.num_motion_types = num_motion_types
        
        # 兼容性评估网络
        self.compatibility_net = nn.Sequential(
            nn.Linear(num_motion_types * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, motion_type1: torch.Tensor, motion_type2: torch.Tensor) -> torch.Tensor:
        """
        计算两种运动类型的兼容性分数
        
        Args:
            motion_type1: (batch_size,) 第一种运动类型
            motion_type2: (batch_size,) 第二种运动类型
            
        Returns:
            compatibility: (batch_size,) 兼容性分数 [0, 1]
        """
        batch_size = motion_type1.size(0)
        
        # 创建one-hot编码
        type1_onehot = F.one_hot(motion_type1, self.num_motion_types).float()
        type2_onehot = F.one_hot(motion_type2, self.num_motion_types).float()
        
        # 拼接输入
        input_features = torch.cat([type1_onehot, type2_onehot], dim=-1)
        
        # 计算兼容性
        compatibility = self.compatibility_net(input_features).squeeze(-1)
        
        return compatibility

class TransitionMotionGenerator(nn.Module):
    """
    过渡动作生成器 - 生成两个运动之间的平滑过渡
    """
    
    def __init__(self, latent_dim: int, motion_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.motion_dim = motion_dim
        
        # 过渡轨迹生成网络
        self.transition_generator = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, hidden_dim),  # +1 for transition phase
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 过渡长度预测网络
        self.duration_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保输出为正值
        )
    
    def predict_transition_duration(self, start_latent: torch.Tensor, end_latent: torch.Tensor) -> torch.Tensor:
        """预测过渡所需的时间长度"""
        input_features = torch.cat([start_latent, end_latent], dim=-1)
        duration = self.duration_predictor(input_features).squeeze(-1)
        return duration
    
    def generate_transition_sequence(self, 
                                   start_latent: torch.Tensor, 
                                   end_latent: torch.Tensor, 
                                   num_steps: int) -> torch.Tensor:
        """
        生成过渡序列
        
        Args:
            start_latent: (batch_size, latent_dim) 起始运动潜在表示
            end_latent: (batch_size, latent_dim) 目标运动潜在表示
            num_steps: 过渡步数
            
        Returns:
            transition_sequence: (batch_size, num_steps, latent_dim) 过渡序列
        """
        batch_size = start_latent.size(0)
        transition_sequence = []
        
        for step in range(num_steps):
            # 计算当前过渡阶段 [0, 1]
            phase = torch.full((batch_size, 1), step / (num_steps - 1), 
                             device=start_latent.device, dtype=start_latent.dtype)
            
            # 生成当前步的潜在表示
            input_features = torch.cat([start_latent, end_latent, phase], dim=-1)
            current_latent = self.transition_generator(input_features)
            
            transition_sequence.append(current_latent)
        
        return torch.stack(transition_sequence, dim=1)

# 损失函数定义
class MotionEncoderLoss(nn.Module):
    """运动编码器的复合损失函数"""
    
    def __init__(self, 
                 recon_weight: float = 1.0,
                 kl_weight: float = 0.1,
                 classification_weight: float = 0.5,
                 compatibility_weight: float = 0.3):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.classification_weight = classification_weight
        self.compatibility_weight = compatibility_weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                original: torch.Tensor,
                reconstructed: torch.Tensor,
                mean: torch.Tensor,
                logvar: torch.Tensor,
                motion_pred: torch.Tensor,
                motion_type: torch.Tensor,
                compatibility_pred: Optional[torch.Tensor] = None,
                compatibility_target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # 重构损失
        recon_loss = self.mse_loss(reconstructed, original)
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.size(0)
        
        # 分类损失
        classification_loss = self.ce_loss(motion_pred, motion_type)
        
        # 总损失
        total_loss = (self.recon_weight * recon_loss + 
                     self.kl_weight * kl_loss + 
                     self.classification_weight * classification_loss)
        
        losses = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'classification_loss': classification_loss
        }
        
        # 兼容性损失（如果提供）
        if compatibility_pred is not None and compatibility_target is not None:
            compatibility_loss = self.mse_loss(compatibility_pred, compatibility_target)
            total_loss += self.compatibility_weight * compatibility_loss
            losses['compatibility_loss'] = compatibility_loss
            losses['total_loss'] = total_loss
        
        return losses

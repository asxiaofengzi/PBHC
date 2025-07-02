import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from .motion_encoder import MotionType, MotionEncoder, MotionCompatibilityMatrix, TransitionMotionGenerator

class FusionController(nn.Module):
    """
    运动融合控制器 - 核心融合决策和执行模块
    
    功能：
    1. 动态选择主导运动模式
    2. 计算运动融合权重
    3. 生成平滑过渡序列
    4. 协调多个专家策略
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 num_motion_types: int = 6,
                 obs_dim: int = 256,
                 action_dim: int = 23,
                 hidden_dim: int = 256,
                 max_active_motions: int = 3):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_motion_types = num_motion_types
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_active_motions = max_active_motions
        
        # 运动选择网络 - 基于当前状态选择活跃的运动模式
        self.motion_selector = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_motion_types),
            nn.Softmax(dim=-1)
        )
        
        # 融合权重生成网络 - 计算不同运动的混合权重
        self.fusion_weight_generator = nn.Sequential(
            nn.Linear(obs_dim + latent_dim + num_motion_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_motion_types),
            nn.Softmax(dim=-1)
        )
        
        # 时间状态编码器 - 编码运动的时间信息
        self.temporal_encoder = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim // 2),  # +2 for phase and duration
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 动作融合网络 - 将多个专家的动作进行融合
        self.action_fusion_net = nn.Sequential(
            nn.Linear(action_dim * num_motion_types + num_motion_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 融合质量评估网络
        self.fusion_quality_evaluator = nn.Sequential(
            nn.Linear(obs_dim + action_dim + num_motion_types, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 状态记录
        self.current_motion_weights = None
        self.transition_state = None
        
    def select_active_motions(self, observation: torch.Tensor, current_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        选择当前活跃的运动模式
        
        Args:
            observation: (batch_size, obs_dim) 当前观测
            current_latent: (batch_size, latent_dim) 当前潜在状态
            
        Returns:
            selection_result: 包含选择结果的字典
        """
        batch_size = observation.size(0)
        
        # 拼接输入特征
        input_features = torch.cat([observation, current_latent], dim=-1)
        
        # 计算运动选择概率
        motion_probs = self.motion_selector(input_features)
        
        # 选择top-k个最可能的运动
        topk_probs, topk_indices = torch.topk(motion_probs, self.max_active_motions, dim=-1)
        
        # 重新归一化概率
        normalized_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        return {
            'motion_probs': motion_probs,
            'active_motion_indices': topk_indices,
            'active_motion_probs': normalized_probs,
            'motion_diversity': self._compute_diversity_score(motion_probs)
        }
    
    def compute_fusion_weights(self, 
                             observation: torch.Tensor,
                             current_latent: torch.Tensor,
                             active_motion_probs: torch.Tensor) -> torch.Tensor:
        """
        计算运动融合权重
        
        Args:
            observation: (batch_size, obs_dim) 当前观测
            current_latent: (batch_size, latent_dim) 当前潜在状态
            active_motion_probs: (batch_size, num_motion_types) 活跃运动概率
            
        Returns:
            fusion_weights: (batch_size, num_motion_types) 融合权重
        """
        # 拼接输入特征
        input_features = torch.cat([observation, current_latent, active_motion_probs], dim=-1)
        
        # 生成融合权重
        fusion_weights = self.fusion_weight_generator(input_features)
        
        # 平滑处理，避免权重变化过于剧烈
        if self.current_motion_weights is not None:
            smoothing_factor = 0.8
            fusion_weights = (smoothing_factor * self.current_motion_weights + 
                            (1 - smoothing_factor) * fusion_weights)
        
        self.current_motion_weights = fusion_weights.detach()
        
        return fusion_weights
    
    def fuse_expert_actions(self, 
                          expert_actions: torch.Tensor,
                          fusion_weights: torch.Tensor,
                          observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        融合多个专家策略的动作
        
        Args:
            expert_actions: (batch_size, num_experts, action_dim) 专家动作
            fusion_weights: (batch_size, num_motion_types) 融合权重
            observation: (batch_size, obs_dim) 当前观测
            
        Returns:
            fusion_result: 包含融合结果的字典
        """
        batch_size, num_experts, action_dim = expert_actions.shape
        
        # 简单加权融合
        weighted_actions = torch.sum(expert_actions * fusion_weights.unsqueeze(-1), dim=1)
        
        # 高级融合网络
        flattened_actions = expert_actions.view(batch_size, -1)
        fusion_input = torch.cat([flattened_actions, fusion_weights], dim=-1)
        advanced_fused_actions = self.action_fusion_net(fusion_input)
        
        # 融合质量评估
        quality_input = torch.cat([observation, advanced_fused_actions, fusion_weights], dim=-1)
        fusion_quality = self.fusion_quality_evaluator(quality_input)
        
        # 根据质量评估选择融合方式
        quality_threshold = 0.7
        use_advanced = (fusion_quality > quality_threshold).float()
        
        final_actions = (use_advanced.unsqueeze(-1) * advanced_fused_actions + 
                        (1 - use_advanced.unsqueeze(-1)) * weighted_actions)
        
        return {
            'fused_actions': final_actions,
            'weighted_actions': weighted_actions,
            'advanced_fused_actions': advanced_fused_actions,
            'fusion_quality': fusion_quality,
            'use_advanced_ratio': use_advanced.mean()
        }
    
    def encode_temporal_state(self, 
                            current_latent: torch.Tensor,
                            motion_phase: torch.Tensor,
                            motion_duration: torch.Tensor) -> torch.Tensor:
        """
        编码时间状态信息
        
        Args:
            current_latent: (batch_size, latent_dim) 当前潜在状态
            motion_phase: (batch_size, 1) 运动阶段 [0, 1]
            motion_duration: (batch_size, 1) 运动持续时间
            
        Returns:
            temporal_encoding: (batch_size, latent_dim) 时间编码
        """
        temporal_input = torch.cat([current_latent, motion_phase, motion_duration], dim=-1)
        temporal_encoding = self.temporal_encoder(temporal_input)
        return temporal_encoding
    
    def _compute_diversity_score(self, motion_probs: torch.Tensor) -> torch.Tensor:
        """计算运动多样性分数（熵）"""
        # 避免log(0)
        eps = 1e-8
        entropy = -torch.sum(motion_probs * torch.log(motion_probs + eps), dim=-1)
        # 归一化到[0,1]
        max_entropy = np.log(self.num_motion_types)
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    def update_transition_state(self, 
                              start_motion_type: int,
                              end_motion_type: int,
                              transition_progress: float):
        """更新过渡状态"""
        self.transition_state = {
            'start_motion': start_motion_type,
            'end_motion': end_motion_type,
            'progress': transition_progress,
            'is_transitioning': transition_progress < 1.0
        }
    
    def get_transition_info(self) -> Optional[Dict]:
        """获取当前过渡信息"""
        return self.transition_state

class AdaptiveFusionScheduler:
    """
    自适应融合调度器 - 管理运动融合的时序和策略
    """
    
    def __init__(self, 
                 smoothing_window: int = 10,
                 transition_threshold: float = 0.3,
                 stability_threshold: float = 0.1):
        self.smoothing_window = smoothing_window
        self.transition_threshold = transition_threshold
        self.stability_threshold = stability_threshold
        
        # 历史记录
        self.weight_history = []
        self.motion_history = []
        self.stability_scores = []
        
    def should_trigger_transition(self, 
                                current_weights: torch.Tensor,
                                target_motion_type: int) -> bool:
        """判断是否应该触发运动过渡"""
        if len(self.weight_history) < self.smoothing_window:
            return False
        
        # 计算权重变化的稳定性
        recent_weights = torch.stack(self.weight_history[-self.smoothing_window:])
        weight_variance = torch.var(recent_weights, dim=0).mean()
        
        # 如果当前主导运动与目标差异较大且权重变化稳定
        current_dominant = torch.argmax(current_weights, dim=-1)
        is_different_motion = (current_dominant != target_motion_type).float().mean() > 0.5
        is_stable = weight_variance < self.stability_threshold
        
        return is_different_motion and is_stable
    
    def compute_transition_speed(self, 
                               start_motion_type: int,
                               end_motion_type: int,
                               compatibility_score: float) -> float:
        """
        根据运动兼容性计算过渡速度
        
        Args:
            start_motion_type: 起始运动类型
            end_motion_type: 目标运动类型
            compatibility_score: 兼容性分数 [0, 1]
            
        Returns:
            transition_speed: 过渡速度，值越大过渡越快
        """
        # 兼容性越高，过渡越快
        base_speed = 0.02  # 基础过渡速度
        compatibility_bonus = compatibility_score * 0.08
        
        # 特殊运动对的调整
        special_pairs = {
            (MotionType.TAICHI.value, MotionType.YOGA.value): 0.05,  # 慢速流畅过渡
            (MotionType.BOXING.value, MotionType.KARATE.value): 0.03,  # 快速格斗过渡
        }
        
        pair_key = (start_motion_type, end_motion_type)
        if pair_key in special_pairs:
            return special_pairs[pair_key]
        
        return base_speed + compatibility_bonus
    
    def update_history(self, weights: torch.Tensor, motion_type: int):
        """更新历史记录"""
        self.weight_history.append(weights.detach().cpu())
        self.motion_history.append(motion_type)
        
        # 保持固定长度
        if len(self.weight_history) > self.smoothing_window * 2:
            self.weight_history = self.weight_history[-self.smoothing_window:]
            self.motion_history = self.motion_history[-self.smoothing_window:]
    
    def get_smoothed_weights(self, current_weights: torch.Tensor) -> torch.Tensor:
        """获取平滑后的权重"""
        if len(self.weight_history) < 3:
            return current_weights
        
        # 使用指数移动平均
        alpha = 0.7
        smoothed = current_weights
        for i in range(min(3, len(self.weight_history))):
            historical_weights = self.weight_history[-(i+1)].to(current_weights.device)
            smoothed = alpha * smoothed + (1 - alpha) * historical_weights
            alpha *= 0.8  # 降低历史权重的影响
        
        return smoothed

class FusionMetrics:
    """融合质量评估指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.smoothness_scores = []
        self.diversity_scores = []
        self.transition_success_rates = []
        self.fusion_quality_scores = []
    
    def update(self, 
              smoothness: float,
              diversity: float,
              transition_success: bool,
              fusion_quality: float):
        """更新指标"""
        self.smoothness_scores.append(smoothness)
        self.diversity_scores.append(diversity)
        self.transition_success_rates.append(float(transition_success))
        self.fusion_quality_scores.append(fusion_quality)
    
    def get_summary(self) -> Dict[str, float]:
        """获取指标汇总"""
        if not self.smoothness_scores:
            return {}
        
        return {
            'avg_smoothness': np.mean(self.smoothness_scores),
            'avg_diversity': np.mean(self.diversity_scores),
            'transition_success_rate': np.mean(self.transition_success_rates),
            'avg_fusion_quality': np.mean(self.fusion_quality_scores),
            'smoothness_std': np.std(self.smoothness_scores),
            'diversity_std': np.std(self.diversity_scores)
        }

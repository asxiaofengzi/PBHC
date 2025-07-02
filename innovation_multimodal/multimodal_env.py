import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# 添加PBHC项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from .motion_encoder import MotionType, MotionEncoder
from .fusion_controller import FusionController, AdaptiveFusionScheduler

class MultimodalMotionTrackingEnv(LeggedRobotMotionTracking):
    """
    多模态运动跟踪环境 - 扩展原有环境以支持多运动融合
    
    新增功能：
    1. 多运动数据管理
    2. 动态运动切换
    3. 融合质量评估
    4. 自适应课程学习
    """
    
    def __init__(self, config, device):
        super().__init__(config, device)
        
        # 多模态相关配置
        self.multimodal_config = config.get('multimodal', {})
        self.enable_fusion = self.multimodal_config.get('enable_fusion', True)
        self.max_active_motions = self.multimodal_config.get('max_active_motions', 3)
        self.fusion_curriculum = self.multimodal_config.get('fusion_curriculum', False)
        
        # 初始化多模态组件
        self._init_multimodal_components()
        
        # 运动状态管理
        self.current_motion_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_motion_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_phases = torch.zeros(self.num_envs, 1, device=self.device)
        self.transition_states = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 融合历史和指标
        self.fusion_history = []
        self.fusion_success_count = 0
        self.total_fusion_attempts = 0
        
    def _init_multimodal_components(self):
        """初始化多模态相关组件"""
        # 运动编码器
        motion_dim = self.num_dofs * 2 + 7  # joint pos + joint vel + root state
        self.motion_encoder = MotionEncoder(
            motion_dim=motion_dim,
            latent_dim=self.multimodal_config.get('latent_dim', 128),
            num_motion_types=len(MotionType),
            hidden_dims=self.multimodal_config.get('encoder_hidden_dims', [512, 256])
        ).to(self.device)
        
        # 融合控制器
        self.fusion_controller = FusionController(
            latent_dim=self.multimodal_config.get('latent_dim', 128),
            num_motion_types=len(MotionType),
            obs_dim=self.config.robot.policy_obs_dim,
            action_dim=self.config.robot.actions_dim,
            max_active_motions=self.max_active_motions
        ).to(self.device)
        
        # 自适应调度器
        self.fusion_scheduler = AdaptiveFusionScheduler(
            smoothing_window=self.multimodal_config.get('smoothing_window', 10),
            transition_threshold=self.multimodal_config.get('transition_threshold', 0.3)
        )
        
        # 扩展运动库以支持多运动类型
        self._extend_motion_lib()
    
    def _extend_motion_lib(self):
        """扩展运动库以支持多运动类型标记"""
        # 为现有运动数据添加类型标记
        self.motion_type_mapping = {
            'Horse-stance_pose': MotionType.TAICHI,
            'Bruce_Lee_pose': MotionType.KARATE,
            'Charleston_dance': MotionType.DANCE,
            'Hooks_punch': MotionType.BOXING,
            'Roundhouse_kick': MotionType.KARATE,
            'Side_kick': MotionType.KARATE
        }
        
        # 创建运动类型索引
        self.motion_types_tensor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for env_id in range(self.num_envs):
            motion_name = self._get_motion_name_for_env(env_id)
            motion_type = self.motion_type_mapping.get(motion_name, MotionType.TAICHI)
            self.motion_types_tensor[env_id] = list(MotionType).index(motion_type)
    
    def _get_motion_name_for_env(self, env_id: int) -> str:
        """获取环境对应的运动名称（简化实现）"""
        # 这里应该根据实际的运动库实现来获取运动名称
        # 暂时使用简化的映射
        motion_names = list(self.motion_type_mapping.keys())
        return motion_names[env_id % len(motion_names)]
    
    def _pre_compute_observations_callback(self):
        """重写观测计算，添加多模态信息"""
        super()._pre_compute_observations_callback()
        
        if self.enable_fusion:
            self._compute_multimodal_observations()
    
    def _compute_multimodal_observations(self):
        """计算多模态相关的观测信息"""
        # 编码当前运动状态
        current_motion_data = self._extract_current_motion_data()
        
        # 运动编码
        with torch.no_grad():
            encoding_result = self.motion_encoder(current_motion_data, self.motion_types_tensor)
            self.current_motion_latent = encoding_result['latent_code']
        
        # 融合控制决策
        if hasattr(self, '_obs_buf_dict') and 'actor_obs' in self._obs_buf_dict:
            actor_obs = self._obs_buf_dict['actor_obs']
            
            # 运动选择
            selection_result = self.fusion_controller.select_active_motions(
                actor_obs, self.current_motion_latent
            )
            
            # 融合权重计算
            self.current_fusion_weights = self.fusion_controller.compute_fusion_weights(
                actor_obs, 
                self.current_motion_latent,
                selection_result['motion_probs']
            )
            
            # 更新调度器历史
            dominant_motion = torch.argmax(self.current_fusion_weights, dim=-1)
            for env_id in range(self.num_envs):
                self.fusion_scheduler.update_history(
                    self.current_fusion_weights[env_id:env_id+1],
                    dominant_motion[env_id].item()
                )
    
    def _extract_current_motion_data(self) -> torch.Tensor:
        """提取当前运动数据用于编码"""
        # 拼接关节位置、关节速度和根状态
        root_state = self.simulator.robot_root_states[:, :7]  # pos + quat
        joint_pos = self.simulator.dof_pos
        joint_vel = self.simulator.dof_vel
        
        motion_data = torch.cat([joint_pos, joint_vel, root_state], dim=-1)
        return motion_data
    
    def _compute_multimodal_rewards(self) -> Dict[str, torch.Tensor]:
        """计算多模态融合相关的奖励"""
        rewards = {}
        
        if not hasattr(self, 'current_fusion_weights'):
            return rewards
        
        # 融合平滑度奖励
        if len(self.fusion_history) > 1:
            prev_weights = self.fusion_history[-1]
            current_weights = self.current_fusion_weights
            
            # 计算权重变化的平滑度
            weight_change = torch.norm(current_weights - prev_weights, dim=-1)
            smoothness_reward = torch.exp(-weight_change * 10.0)  # 权重变化越小，奖励越高
            rewards['fusion_smoothness'] = smoothness_reward
        
        # 运动多样性奖励（适度鼓励）
        weights_entropy = -torch.sum(self.current_fusion_weights * 
                                   torch.log(self.current_fusion_weights + 1e-8), dim=-1)
        max_entropy = np.log(len(MotionType))
        normalized_entropy = weights_entropy / max_entropy
        
        # 适度的多样性奖励（避免过度切换）
        diversity_reward = torch.clamp(normalized_entropy, 0.0, 0.7)
        rewards['motion_diversity'] = diversity_reward
        
        # 融合质量奖励
        if hasattr(self, '_last_fusion_quality'):
            rewards['fusion_quality'] = self._last_fusion_quality
        
        # 过渡成功奖励
        if torch.any(self.transition_states):
            transition_mask = self.transition_states.float()
            transition_success = self._evaluate_transition_success()
            rewards['transition_success'] = transition_success * transition_mask
        
        return rewards
    
    def _evaluate_transition_success(self) -> torch.Tensor:
        """评估运动过渡的成功程度"""
        # 基于当前状态与目标运动的匹配度
        if not hasattr(self, 'target_motion_latent'):
            return torch.ones(self.num_envs, device=self.device)
        
        # 计算潜在空间距离
        latent_distance = torch.norm(
            self.current_motion_latent - self.target_motion_latent, dim=-1
        )
        
        # 转换为成功分数
        success_score = torch.exp(-latent_distance)
        return success_score
    
    def _compute_reward(self):
        """重写奖励计算，添加多模态奖励"""
        # 调用父类奖励计算
        super()._compute_reward()
        
        # 添加多模态奖励
        if self.enable_fusion:
            multimodal_rewards = self._compute_multimodal_rewards()
            
            # 将多模态奖励添加到总奖励中
            for reward_name, reward_value in multimodal_rewards.items():
                reward_scale = self.multimodal_config.get(f'{reward_name}_scale', 0.1)
                self.rew_buf += reward_scale * reward_value
                
                # 记录到日志
                if reward_name not in self.extras:
                    self.extras[reward_name] = reward_value.mean()
    
    def trigger_motion_transition(self, env_ids: torch.Tensor, target_motion_types: torch.Tensor):
        """触发运动过渡"""
        if len(env_ids) == 0:
            return
        
        self.transition_states[env_ids] = True
        self.target_motion_types[env_ids] = target_motion_types
        self.total_fusion_attempts += len(env_ids)
        
        # 生成目标运动的潜在表示
        target_motion_data = self._generate_target_motion_data(env_ids, target_motion_types)
        with torch.no_grad():
            target_encoding = self.motion_encoder(target_motion_data, target_motion_types)
            self.target_motion_latent = target_encoding['latent_code']
    
    def _generate_target_motion_data(self, env_ids: torch.Tensor, target_motion_types: torch.Tensor) -> torch.Tensor:
        """生成目标运动的数据（简化实现）"""
        # 这里应该从运动库中采样目标运动的数据
        # 暂时使用当前数据作为占位符
        current_data = self._extract_current_motion_data()
        return current_data[env_ids]
    
    def update_curriculum(self):
        """更新课程学习进度"""
        if not self.fusion_curriculum:
            return
        
        # 根据融合成功率调整课程难度
        if self.total_fusion_attempts > 100:
            success_rate = self.fusion_success_count / self.total_fusion_attempts
            
            if success_rate > 0.8:
                # 增加融合难度：更多同时激活的运动
                self.max_active_motions = min(len(MotionType), self.max_active_motions + 1)
            elif success_rate < 0.3:
                # 降低融合难度：减少同时激活的运动
                self.max_active_motions = max(2, self.max_active_motions - 1)
            
            # 重置计数器
            self.fusion_success_count = 0
            self.total_fusion_attempts = 0
    
    def _post_physics_step(self):
        """重写物理步骤后处理，添加多模态逻辑"""
        super()._post_physics_step()
        
        if self.enable_fusion:
            # 更新融合历史
            if hasattr(self, 'current_fusion_weights'):
                self.fusion_history.append(self.current_fusion_weights.clone())
                if len(self.fusion_history) > 50:  # 保持固定长度
                    self.fusion_history.pop(0)
            
            # 检查过渡完成
            self._check_transition_completion()
            
            # 更新课程学习
            if self.common_step_counter % 1000 == 0:
                self.update_curriculum()
    
    def _check_transition_completion(self):
        """检查运动过渡是否完成"""
        if not torch.any(self.transition_states):
            return
        
        transitioning_envs = self.transition_states.nonzero(as_tuple=False).squeeze(-1)
        
        for env_id in transitioning_envs:
            # 检查是否达到目标运动
            current_dominant = torch.argmax(self.current_fusion_weights[env_id])
            target_motion = self.target_motion_types[env_id]
            
            if current_dominant == target_motion:
                self.transition_states[env_id] = False
                self.fusion_success_count += 1
                
                # 更新当前运动类型
                self.current_motion_types[env_id] = target_motion
    
    def get_multimodal_info(self) -> Dict:
        """获取多模态状态信息"""
        info = {
            'fusion_enabled': self.enable_fusion,
            'max_active_motions': self.max_active_motions,
            'transition_envs': self.transition_states.sum().item(),
            'total_envs': self.num_envs
        }
        
        if hasattr(self, 'current_fusion_weights'):
            info.update({
                'avg_fusion_weights': self.current_fusion_weights.mean(dim=0).cpu().numpy(),
                'dominant_motions': torch.argmax(self.current_fusion_weights, dim=-1).cpu().numpy(),
                'fusion_entropy': self._compute_fusion_entropy().mean().item()
            })
        
        if self.total_fusion_attempts > 0:
            info['fusion_success_rate'] = self.fusion_success_count / self.total_fusion_attempts
        
        return info
    
    def _compute_fusion_entropy(self) -> torch.Tensor:
        """计算融合权重的熵"""
        if not hasattr(self, 'current_fusion_weights'):
            return torch.zeros(self.num_envs, device=self.device)
        
        weights = self.current_fusion_weights
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        return entropy
    
    def reset_multimodal_state(self, env_ids: torch.Tensor):
        """重置多模态状态"""
        if len(env_ids) == 0:
            return
        
        self.transition_states[env_ids] = False
        self.current_motion_types[env_ids] = self.motion_types_tensor[env_ids]
        self.target_motion_types[env_ids] = self.motion_types_tensor[env_ids]
        
        # 重置融合控制器状态
        if hasattr(self.fusion_controller, 'current_motion_weights'):
            if self.fusion_controller.current_motion_weights is not None:
                # 重置为均匀分布
                uniform_weights = torch.ones(len(env_ids), len(MotionType), device=self.device) / len(MotionType)
                self.fusion_controller.current_motion_weights[env_ids] = uniform_weights

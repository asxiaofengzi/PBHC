import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import sys
import os

# 添加PBHC项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from humanoidverse.agents.mh_ppo.mh_ppo import MHPPO
from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from .motion_encoder import MotionEncoder, MotionEncoderLoss, MotionType
from .fusion_controller import FusionController
from .multimodal_env import MultimodalMotionTrackingEnv

class MultiExpertActor(nn.Module):
    """
    多专家Actor网络 - 每种运动类型对应一个专家策略
    """
    
    def __init__(self, 
                 obs_dim_dict: Dict,
                 module_config_dict: Dict,
                 num_actions: int,
                 num_motion_types: int,
                 init_noise_std: float = 1.0,
                 shared_backbone: bool = True):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_motion_types = num_motion_types
        self.shared_backbone = shared_backbone
        
        # 共享主干网络（可选）
        if shared_backbone:
            backbone_dims = module_config_dict['layer_config']['hidden_dims'][:-1]
            self.shared_backbone_net = self._build_backbone(
                obs_dim_dict['actor_obs'], backbone_dims
            )
            expert_input_dim = backbone_dims[-1]
        else:
            self.shared_backbone_net = None
            expert_input_dim = obs_dim_dict['actor_obs']
        
        # 为每种运动类型创建专家网络
        self.motion_experts = nn.ModuleList()
        for motion_type in range(num_motion_types):
            expert = self._build_expert_network(
                expert_input_dim, 
                module_config_dict['layer_config']['hidden_dims'][-1:],
                num_actions
            )
            self.motion_experts.append(expert)
        
        # 运动类型选择网络
        self.motion_selector = nn.Sequential(
            nn.Linear(obs_dim_dict['actor_obs'], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_motion_types),
            nn.Softmax(dim=-1)
        )
        
        # 动作噪声参数
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # 禁用参数验证以提高速度
        from torch.distributions import Normal
        Normal.set_default_validate_args = False
    
    def _build_backbone(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """构建共享主干网络"""
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
    
    def _build_expert_network(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
        """构建专家网络"""
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
    
    def forward_experts(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播所有专家
        
        Args:
            obs: (batch_size, obs_dim) 观测
            
        Returns:
            expert_actions: (batch_size, num_motion_types, num_actions) 所有专家的动作
        """
        batch_size = obs.size(0)
        
        # 共享主干特征提取
        if self.shared_backbone_net is not None:
            shared_features = self.shared_backbone_net(obs)
        else:
            shared_features = obs
        
        # 所有专家并行推理
        expert_outputs = []
        for expert in self.motion_experts:
            expert_output = expert(shared_features)
            expert_outputs.append(expert_output)
        
        # 堆叠专家输出
        expert_actions = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, num_actions)
        
        return expert_actions
    
    def select_motion_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """选择运动权重"""
        motion_weights = self.motion_selector(obs)
        return motion_weights
    
    def update_distribution(self, obs: torch.Tensor, fusion_weights: Optional[torch.Tensor] = None):
        """更新动作分布"""
        # 获取所有专家的动作
        expert_actions = self.forward_experts(obs)
        
        # 如果没有提供融合权重，使用内部选择器
        if fusion_weights is None:
            fusion_weights = self.select_motion_weights(obs)
        
        # 加权融合专家动作
        fused_mean = torch.sum(expert_actions * fusion_weights.unsqueeze(-1), dim=1)
        
        # 创建分布
        from torch.distributions import Normal
        self.distribution = Normal(fused_mean, fused_mean * 0. + self.std)
    
    def act(self, obs: torch.Tensor, fusion_weights: Optional[torch.Tensor] = None, **kwargs):
        """采样动作"""
        self.update_distribution(obs, fusion_weights)
        return self.distribution.sample()
    
    def act_inference(self, obs: torch.Tensor, fusion_weights: Optional[torch.Tensor] = None):
        """推理动作（无噪声）"""
        expert_actions = self.forward_experts(obs)
        
        if fusion_weights is None:
            fusion_weights = self.select_motion_weights(obs)
        
        fused_actions = torch.sum(expert_actions * fusion_weights.unsqueeze(-1), dim=1)
        return fused_actions
    
    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution else None
    
    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution else None
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) if self.distribution else None
    
    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1) if self.distribution else None
    
    def reset(self, dones=None):
        pass

class MultimodalPPO(MHPPO):
    """
    多模态PPO算法 - 扩展MHPPO以支持多模态运动融合学习
    """
    
    def __init__(self, env: MultimodalMotionTrackingEnv, config, log_dir=None, device='cpu'):
        # 确保环境是多模态环境
        assert isinstance(env, MultimodalMotionTrackingEnv), "Environment must be MultimodalMotionTrackingEnv"
        
        super().__init__(env, config, log_dir, device)
        
        # 多模态相关配置
        self.multimodal_config = config.get('multimodal', {})
        self.enable_expert_training = self.multimodal_config.get('enable_expert_training', True)
        self.encoder_learning_rate = self.multimodal_config.get('encoder_learning_rate', 1e-4)
        self.fusion_loss_weight = self.multimodal_config.get('fusion_loss_weight', 0.1)
        
        # 预训练阶段标志
        self.pretraining_phase = self.multimodal_config.get('start_with_pretraining', True)
        self.pretraining_iterations = self.multimodal_config.get('pretraining_iterations', 5000)
        self.current_phase = 'pretraining' if self.pretraining_phase else 'multimodal'
        
        # 重新设置模型和优化器
        self._setup_multimodal_models()
    
    def _setup_multimodal_models(self):
        """设置多模态模型"""
        # 多专家Actor
        self.actor = MultiExpertActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            num_motion_types=len(MotionType),
            init_noise_std=self.config.init_noise_std,
            shared_backbone=self.multimodal_config.get('shared_backbone', True)
        ).to(self.device)
        
        # Critic保持不变
        self.critic = PPOCritic(
            self.algo_obs_dim_dict,
            self.config.module_dict.critic
        ).to(self.device)
        
        # 获取运动编码器和融合控制器（从环境中）
        self.motion_encoder = self.env.motion_encoder
        self.fusion_controller = self.env.fusion_controller
        
        # 设置优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        
        # 运动编码器优化器
        self.encoder_optimizer = optim.Adam(
            self.motion_encoder.parameters(), 
            lr=self.encoder_learning_rate
        )
        
        # 融合控制器优化器
        self.fusion_optimizer = optim.Adam(
            self.fusion_controller.parameters(),
            lr=self.encoder_learning_rate
        )
        
        # 损失函数
        
    def _setup_models_and_optimizer(self):
        """重写基类方法以使用多模态模型"""
        # 设置奖励函数数量
        self.config.module_dict.critic['output_dim'][-1] = self.num_rew_fn
        
        # 调用多模态模型设置
        self._setup_multimodal_models()
        
        print("🎭 Multimodal Actor:", self.actor)
        print("🧠 Critic:", self.critic)
        self.encoder_loss_fn = MotionEncoderLoss(
            recon_weight=self.multimodal_config.get('recon_weight', 1.0),
            kl_weight=self.multimodal_config.get('kl_weight', 0.1),
            classification_weight=self.multimodal_config.get('classification_weight', 0.5)
        )
    
    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        """重写Actor rollout步骤以支持多专家"""
        # 获取融合权重
        if hasattr(self.env, 'current_fusion_weights'):
            fusion_weights = self.env.current_fusion_weights
        else:
            fusion_weights = None
        
        # 使用融合权重采样动作
        actions = self.actor.act(obs_dict["actor_obs"], fusion_weights)
        policy_state_dict["actions"] = actions
        
        # 记录专家信息
        if fusion_weights is not None:
            policy_state_dict["fusion_weights"] = fusion_weights
            
            # 记录专家动作（用于分析）
            with torch.no_grad():
                expert_actions = self.actor.forward_experts(obs_dict["actor_obs"])
                policy_state_dict["expert_actions"] = expert_actions
        
        # 其他信息
        action_mean = self.actor.action_mean.detach()
        action_sigma = self.actor.action_std.detach() 
        actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
        
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob
        
        return policy_state_dict
    
    def _update_motion_encoder(self, policy_state_dict) -> Dict[str, float]:
        """更新运动编码器"""
        if not hasattr(self.env, '_extract_current_motion_data'):
            return {}
        
        # 提取运动数据
        motion_data = self.env._extract_current_motion_data()
        motion_types = self.env.motion_types_tensor
        
        # 前向传播
        encoding_result = self.motion_encoder(motion_data, motion_types)
        
        # 计算损失
        losses = self.encoder_loss_fn(
            original=motion_data,
            reconstructed=encoding_result['reconstructed'],
            mean=encoding_result['mean'],
            logvar=encoding_result['logvar'],
            motion_pred=encoding_result['motion_pred'],
            motion_type=motion_types
        )
        
        # 反向传播
        self.encoder_optimizer.zero_grad()
        losses['total_loss'].backward()
        nn.utils.clip_grad_norm_(self.motion_encoder.parameters(), self.max_grad_norm)
        self.encoder_optimizer.step()
        
        # 返回损失信息
        return {k: v.item() for k, v in losses.items()}
    
    def _update_fusion_controller(self, policy_state_dict) -> Dict[str, float]:
        """更新融合控制器"""
        if not hasattr(self.env, 'current_fusion_weights'):
            return {}
        
        # 获取观测和当前状态
        obs = policy_state_dict.get('actor_obs')
        current_latent = self.env.current_motion_latent
        
        if obs is None or current_latent is None:
            return {}
        
        # 前向传播融合控制器
        selection_result = self.fusion_controller.select_active_motions(obs, current_latent)
        fusion_weights = self.fusion_controller.compute_fusion_weights(
            obs, current_latent, selection_result['motion_probs']
        )
        
        # 融合专家动作
        if 'expert_actions' in policy_state_dict:
            fusion_result = self.fusion_controller.fuse_expert_actions(
                policy_state_dict['expert_actions'],
                fusion_weights,
                obs
            )
            
            # 融合损失：预期动作与实际动作的差异
            actual_actions = policy_state_dict['actions']
            expected_actions = fusion_result['fused_actions']
            
            fusion_loss = nn.MSELoss()(actual_actions, expected_actions.detach())
            
            # 反向传播
            self.fusion_optimizer.zero_grad()
            fusion_loss.backward()
            nn.utils.clip_grad_norm_(self.fusion_controller.parameters(), self.max_grad_norm)
            self.fusion_optimizer.step()
            
            return {
                'fusion_loss': fusion_loss.item(),
                'fusion_quality': fusion_result['fusion_quality'].mean().item(),
                'use_advanced_ratio': fusion_result['use_advanced_ratio'].item()
            }
        
        return {}
    
    def _update_algo_step(self, policy_state_dict, loss_dict):
        """重写算法更新步骤"""
        # 标准PPO更新
        loss_dict = super()._update_algo_step(policy_state_dict, loss_dict)
        
        # 根据当前阶段决定是否更新多模态组件
        if self.current_phase == 'pretraining':
            # 预训练阶段：只更新运动编码器
            encoder_losses = self._update_motion_encoder(policy_state_dict)
            loss_dict.update({f'encoder_{k}': v for k, v in encoder_losses.items()})
            
        elif self.current_phase == 'multimodal':
            # 多模态阶段：更新所有组件
            if self.enable_expert_training:
                encoder_losses = self._update_motion_encoder(policy_state_dict)
                fusion_losses = self._update_fusion_controller(policy_state_dict)
                
                loss_dict.update({f'encoder_{k}': v for k, v in encoder_losses.items()})
                loss_dict.update({f'fusion_{k}': v for k, v in fusion_losses.items()})
        
        return loss_dict
    
    def _check_phase_transition(self):
        """检查是否需要切换训练阶段"""
        if (self.current_phase == 'pretraining' and 
            self.current_learning_iteration >= self.pretraining_iterations):
            
            self.current_phase = 'multimodal'
            print(f"Switching to multimodal training phase at iteration {self.current_learning_iteration}")
            
            # 启用融合功能
            if hasattr(self.env, 'enable_fusion'):
                self.env.enable_fusion = True
    
    def _training_step(self):
        """重写训练步骤"""
        # 检查阶段切换
        self._check_phase_transition()
        
        # 调用父类训练步骤
        loss_dict = super()._training_step()
        
        return loss_dict
    
    def _post_epoch_logging(self, log_dict, width=80, pad=40):
        """重写日志记录，添加多模态信息"""
        super()._post_epoch_logging(log_dict, width, pad)
        
        # 添加多模态状态信息
        multimodal_info = self.env.get_multimodal_info()
        
        # 记录到tensorboard
        iteration = log_dict['it']
        for key, value in multimodal_info.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Multimodal/{key}', value, iteration)
            elif isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    self.writer.add_scalar(f'Multimodal/{key}_{i}', v, iteration)
        
        # 记录当前训练阶段
        self.writer.add_scalar('Training/phase', 
                             1 if self.current_phase == 'multimodal' else 0, 
                             iteration)
        
        # 打印多模态信息
        print(f"\nMultimodal Info:")
        print(f"  Phase: {self.current_phase}")
        print(f"  Fusion Success Rate: {multimodal_info.get('fusion_success_rate', 'N/A')}")
        print(f"  Active Motions: {multimodal_info.get('max_active_motions', 'N/A')}")
        print(f"  Transitioning Envs: {multimodal_info.get('transition_envs', 0)}/{multimodal_info.get('total_envs', 0)}")
    
    def save(self, path, infos=None):
        """保存模型，包括多模态组件"""
        save_dict = {
            'model_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'encoder_state_dict': self.motion_encoder.state_dict(),
            'fusion_controller_state_dict': self.fusion_controller.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'fusion_optimizer_state_dict': self.fusion_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'current_phase': self.current_phase,
            'infos': infos,
        }
        torch.save(save_dict, path)
    
    def load(self, ckpt_path):
        """加载模型，包括多模态组件"""
        loaded_dict = torch.load(ckpt_path, map_location=self.device)
        
        # 加载标准组件
        self.actor.load_state_dict(loaded_dict['model_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])
        
        if self.load_optimizer:
            self.actor_optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer_state_dict'])
        
        # 加载多模态组件
        if 'encoder_state_dict' in loaded_dict:
            self.motion_encoder.load_state_dict(loaded_dict['encoder_state_dict'])
        
        if 'fusion_controller_state_dict' in loaded_dict:
            self.fusion_controller.load_state_dict(loaded_dict['fusion_controller_state_dict'])
        
        if self.load_optimizer:
            if 'encoder_optimizer_state_dict' in loaded_dict:
                self.encoder_optimizer.load_state_dict(loaded_dict['encoder_optimizer_state_dict'])
            if 'fusion_optimizer_state_dict' in loaded_dict:
                self.fusion_optimizer.load_state_dict(loaded_dict['fusion_optimizer_state_dict'])
        
        # 恢复训练阶段
        if 'current_phase' in loaded_dict:
            self.current_phase = loaded_dict['current_phase']
        
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos', {})

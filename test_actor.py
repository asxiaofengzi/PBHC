#!/usr/bin/env python3
"""
快速测试多模态actor是否能正确调用
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import torch
    print("🔧 测试MultiExpertActor...")
    
    # 创建一个简单的测试
    from innovation_multimodal.multimodal_ppo import MultiExpertActor
    
    # 测试参数
    obs_dim_dict = {'actor_obs': 100}
    module_config_dict = {
        'layer_config': {
            'hidden_dims': [512, 256, 128]
        }
    }
    num_actions = 23
    num_motion_types = 6
    
    actor = MultiExpertActor(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=module_config_dict,
        num_actions=num_actions,
        num_motion_types=num_motion_types,
        init_noise_std=0.8
    )
    
    print("✅ MultiExpertActor创建成功!")
    
    # 测试act方法
    obs = torch.randn(10, 100)  # batch_size=10, obs_dim=100
    fusion_weights = torch.randn(10, 6)  # batch_size=10, num_motion_types=6
    
    # 测试带融合权重的调用
    actions = actor.act(obs, fusion_weights)
    print(f"✅ act方法调用成功! 动作形状: {actions.shape}")
    
    # 测试不带融合权重的调用
    actions = actor.act(obs)
    print(f"✅ act方法（无融合权重）调用成功! 动作形状: {actions.shape}")
    
    print("\n🎉 MultiExpertActor测试通过!")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

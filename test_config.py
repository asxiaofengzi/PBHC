#!/usr/bin/env python3
"""
测试多模态配置是否能正确加载
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import hydra
    from omegaconf import OmegaConf
    from hydra import initialize, compose
    
    print("🔧 测试多模态配置加载...")
    
    # 测试配置加载
    with initialize(config_path="humanoidverse/config", version_base="1.1"):
        cfg = compose(
            config_name="multimodal_base",
            overrides=[
                "+simulator=isaacgym",
                "+terrain=terrain_locomotion_plane", 
                "project_name=MultimodalMotionFusion",
                "num_envs=128",
                "+obs=motion_tracking/main",
                "+robot=g1/g1_23dof_lock_wrist",
                "+domain_rand=main",
                "+rewards=motion_tracking/main",
                "experiment_name=multimodal_debug",
                "seed=1",
                "+device=cuda:0"
            ]
        )
        
        print("✅ 配置加载成功!")
        print(f"算法目标: {cfg.algo._target_}")
        print(f"环境目标: {cfg.env._target_}")
        print(f"模块字典键: {list(cfg.algo.config.module_dict.keys())}")
        print(f"Critic输出维度: {cfg.algo.config.module_dict.critic.output_dim}")
        print(f"运动文件: {cfg.robot.motion.motion_file}")
        
        # 检查多模态配置
        if hasattr(cfg.algo.config, 'multimodal_config'):
            print(f"多模态专家数量: {cfg.algo.config.multimodal_config.multi_expert.num_experts}")
            print("✅ 多模态配置完整!")
        
        print("\n🎉 所有配置验证通过!")
        
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

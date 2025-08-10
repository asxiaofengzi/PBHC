#!/usr/bin/env python3
"""
修复MultimodalPPO存储错误的脚本
"""

import os
import shutil

def fix_multimodal_storage():
    """修复MultimodalPPO中的存储设置问题"""
    
    # 文件路径
    multimodal_ppo_path = "/home/js/xiaofengzi/PBHC/innovation_multimodal/multimodal_ppo.py"
    backup_path = multimodal_ppo_path + ".backup_storage"
    
    # 备份原文件
    shutil.copy2(multimodal_ppo_path, backup_path)
    print(f"✅ 已备份原文件到: {backup_path}")
    
    # 读取原文件
    with open(multimodal_ppo_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找setup方法的位置
    setup_method_start = content.find("def setup(self):")
    if setup_method_start == -1:
        print("❌ 未找到setup方法")
        return
    
    # 查找setup方法结束位置（下一个方法开始）
    next_method_start = content.find("\n    def ", setup_method_start + 1)
    if next_method_start == -1:
        print("❌ 未找到setup方法结束位置")
        return
    
    # 提取setup方法内容
    setup_method_content = content[setup_method_start:next_method_start]
    
    # 新的setup方法和_setup_storage方法
    new_setup_methods = '''    def setup(self):
        """重写setup方法以使用多模态模型"""
        print("Setting up Multimodal PPO")
        self._setup_multimodal_models()
        print("Setting up Storage") 
        self._setup_storage()
    
    def _setup_storage(self):
        """重写存储设置以支持多模态数据"""
        # 调用父类的存储设置
        super()._setup_storage()
        
        # 注册多模态专用键
        # 融合权重
        self.storage.register_key('fusion_weights', shape=(len(MotionType),), dtype=torch.float)
        
        # 专家动作
        self.storage.register_key('expert_actions', shape=(len(MotionType), self.num_act), dtype=torch.float)
        
        print(f"✅ 已注册多模态存储键: fusion_weights, expert_actions")
'''
    
    # 替换setup方法
    new_content = content[:setup_method_start] + new_setup_methods + content[next_method_start:]
    
    # 写入修改后的文件
    with open(multimodal_ppo_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 已修复MultimodalPPO存储设置")
    print("\n🔧 修复内容:")
    print("1. 添加了_setup_storage方法重写")
    print("2. 注册了fusion_weights和expert_actions存储键")
    print("3. 确保存储系统支持多模态数据")

if __name__ == "__main__":
    fix_multimodal_storage()

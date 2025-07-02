#!/bin/bash
# 多模态运动融合学习系统训练启动脚本

echo "🚀 启动多模态运动融合学习训练"
echo "=================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 基础训练命令
BASE_CMD="python humanoidverse/train_agent.py"

# 多模态配置
MULTIMODAL_CONFIG="--config-name=multimodal_base"

# 基础参数
BASIC_PARAMS="\
+simulator=isaacgym \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
+device=cuda:0"

# 默认参数
DEFAULT_PARAMS="\
project_name=MultimodalMotionFusion \
experiment_name=multimodal_training \
num_envs=256 \
seed=1"

# 检查是否提供了自定义参数
if [ $# -eq 0 ]; then
    echo "使用默认参数..."
    FULL_CMD="$BASE_CMD $MULTIMODAL_CONFIG $BASIC_PARAMS $DEFAULT_PARAMS"
else
    echo "使用自定义参数: $*"
    FULL_CMD="$BASE_CMD $MULTIMODAL_CONFIG $BASIC_PARAMS $*"
fi

echo "执行命令:"
echo "$FULL_CMD"
echo ""

# 执行训练
eval $FULL_CMD

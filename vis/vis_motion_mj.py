import os  # 导入操作系统相关功能模块
import sys  # 导入系统相关功能模块
import time  # 导入时间相关功能模块，用于计时
import argparse  # 导入命令行参数解析模块
import pdb  # 导入Python调试器
import os.path as osp  # 导入路径处理模块并重命名为osp

sys.path.append(os.getcwd())  # 将当前工作目录添加到Python模块搜索路径中，以便导入本地模块

from phc.utils.motion_lib_smpl import MotionLibSMPL  # 导入SMPL动作库工具
from poselib.poselib.skeleton.skeleton3d import SkeletonTree  # 导入3D骨架树结构
import torch  # 导入PyTorch深度学习框架

import numpy as np  # 导入NumPy数值计算库
import math  # 导入数学函数库
from copy import deepcopy  # 导入深拷贝函数
from collections import defaultdict  # 导入默认字典数据结构
import mujoco  # 导入MuJoCo物理引擎
import mujoco.viewer  # 导入MuJoCo可视化工具
from scipy.spatial.transform import Rotation as sRot  # 导入scipy的旋转变换模块并重命名为sRot
from easydict import EasyDict  # 导入EasyDict，使字典可以通过属性访问
from phc.utils.motion_lib_base import FixHeightMode  # 导入高度修复模式枚举
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot  # 导入SMPL机器人模型

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """向MuJoCo场景添加一个胶囊体可视化元素。"""
    if scene.ngeom >= scene.maxgeom:  # 检查场景是否已达到几何体上限
        return
    scene.ngeom += 1  # 增加几何体计数
    # 初始化一个新的胶囊体，并使用mjv_makeConnector添加到场景中
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back(keycode):
    """按键回调函数，处理键盘输入控制模拟过程"""
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused  # 声明使用全局变量
    if chr(keycode) == "T":  # 如果按键是T
        print("Next Motion")  # 打印提示信息
        curr_start += num_motions  # 更新动作起始索引
        motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)  # 加载下一组动作
    elif chr(keycode) == "R":  # 如果按键是R
        print("Reset")  # 打印提示信息
        time_step = 0  # 重置时间步
    elif chr(keycode) == " ":  # 如果按键是空格
        print("Paused")  # 打印提示信息
        paused = not paused  # 切换暂停状态
    else:
        print("not mapped", chr(keycode))  # 打印未映射的按键信息
    
    
        
if __name__ == "__main__":  # 主程序入口
    device = torch.device("cpu")  # 设置PyTorch使用CPU设备
    motion_file = "sample_data/amass_isaac_standing_upright_slim.pkl"  # 设置动作文件路径
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False  # 初始化全局变量
    motion_lib_cfg = EasyDict({  # 使用EasyDict创建动作库配置
                    "motion_file": motion_file,  # 动作文件路径
                    "device": torch.device("cpu"),  # 设备设置为CPU
                    "fix_height": FixHeightMode.full_fix,  # 设置高度修复模式为完全修复
                    "min_length": -1,  # 最小长度限制，-1表示无限制
                    "max_length": -1,  # 最大长度限制，-1表示无限制
                    "im_eval": False,  # 不使用评估模式
                    "multi_thread": False,  # 不使用多线程
                    "smpl_type": 'smpl',  # SMPL模型类型
                    "randomrize_heading": True,  # 随机化朝向
                    "device": device,  # 设备
                })
    
    robot_cfg = {  # 创建SMPL机器人配置字典
        "mesh": False,  # 不使用网格
        "rel_joint_lm": False,  # 不使用相对关节限制
        "upright_start": True,  # 使用直立起始姿势
        "remove_toe": False,  # 不移除脚趾关节
        "real_weight_porpotion_capsules": True,  # 使用真实重量比例的胶囊体
        "real_weight_porpotion_boxes": True,  # 使用真实重量比例的盒体
        "model": "smpl",  # 使用SMPL模型
        "big_ankle": True,  # 使用大踝关节
        "freeze_hand": False,  # 不冻结手部
        "box_body": True,  # 使用盒体表示身体
        "body_params": {},  # 身体参数，空字典
        "joint_params": {},  # 关节参数，空字典
        "geom_params": {},  # 几何体参数，空字典
        "actuator_params": {},  # 执行器参数，空字典
    }
    smpl_robot = SMPL_Robot(  # 创建SMPL机器人实例
        robot_cfg,  # 传入机器人配置
        data_dir="data/smpl",  # 设置SMPL数据目录
    )
    
    gender_beta = np.zeros((17))  # 创建性别和体型参数，全零表示默认中性体型
    smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)  # 从骨架加载SMPL机器人
    test_good = f"/tmp/smpl/test_good.xml"  # 设置临时XML文件路径
    smpl_robot.write_xml(test_good)  # 将机器人模型写入临时XML文件
    smpl_robot.write_xml("test.xml")  # 将机器人模型写入测试XML文件
    sk_tree = SkeletonTree.from_mjcf(test_good)  # 从MuJoCo XML文件创建骨架树
    motion_lib = MotionLibSMPL(motion_lib_cfg)  # 创建SMPL动作库实例
    motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)  # 加载动作数据
    
    mj_model = mujoco.MjModel.from_xml_path(test_good)  # 从XML文件加载MuJoCo模型
    mj_data = mujoco.MjData(mj_model)  # 创建MuJoCo数据实例

    
    # model = load_model_from_path(f"phc/data/assets/mjcf/amp_humanoid.xml")  # 注释掉的代码，用于加载人形模型
    mj_model.opt.timestep = dt  # 设置MuJoCo模型的时间步长
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:  # 启动被动式MuJoCo查看器
        for _ in range(len(sk_tree._node_indices)):  # 遍历骨架树的所有节点
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.01, np.array([1, 0, 0, 1]))  # 为每个节点添加可视化胶囊体
        # 自动在30秒后关闭查看器
        while viewer.is_running():  # 当查看器运行时循环
            step_start = time.time()  # 记录步骤开始时间
            motion_len = motion_lib.get_motion_length(motion_id).item()  # 获取当前动作的长度
            motion_time = time_step % motion_len  # 计算当前动作时间（循环播放）
            motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(device), torch.tensor([motion_time]).to(device))  # 获取当前时间点的动作状态

            # 解包动作状态，包括根位置、根旋转、自由度位置、速度和角速度等多个属性
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

            mj_data.qpos[:3] = root_pos[0].cpu().numpy()  # 设置MuJoCo位置状态（根位置）
            mj_data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]  # 设置MuJoCo四元数旋转状态（根旋转）
            mj_data.qpos[7:] = sRot.from_rotvec(dof_pos[0].cpu().numpy().reshape(-1, 3)).as_euler("XYZ").flatten()  # 将旋转向量转换为欧拉角并设置关节角度
            
            mujoco.mj_forward(mj_model, mj_data)  # 执行MuJoCo前向动力学计算
            if not paused:  # 如果未暂停
                time_step += dt  # 累加时间步

            for i in range(rb_pos.shape[1]):  # 遍历所有刚体
                viewer.user_scn.geoms[i].pos = rb_pos[0, i]  # 更新每个刚体的位置用于可视化
                
            # 同步可视化，应用更改、扰动和从GUI更新选项
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)  # 计算到下一步的等待时间
            if time_until_next_step > 0:  # 如果需要等待
                time.sleep(time_until_next_step)  # 等待直到达到模拟时间步的设定间隔

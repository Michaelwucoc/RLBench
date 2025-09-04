#!/usr/bin/env python3
"""
快速多视角数据提取工具
简化版本，专门用于快速提取类似PerAct格式的多视角数据
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict
from PIL import Image
import cv2

from rlbench import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.backend.utils import task_file_to_task_class


class QuickMultiViewExtractor:
    """快速多视角数据提取器"""
    
    def __init__(self, image_size=(640, 480), headless=True):
        self.image_size = image_size
        self.headless = headless
        self.env = None
        
        # RLBench的5个默认相机视角
        self.view_names = [
            "left_shoulder_camera",
            "right_shoulder_camera", 
            "overhead_camera",
            "wrist_camera",
            "front_camera"
        ]
    
    def setup_environment(self):
        """设置环境"""
        # 配置观测，启用所有相机
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        
        # 设置图像尺寸
        for view_name in self.view_names:
            camera_config = getattr(obs_config, view_name)
            camera_config.image_size = self.image_size
            camera_config.depth_in_meters = False  # 深度值范围0-1
            camera_config.masks_as_one_channel = False
        
        # 创建环境
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete()
        )
        
        self.env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=self.headless
        )
        self.env.launch()
    
    def extract_task_data(self, task_name: str, num_episodes: int = 5, save_dir: str = "/tmp/rlbench_data"):
        """
        提取指定任务的多视角数据
        
        Args:
            task_name: 任务名称
            num_episodes: episode数量
            save_dir: 保存目录
        """
        if self.env is None:
            self.setup_environment()
        
        # 获取任务类
        task_class = task_file_to_task_class(task_name)
        task_env = self.env.get_task(task_class)
        
        # 创建保存目录
        task_save_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_save_dir, exist_ok=True)
        
        print(f"开始提取任务 {task_name} 的数据...")
        
        for episode_idx in range(num_episodes):
            print(f"提取Episode {episode_idx + 1}/{num_episodes}")
            
            # 重置任务
            descriptions, obs = task_env.reset()
            
            # 获取演示数据
            demos, = task_env.get_demos(amount=1, live_demos=True)
            
            # 创建episode目录
            episode_dir = os.path.join(task_save_dir, f"episode_{episode_idx:03d}")
            os.makedirs(episode_dir, exist_ok=True)
            
            # 保存任务描述
            with open(os.path.join(episode_dir, "descriptions.json"), 'w') as f:
                json.dump(descriptions, f, indent=2)
            
            # 提取每个步骤的数据
            episode_data = []
            for step_idx, obs in enumerate(demos):
                step_data = self._extract_step_data(obs, step_idx)
                episode_data.append(step_data)
            
            # 保存episode数据
            self._save_episode_data(episode_data, episode_dir)
            
            # 创建视频
            self._create_episode_videos(episode_data, episode_dir)
        
        print(f"任务 {task_name} 数据提取完成，保存到: {task_save_dir}")
    
    def _extract_step_data(self, obs, step_idx: int) -> Dict:
        """提取单步数据"""
        step_data = {
            "step": step_idx,
            "views": {},
            "low_dim": {}
        }
        
        # 提取每个视角的数据
        for view_name in self.view_names:
            view_data = {}
            
            # RGB图像
            rgb_attr = view_name.replace('_camera', '_rgb')
            if hasattr(obs, rgb_attr):
                view_data["rgb"] = getattr(obs, rgb_attr)
            
            # 深度图像
            depth_attr = view_name.replace('_camera', '_depth')
            if hasattr(obs, depth_attr):
                view_data["depth"] = getattr(obs, depth_attr)
            
            # 点云
            point_cloud_attr = view_name.replace('_camera', '_point_cloud')
            if hasattr(obs, point_cloud_attr):
                view_data["point_cloud"] = getattr(obs, point_cloud_attr)
            
            # 掩码
            mask_attr = view_name.replace('_camera', '_mask')
            if hasattr(obs, mask_attr):
                view_data["mask"] = getattr(obs, mask_attr)
            
            # 相机参数（从misc中获取）
            if hasattr(obs, 'misc') and obs.misc:
                intrinsics_key = f"{view_name}_intrinsics"
                extrinsics_key = f"{view_name}_extrinsics"
                
                if intrinsics_key in obs.misc:
                    view_data["intrinsics"] = obs.misc[intrinsics_key]
                if extrinsics_key in obs.misc:
                    view_data["extrinsics"] = obs.misc[extrinsics_key]
            
            step_data["views"][view_name] = view_data
        
        # 提取低维数据
        low_dim_attrs = [
            "joint_velocities", "joint_positions", "joint_forces",
            "gripper_open", "gripper_pose", "gripper_joint_positions",
            "gripper_touch_forces", "task_low_dim_state"
        ]
        
        for attr in low_dim_attrs:
            if hasattr(obs, attr):
                value = getattr(obs, attr)
                if value is not None:
                    step_data["low_dim"][attr] = value
        
        return step_data
    
    def _save_episode_data(self, episode_data: List[Dict], episode_dir: str):
        """保存episode数据"""
        # 保存为pickle格式（类似PerAct）
        with open(os.path.join(episode_dir, "episode.pkl"), 'wb') as f:
            pickle.dump(episode_data, f)
        
        # 保存为JSON格式（便于查看）
        json_data = []
        for step_data in episode_data:
            json_step = {
                "step": step_data["step"],
                "views": {},
                "low_dim": {}
            }
            
            # 转换numpy数组为列表
            for view_name, view_data in step_data["views"].items():
                json_view = {}
                for key, value in view_data.items():
                    if isinstance(value, np.ndarray):
                        json_view[key] = value.tolist()
                    else:
                        json_view[key] = value
                json_step["views"][view_name] = json_view
            
            for key, value in step_data["low_dim"].items():
                if isinstance(value, np.ndarray):
                    json_step["low_dim"][key] = value.tolist()
                else:
                    json_step["low_dim"][key] = value
            
            json_data.append(json_step)
        
        with open(os.path.join(episode_dir, "episode.json"), 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # 保存图像文件
        self._save_images(episode_data, episode_dir)
    
    def _save_images(self, episode_data: List[Dict], episode_dir: str):
        """保存图像文件"""
        for view_name in self.view_names:
            view_dir = os.path.join(episode_dir, view_name)
            os.makedirs(view_dir, exist_ok=True)
            
            # 创建子目录
            rgb_dir = os.path.join(view_dir, "rgb")
            depth_dir = os.path.join(view_dir, "depth")
            mask_dir = os.path.join(view_dir, "mask")
            
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            # 保存每帧图像
            for step_data in episode_data:
                step_idx = step_data["step"]
                
                if view_name in step_data["views"]:
                    view_data = step_data["views"][view_name]
                    
                    # 保存RGB图像
                    if "rgb" in view_data:
                        rgb_path = os.path.join(rgb_dir, f"frame_{step_idx:06d}.png")
                        Image.fromarray(view_data["rgb"]).save(rgb_path)
                    
                    # 保存深度图像
                    if "depth" in view_data:
                        depth_path = os.path.join(depth_dir, f"frame_{step_idx:06d}.npy")
                        np.save(depth_path, view_data["depth"])
                    
                    # 保存掩码
                    if "mask" in view_data:
                        mask_path = os.path.join(mask_dir, f"frame_{step_idx:06d}.png")
                        mask_image = (view_data["mask"] * 255).astype(np.uint8)
                        Image.fromarray(mask_image).save(mask_path)
            
            # 保存相机参数
            if episode_data and view_name in episode_data[0]["views"]:
                view_data = episode_data[0]["views"][view_name]
                camera_params = {}
                
                if "intrinsics" in view_data:
                    camera_params["intrinsics"] = view_data["intrinsics"].tolist()
                if "extrinsics" in view_data:
                    camera_params["extrinsics"] = view_data["extrinsics"].tolist()
                
                if camera_params:
                    params_path = os.path.join(view_dir, "camera_params.json")
                    with open(params_path, 'w') as f:
                        json.dump(camera_params, f, indent=2)
    
    def _create_episode_videos(self, episode_data: List[Dict], episode_dir: str):
        """创建episode视频"""
        video_dir = os.path.join(episode_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        for view_name in self.view_names:
            # 收集该视角的所有RGB帧
            rgb_frames = []
            for step_data in episode_data:
                if (view_name in step_data["views"] and 
                    "rgb" in step_data["views"][view_name]):
                    rgb_frames.append(step_data["views"][view_name]["rgb"])
            
            if not rgb_frames:
                continue
            
            # 创建视频
            video_path = os.path.join(video_dir, f"{view_name}.mp4")
            self._create_video_from_frames(rgb_frames, video_path)
    
    def _create_video_from_frames(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """从帧列表创建视频"""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # 转换RGB到BGR
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
        
        video_writer.release()
        print(f"视频已保存: {output_path}")
    
    def extract_all_tasks(self, task_names: List[str] = None, num_episodes: int = 3, save_dir: str = "/tmp/rlbench_data"):
        """提取多个任务的数据"""
        if task_names is None:
            # 使用一些常用任务
            task_names = [
                "ReachTarget",
                "PickUpCup", 
                "PutIntoDrawer",
                "StackBlocks",
                "OpenDoor"
            ]
        
        for task_name in task_names:
            try:
                self.extract_task_data(task_name, num_episodes, save_dir)
            except Exception as e:
                print(f"提取任务 {task_name} 时出错: {e}")
                continue
    
    def shutdown(self):
        """关闭环境"""
        if self.env:
            self.env.shutdown()


def main():
    """主函数"""
    # 创建提取器
    extractor = QuickMultiViewExtractor(
        image_size=(640, 480),
        headless=True
    )
    
    try:
        # 提取单个任务的数据
        print("开始提取多视角数据...")
        extractor.extract_task_data(
            task_name="ReachTarget",
            num_episodes=3,
            save_dir="/tmp/rlbench_quick_data"
        )
        
        # 或者提取多个任务
        # extractor.extract_all_tasks(
        #     task_names=["ReachTarget", "PickUpCup"],
        #     num_episodes=2,
        #     save_dir="/tmp/rlbench_quick_data"
        # )
        
        print("数据提取完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
自定义相机视角扩展工具
允许用户轻松添加更多相机视角来增强多视角数据提取

Remodified By Milk, Version 3.9.2 on Sep 4
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode

from rlbench import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.backend.utils import task_file_to_task_class


@dataclass
class CameraConfig:
    """相机配置"""
    name: str
    position: Tuple[float, float, float]  # (x, y, z)
    orientation: Tuple[float, float, float]  # (roll, pitch, yaw) in degrees
    resolution: Tuple[int, int] = (640, 480)
    fov: float = 60.0
    near_plane: float = 0.01
    far_plane: float = 10.0


class CustomCameraExtractor:
    """自定义相机提取器"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.env = None
        self.scene = None
        self.custom_cameras = {}
        
        # RLBench默认相机
        self.default_cameras = [
            "left_shoulder_camera",
            "right_shoulder_camera", 
            "overhead_camera",
            "wrist_camera",
            "front_camera"
        ]
    
    def setup_environment(self):
        """设置环境"""
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        
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
        self.scene = self.env._scene
    
    def add_custom_cameras(self, camera_configs: List[CameraConfig]):
        """添加自定义相机"""
        for config in camera_configs:
            # 创建相机
            camera = VisionSensor.create(config.resolution)
            
            # 设置位置和朝向
            camera.set_position(config.position)
            camera.set_orientation(np.radians(config.orientation))
            
            # 设置相机参数
            camera.set_near_clipping_plane(config.near_plane)
            camera.set_far_clipping_plane(config.far_plane)
            camera.set_render_mode(RenderMode.OPENGL3)
            
            self.custom_cameras[config.name] = camera
            print(f"添加自定义相机: {config.name}")
    
    def get_camera_data(self, camera_name: str) -> Dict:
        """获取相机数据"""
        if camera_name in self.custom_cameras:
            camera = self.custom_cameras[camera_name]
        elif camera_name in self.default_cameras:
            # 从场景中获取默认相机
            camera_attr = f"_cam_{camera_name.replace('_camera', '')}"
            if hasattr(self.scene, camera_attr):
                camera = getattr(self.scene, camera_attr)
            else:
                return None
        else:
            return None
        
        try:
            # 捕获图像
            rgb_image = camera.capture_rgb()
            depth_image = camera.capture_depth()
            
            # 获取相机参数
            intrinsics = camera.get_intrinsic_matrix()
            extrinsics = camera.get_matrix()
            
            return {
                "rgb": (rgb_image * 255).astype(np.uint8),
                "depth": depth_image,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "near_plane": camera.get_near_clipping_plane(),
                "far_plane": camera.get_far_clipping_plane()
            }
        except Exception as e:
            print(f"获取相机 {camera_name} 数据时出错: {e}")
            return None
    
    def extract_multi_view_data(self, task_name: str, num_episodes: int = 1, save_dir: str = "/tmp/custom_camera_data"):
        """提取多视角数据"""
        if self.env is None:
            self.setup_environment()
        
        task_class = task_file_to_task_class(task_name)
        task_env = self.env.get_task(task_class)
        
        # 创建保存目录
        task_save_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_save_dir, exist_ok=True)
        
        all_camera_names = self.default_cameras + list(self.custom_cameras.keys())
        
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
                step_data = {
                    "step": step_idx,
                    "cameras": {}
                }
                
                # 提取默认相机数据（从obs中获取）
                for camera_name in self.default_cameras:
                    camera_data = self._extract_default_camera_data(obs, camera_name)
                    if camera_data:
                        step_data["cameras"][camera_name] = camera_data
                
                # 提取自定义相机数据
                for camera_name in self.custom_cameras.keys():
                    camera_data = self.get_camera_data(camera_name)
                    if camera_data:
                        step_data["cameras"][camera_name] = camera_data
                
                episode_data.append(step_data)
            
            # 保存episode数据
            self._save_episode_data(episode_data, episode_dir, all_camera_names)
        
        print(f"任务 {task_name} 数据提取完成")
    
    def _extract_default_camera_data(self, obs, camera_name: str) -> Dict:
        """提取默认相机数据"""
        camera_data = {}
        
        # RGB图像
        rgb_attr = camera_name.replace('_camera', '_rgb')
        if hasattr(obs, rgb_attr):
            camera_data["rgb"] = getattr(obs, rgb_attr)
        
        # 深度图像
        depth_attr = camera_name.replace('_camera', '_depth')
        if hasattr(obs, depth_attr):
            camera_data["depth"] = getattr(obs, depth_attr)
        
        # 相机参数
        if hasattr(obs, 'misc') and obs.misc:
            intrinsics_key = f"{camera_name}_intrinsics"
            extrinsics_key = f"{camera_name}_extrinsics"
            
            if intrinsics_key in obs.misc:
                camera_data["intrinsics"] = obs.misc[intrinsics_key]
            if extrinsics_key in obs.misc:
                camera_data["extrinsics"] = obs.misc[extrinsics_key]
        
        return camera_data if camera_data else None
    
    def _save_episode_data(self, episode_data: List[Dict], episode_dir: str, camera_names: List[str]):
        """保存episode数据"""
        # 保存为JSON格式
        json_data = []
        for step_data in episode_data:
            json_step = {
                "step": step_data["step"],
                "cameras": {}
            }
            
            for camera_name, camera_data in step_data["cameras"].items():
                json_camera = {}
                for key, value in camera_data.items():
                    if isinstance(value, np.ndarray):
                        json_camera[key] = value.tolist()
                    else:
                        json_camera[key] = value
                json_step["cameras"][camera_name] = json_camera
            
            json_data.append(json_step)
        
        with open(os.path.join(episode_dir, "episode.json"), 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # 保存图像文件
        for camera_name in camera_names:
            camera_dir = os.path.join(episode_dir, camera_name)
            os.makedirs(camera_dir, exist_ok=True)
            
            rgb_dir = os.path.join(camera_dir, "rgb")
            depth_dir = os.path.join(camera_dir, "depth")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            
            # 保存每帧图像
            for step_data in episode_data:
                step_idx = step_data["step"]
                
                if camera_name in step_data["cameras"]:
                    camera_data = step_data["cameras"][camera_name]
                    
                    # 保存RGB图像
                    if "rgb" in camera_data:
                        from PIL import Image
                        rgb_path = os.path.join(rgb_dir, f"frame_{step_idx:06d}.png")
                        Image.fromarray(camera_data["rgb"]).save(rgb_path)
                    
                    # 保存深度图像
                    if "depth" in camera_data:
                        depth_path = os.path.join(depth_dir, f"frame_{step_idx:06d}.npy")
                        np.save(depth_path, camera_data["depth"])
            
            # 保存相机参数
            if episode_data and camera_name in episode_data[0]["cameras"]:
                camera_data = episode_data[0]["cameras"][camera_name]
                camera_params = {}
                
                for key in ["intrinsics", "extrinsics", "near_plane", "far_plane"]:
                    if key in camera_data:
                        if isinstance(camera_data[key], np.ndarray):
                            camera_params[key] = camera_data[key].tolist()
                        else:
                            camera_params[key] = camera_data[key]
                
                if camera_params:
                    params_path = os.path.join(camera_dir, "camera_params.json")
                    with open(params_path, 'w') as f:
                        json.dump(camera_params, f, indent=2)
    
    def shutdown(self):
        """关闭环境"""
        if self.env:
            self.env.shutdown()


def create_enhanced_camera_configs() -> List[CameraConfig]:
    """创建增强的相机配置"""
    return [
        # 侧面视角
        CameraConfig(
            name="side_left_camera",
            position=(0.6, 0.8, 0.4),
            orientation=(0, -20, 90),
            resolution=(640, 480)
        ),
        CameraConfig(
            name="side_right_camera",
            position=(-0.6, 0.8, 0.4),
            orientation=(0, -20, -90),
            resolution=(640, 480)
        ),
        
        # 俯视视角
        CameraConfig(
            name="top_down_camera",
            position=(0, 0, 1.5),
            orientation=(0, -90, 0),
            resolution=(640, 480)
        ),
        
        # 近距离视角
        CameraConfig(
            name="close_up_camera",
            position=(0, 0.4, 0.1),
            orientation=(0, 0, 0),
            resolution=(640, 480)
        ),
        
        # 斜角视角
        CameraConfig(
            name="diagonal_camera",
            position=(0.3, 0.6, 0.5),
            orientation=(0, -30, 45),
            resolution=(640, 480)
        ),
        
        # 后方视角
        CameraConfig(
            name="back_camera",
            position=(0, -0.8, 0.4),
            orientation=(0, -20, 180),
            resolution=(640, 480)
        )
    ]


def main():
    """主函数"""
    # 创建自定义相机配置
    custom_cameras = create_enhanced_camera_configs()
    
    # 创建提取器
    extractor = CustomCameraExtractor(headless=True)
    
    try:
        # 设置环境
        extractor.setup_environment()
        
        # 添加自定义相机
        extractor.add_custom_cameras(custom_cameras)
        
        # 提取数据
        print("开始提取增强多视角数据...")
        extractor.extract_multi_view_data(
            task_name="ReachTarget",
            num_episodes=2,
            save_dir="/tmp/custom_camera_data"
        )
        
        print("数据提取完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.shutdown()


if __name__ == "__main__":
    main()

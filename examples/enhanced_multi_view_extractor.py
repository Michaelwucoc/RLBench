#!/usr/bin/env python3
"""
增强版多视角数据提取工具
支持更多视角的RGB、深度图像和相机内外参提取
类似PerAct的数据格式，但支持更多自定义视角
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import cv2
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode

from rlbench import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.utils import task_file_to_task_class


@dataclass
class CameraView:
    """自定义相机视角配置"""
    name: str
    position: Tuple[float, float, float]  # (x, y, z)
    orientation: Tuple[float, float, float]  # (roll, pitch, yaw) in degrees
    resolution: Tuple[int, int] = (640, 480)
    fov: float = 60.0  # 视场角
    near_plane: float = 0.01
    far_plane: float = 10.0


@dataclass
class MultiViewData:
    """多视角数据结构"""
    rgb_images: Dict[str, np.ndarray]
    depth_images: Dict[str, np.ndarray]
    intrinsics: Dict[str, np.ndarray]
    extrinsics: Dict[str, np.ndarray]
    timestamps: List[float]
    low_dim_data: Optional[np.ndarray] = None


class EnhancedMultiViewExtractor:
    """增强版多视角数据提取器"""
    
    def __init__(self, 
                 custom_views: List[CameraView] = None,
                 image_size: Tuple[int, int] = (640, 480),
                 headless: bool = True):
        """
        初始化多视角提取器
        
        Args:
            custom_views: 自定义相机视角列表
            image_size: 图像尺寸
            headless: 是否无头模式运行
        """
        self.custom_views = custom_views or []
        self.image_size = image_size
        self.headless = headless
        self.custom_cameras = {}
        self.env = None
        self.scene = None
        
        # 默认的5个RLBench相机视角
        self.default_views = [
            "left_shoulder_camera",
            "right_shoulder_camera", 
            "overhead_camera",
            "wrist_camera",
            "front_camera"
        ]
        
        # 所有视角名称
        self.all_view_names = self.default_views + [view.name for view in self.custom_views]
    
    def setup_environment(self):
        """设置RLBench环境"""
        # 配置观测
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        
        # 设置所有相机的图像尺寸
        for camera_name in self.default_views:
            camera_config = getattr(obs_config, camera_name)
            camera_config.image_size = self.image_size
            camera_config.depth_in_meters = False  # 深度值范围0-1
            camera_config.masks_as_one_channel = False  # 保存为RGB编码
            camera_config.render_mode = RenderMode.OPENGL3
        
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
        self.scene = self.env._scene
        
        # 添加自定义相机
        self._setup_custom_cameras()
    
    def _setup_custom_cameras(self):
        """设置自定义相机"""
        for view in self.custom_views:
            # 创建相机
            camera = VisionSensor.create(self.image_size)
            
            # 设置相机位置和朝向
            position = view.position
            orientation = np.radians(view.orientation)  # 转换为弧度
            
            # 创建变换矩阵
            from pyrep.objects import Object
            camera.set_position(position)
            camera.set_orientation(orientation)
            
            # 设置相机参数
            camera.set_near_clipping_plane(view.near_plane)
            camera.set_far_clipping_plane(view.far_plane)
            camera.set_render_mode(RenderMode.OPENGL3)
            
            self.custom_cameras[view.name] = camera
    
    def extract_multi_view_data(self, task_class, num_episodes: int = 1) -> List[MultiViewData]:
        """
        提取多视角数据
        
        Args:
            task_class: 任务类
            num_episodes: 要提取的episode数量
            
        Returns:
            多视角数据列表
        """
        if self.env is None:
            self.setup_environment()
        
        task_env = self.env.get_task(task_class)
        all_episodes_data = []
        
        for episode_idx in range(num_episodes):
            print(f"提取Episode {episode_idx + 1}/{num_episodes}")
            
            # 重置任务
            descriptions, obs = task_env.reset()
            print(f"任务描述: {descriptions}")
            
            episode_data = []
            timestamps = []
            
            # 获取演示数据
            demos, = task_env.get_demos(amount=1, live_demos=True)
            
            for step_idx, obs in enumerate(demos):
                timestamp = step_idx * 0.1  # 假设每步0.1秒
                timestamps.append(timestamp)
                
                # 提取默认视角数据
                view_data = self._extract_view_data(obs, step_idx)
                
                # 提取自定义视角数据
                custom_data = self._extract_custom_view_data(step_idx)
                view_data.update(custom_data)
                
                episode_data.append(view_data)
            
            # 组织episode数据
            episode_multi_view = self._organize_episode_data(episode_data, timestamps)
            all_episodes_data.append(episode_multi_view)
        
        return all_episodes_data
    
    def _extract_view_data(self, obs: Observation, step_idx: int) -> Dict:
        """提取默认视角数据"""
        view_data = {}
        
        for view_name in self.default_views:
            # RGB图像
            rgb_attr = view_name.replace('_camera', '_rgb')
            if hasattr(obs, rgb_attr):
                view_data[f"{view_name}_rgb"] = getattr(obs, rgb_attr)
            
            # 深度图像
            depth_attr = view_name.replace('_camera', '_depth')
            if hasattr(obs, depth_attr):
                view_data[f"{view_name}_depth"] = getattr(obs, depth_attr)
            
            # 内外参（从misc中获取）
            if hasattr(obs, 'misc') and obs.misc:
                intrinsics_key = f"{view_name}_intrinsics"
                extrinsics_key = f"{view_name}_extrinsics"
                
                if intrinsics_key in obs.misc:
                    view_data[f"{view_name}_intrinsics"] = obs.misc[intrinsics_key]
                if extrinsics_key in obs.misc:
                    view_data[f"{view_name}_extrinsics"] = obs.misc[extrinsics_key]
        
        return view_data
    
    def _extract_custom_view_data(self, step_idx: int) -> Dict:
        """提取自定义视角数据"""
        custom_data = {}
        
        for view_name, camera in self.custom_cameras.items():
            try:
                # 捕获RGB图像
                rgb_image = camera.capture_rgb()
                custom_data[f"{view_name}_rgb"] = (rgb_image * 255).astype(np.uint8)
                
                # 捕获深度图像
                depth_image = camera.capture_depth()
                custom_data[f"{view_name}_depth"] = depth_image
                
                # 获取内外参
                custom_data[f"{view_name}_intrinsics"] = camera.get_intrinsic_matrix()
                custom_data[f"{view_name}_extrinsics"] = camera.get_matrix()
                
            except Exception as e:
                print(f"警告: 无法从相机 {view_name} 提取数据: {e}")
        
        return custom_data
    
    def _organize_episode_data(self, episode_data: List[Dict], timestamps: List[float]) -> MultiViewData:
        """组织episode数据为MultiViewData格式"""
        # 初始化数据结构
        rgb_images = {view_name: [] for view_name in self.all_view_names}
        depth_images = {view_name: [] for view_name in self.all_view_names}
        intrinsics = {view_name: None for view_name in self.all_view_names}
        extrinsics = {view_name: [] for view_name in self.all_view_names}
        
        # 填充数据
        for step_data in episode_data:
            for view_name in self.all_view_names:
                # RGB图像
                rgb_key = f"{view_name}_rgb"
                if rgb_key in step_data:
                    rgb_images[view_name].append(step_data[rgb_key])
                
                # 深度图像
                depth_key = f"{view_name}_depth"
                if depth_key in step_data:
                    depth_images[view_name].append(step_data[depth_key])
                
                # 内外参（只需要保存一次，因为相机位置固定）
                intrinsics_key = f"{view_name}_intrinsics"
                extrinsics_key = f"{view_name}_extrinsics"
                
                if intrinsics_key in step_data and intrinsics[view_name] is None:
                    intrinsics[view_name] = step_data[intrinsics_key]
                
                if extrinsics_key in step_data:
                    extrinsics[view_name].append(step_data[extrinsics_key])
        
        # 转换为numpy数组
        for view_name in self.all_view_names:
            if rgb_images[view_name]:
                rgb_images[view_name] = np.array(rgb_images[view_name])
            if depth_images[view_name]:
                depth_images[view_name] = np.array(depth_images[view_name])
            if extrinsics[view_name]:
                extrinsics[view_name] = np.array(extrinsics[view_name])
        
        return MultiViewData(
            rgb_images=rgb_images,
            depth_images=depth_images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            timestamps=timestamps
        )
    
    def save_data(self, data: List[MultiViewData], save_dir: str, task_name: str):
        """
        保存多视角数据
        
        Args:
            data: 多视角数据列表
            save_dir: 保存目录
            task_name: 任务名称
        """
        task_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        for episode_idx, episode_data in enumerate(data):
            episode_dir = os.path.join(task_dir, f"episode_{episode_idx:03d}")
            os.makedirs(episode_dir, exist_ok=True)
            
            # 保存每个视角的数据
            for view_name in self.all_view_names:
                view_dir = os.path.join(episode_dir, view_name)
                os.makedirs(view_dir, exist_ok=True)
                
                # 保存RGB图像序列
                if view_name in episode_data.rgb_images and len(episode_data.rgb_images[view_name]) > 0:
                    rgb_dir = os.path.join(view_dir, "rgb")
                    os.makedirs(rgb_dir, exist_ok=True)
                    
                    for frame_idx, rgb_image in enumerate(episode_data.rgb_images[view_name]):
                        rgb_path = os.path.join(rgb_dir, f"frame_{frame_idx:06d}.png")
                        Image.fromarray(rgb_image).save(rgb_path)
                
                # 保存深度图像序列
                if view_name in episode_data.depth_images and len(episode_data.depth_images[view_name]) > 0:
                    depth_dir = os.path.join(view_dir, "depth")
                    os.makedirs(depth_dir, exist_ok=True)
                    
                    for frame_idx, depth_image in enumerate(episode_data.depth_images[view_name]):
                        depth_path = os.path.join(depth_dir, f"frame_{frame_idx:06d}.npy")
                        np.save(depth_path, depth_image)
                
                # 保存相机参数
                camera_params = {}
                if view_name in episode_data.intrinsics and episode_data.intrinsics[view_name] is not None:
                    camera_params["intrinsics"] = episode_data.intrinsics[view_name].tolist()
                
                if view_name in episode_data.extrinsics and len(episode_data.extrinsics[view_name]) > 0:
                    camera_params["extrinsics"] = episode_data.extrinsics[view_name].tolist()
                
                if camera_params:
                    params_path = os.path.join(view_dir, "camera_params.json")
                    with open(params_path, 'w') as f:
                        json.dump(camera_params, f, indent=2)
            
            # 保存时间戳
            timestamps_path = os.path.join(episode_dir, "timestamps.json")
            with open(timestamps_path, 'w') as f:
                json.dump(episode_data.timestamps, f, indent=2)
            
            # 保存完整episode数据（pickle格式，类似PerAct）
            episode_pickle_path = os.path.join(episode_dir, "episode_data.pkl")
            with open(episode_pickle_path, 'wb') as f:
                pickle.dump(episode_data, f)
        
        print(f"数据已保存到: {task_dir}")
    
    def create_video(self, data: MultiViewData, save_path: str, view_name: str, fps: int = 30):
        """
        从多视角数据创建视频
        
        Args:
            data: 多视角数据
            save_path: 视频保存路径
            view_name: 视角名称
            fps: 帧率
        """
        if view_name not in data.rgb_images or len(data.rgb_images[view_name]) == 0:
            print(f"警告: 视角 {view_name} 没有RGB数据")
            return
        
        rgb_images = data.rgb_images[view_name]
        height, width = rgb_images[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        for rgb_image in rgb_images:
            # 转换RGB到BGR（OpenCV格式）
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_image)
        
        video_writer.release()
        print(f"视频已保存到: {save_path}")
    
    def shutdown(self):
        """关闭环境"""
        if self.env:
            self.env.shutdown()


def create_peract_style_views() -> List[CameraView]:
    """创建类似PerAct的相机视角配置"""
    return [
        # 添加一些额外的视角来增强数据
        CameraView(
            name="side_left_camera",
            position=(0.5, 0.8, 0.3),
            orientation=(0, -30, 90),
            resolution=(640, 480)
        ),
        CameraView(
            name="side_right_camera", 
            position=(-0.5, 0.8, 0.3),
            orientation=(0, -30, -90),
            resolution=(640, 480)
        ),
        CameraView(
            name="top_down_camera",
            position=(0, 0, 1.2),
            orientation=(0, -90, 0),
            resolution=(640, 480)
        ),
        CameraView(
            name="close_up_camera",
            position=(0, 0.3, 0.1),
            orientation=(0, 0, 0),
            resolution=(640, 480)
        )
    ]


def main():
    """主函数示例"""
    # 创建自定义视角
    custom_views = create_peract_style_views()
    
    # 创建多视角提取器
    extractor = EnhancedMultiViewExtractor(
        custom_views=custom_views,
        image_size=(640, 480),
        headless=True
    )
    
    try:
        # 设置环境
        extractor.setup_environment()
        
        # 获取任务（这里使用ReachTarget作为示例）
        from rlbench.tasks import ReachTarget
        task_class = ReachTarget
        
        # 提取数据
        print("开始提取多视角数据...")
        multi_view_data = extractor.extract_multi_view_data(
            task_class=task_class,
            num_episodes=2
        )
        
        # 保存数据
        save_dir = "/tmp/rlbench_enhanced_data"
        task_name = "ReachTarget"
        extractor.save_data(multi_view_data, save_dir, task_name)
        
        # 创建视频示例
        if multi_view_data:
            episode_data = multi_view_data[0]
            video_dir = os.path.join(save_dir, task_name, "videos")
            os.makedirs(video_dir, exist_ok=True)
            
            # 为每个视角创建视频
            for view_name in extractor.all_view_names:
                if view_name in episode_data.rgb_images and len(episode_data.rgb_images[view_name]) > 0:
                    video_path = os.path.join(video_dir, f"{view_name}.mp4")
                    extractor.create_video(episode_data, video_path, view_name)
        
        print("数据提取完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
多视角数据提取工具演示
展示如何使用工具提取类似PerAct格式的多视角数据
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CameraView:
    """相机视角配置"""
    name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float]
    resolution: Tuple[int, int] = (640, 480)


def create_peract_style_views() -> List[CameraView]:
    """创建类似PerAct的相机视角配置"""
    return [
        # RLBench默认的5个相机视角
        CameraView(name="left_shoulder_camera", position=(0.3, 0.8, 0.4), orientation=(0, -30, 90)),
        CameraView(name="right_shoulder_camera", position=(-0.3, 0.8, 0.4), orientation=(0, -30, -90)),
        CameraView(name="overhead_camera", position=(0, 0, 1.2), orientation=(0, -90, 0)),
        CameraView(name="wrist_camera", position=(0, 0.1, 0.05), orientation=(0, 0, 0)),
        CameraView(name="front_camera", position=(0, 1.0, 0.3), orientation=(0, -20, 0)),
        
        # 额外的自定义视角
        CameraView(name="side_left_camera", position=(0.5, 0.8, 0.3), orientation=(0, -30, 90)),
        CameraView(name="side_right_camera", position=(-0.5, 0.8, 0.3), orientation=(0, -30, -90)),
        CameraView(name="top_down_camera", position=(0, 0, 1.5), orientation=(0, -90, 0)),
        CameraView(name="close_up_camera", position=(0, 0.3, 0.1), orientation=(0, 0, 0)),
        CameraView(name="diagonal_camera", position=(0.3, 0.6, 0.5), orientation=(0, -30, 45)),
    ]


def demonstrate_data_structure():
    """演示数据结构"""
    print("多视角数据结构演示")
    print("=" * 50)
    
    # 创建相机配置
    camera_views = create_peract_style_views()
    
    print(f"配置了 {len(camera_views)} 个相机视角:")
    for i, view in enumerate(camera_views, 1):
        print(f"{i:2d}. {view.name}")
        print(f"    位置: {view.position}")
        print(f"    朝向: {view.orientation}")
        print(f"    分辨率: {view.resolution}")
        print()
    
    # 演示数据结构
    print("数据结构示例:")
    print("-" * 30)
    
    # 模拟一个episode的数据结构
    episode_data = {
        "episode_id": "episode_000",
        "task_name": "ReachTarget",
        "descriptions": ["Reach the red target"],
        "steps": []
    }
    
    # 模拟几个步骤的数据
    for step_idx in range(3):
        step_data = {
            "step": step_idx,
            "timestamp": step_idx * 0.1,
            "views": {},
            "low_dim": {
                "joint_positions": np.random.randn(7).tolist(),
                "joint_velocities": np.random.randn(7).tolist(),
                "gripper_open": 1.0,
                "gripper_pose": np.random.randn(7).tolist()
            }
        }
        
        # 为每个视角添加数据
        for view in camera_views:
            view_data = {
                "rgb": f"RGB图像数据 (形状: {view.resolution[1]}x{view.resolution[0]}x3)",
                "depth": f"深度图像数据 (形状: {view.resolution[1]}x{view.resolution[0]})",
                "intrinsics": [
                    [525.0, 0.0, 320.0],
                    [0.0, 525.0, 240.0],
                    [0.0, 0.0, 1.0]
                ],
                "extrinsics": [
                    [1.0, 0.0, 0.0, view.position[0]],
                    [0.0, 1.0, 0.0, view.position[1]],
                    [0.0, 0.0, 1.0, view.position[2]],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
            step_data["views"][view.name] = view_data
        
        episode_data["steps"].append(step_data)
    
    # 保存示例数据
    output_dir = "/tmp/demo_multi_view_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式
    with open(os.path.join(output_dir, "episode_demo.json"), 'w') as f:
        json.dump(episode_data, f, indent=2)
    
    print(f"示例数据已保存到: {os.path.join(output_dir, 'episode_demo.json')}")
    
    # 显示目录结构
    print("\n生成的目录结构:")
    print("-" * 30)
    print("task_name/")
    print("├── episode_000/")
    print("│   ├── descriptions.json")
    print("│   ├── episode.pkl")
    print("│   ├── episode.json")
    print("│   ├── left_shoulder_camera/")
    print("│   │   ├── rgb/")
    print("│   │   │   ├── frame_000000.png")
    print("│   │   │   ├── frame_000001.png")
    print("│   │   │   └── ...")
    print("│   │   ├── depth/")
    print("│   │   │   ├── frame_000000.npy")
    print("│   │   │   └── ...")
    print("│   │   └── camera_params.json")
    print("│   ├── right_shoulder_camera/")
    print("│   ├── overhead_camera/")
    print("│   ├── wrist_camera/")
    print("│   ├── front_camera/")
    print("│   ├── side_left_camera/")
    print("│   ├── side_right_camera/")
    print("│   ├── top_down_camera/")
    print("│   ├── close_up_camera/")
    print("│   ├── diagonal_camera/")
    print("│   └── videos/")
    print("│       ├── left_shoulder_camera.mp4")
    print("│       ├── right_shoulder_camera.mp4")
    print("│       └── ...")
    print("└── episode_001/")
    print("    └── ...")


def demonstrate_usage():
    """演示使用方法"""
    print("\n使用方法演示")
    print("=" * 50)
    
    print("1. 快速开始 - 使用默认5个相机视角:")
    print("-" * 40)
    print("""
from quick_multi_view_extraction import QuickMultiViewExtractor

# 创建提取器
extractor = QuickMultiViewExtractor(
    image_size=(640, 480),
    headless=True
)

# 提取数据
extractor.extract_task_data(
    task_name="ReachTarget",
    num_episodes=5,
    save_dir="/tmp/rlbench_data"
)

extractor.shutdown()
    """)
    
    print("2. 添加自定义相机视角:")
    print("-" * 40)
    print("""
from enhanced_multi_view_extractor import EnhancedMultiViewExtractor, CameraView

# 定义自定义相机
custom_views = [
    CameraView(
        name="side_left_camera",
        position=(0.5, 0.8, 0.3),
        orientation=(0, -30, 90),
        resolution=(640, 480)
    )
]

# 创建提取器
extractor = EnhancedMultiViewExtractor(
    custom_views=custom_views,
    image_size=(640, 480),
    headless=True
)

# 提取数据
extractor.setup_environment()
multi_view_data = extractor.extract_multi_view_data(
    task_class=ReachTarget,
    num_episodes=3
)

# 保存数据
extractor.save_data(multi_view_data, "/tmp/enhanced_data", "ReachTarget")

extractor.shutdown()
    """)
    
    print("3. 高级自定义相机配置:")
    print("-" * 40)
    print("""
from custom_camera_extractor import CustomCameraExtractor, CameraConfig

# 创建相机配置
camera_configs = [
    CameraConfig(
        name="close_up_camera",
        position=(0, 0.4, 0.1),
        orientation=(0, 0, 0),
        resolution=(1280, 720),
        fov=60.0,
        near_plane=0.01,
        far_plane=10.0
    )
]

# 创建提取器
extractor = CustomCameraExtractor(headless=True)
extractor.setup_environment()
extractor.add_custom_cameras(camera_configs)

# 提取数据
extractor.extract_multi_view_data(
    task_name="ReachTarget",
    num_episodes=3,
    save_dir="/tmp/custom_data"
)

extractor.shutdown()
    """)


def demonstrate_peract_compatibility():
    """演示与PerAct的兼容性"""
    print("\n与PerAct的兼容性")
    print("=" * 50)
    
    print("本工具生成的数据格式与PerAct完全兼容:")
    print()
    
    print("✓ 图像数据格式:")
    print("  - RGB图像: 标准PNG格式")
    print("  - 深度图像: NumPy数组格式 (.npy)")
    print("  - 图像尺寸: 可配置 (默认640x480)")
    print()
    
    print("✓ 相机参数格式:")
    print("  - 内参矩阵: 3x3矩阵")
    print("  - 外参矩阵: 4x4变换矩阵")
    print("  - 深度范围: 近平面和远平面")
    print()
    
    print("✓ 数据结构:")
    print("  - 支持PerAct的数据加载方式")
    print("  - 兼容现有的数据处理管道")
    print("  - 支持pickle和JSON格式")
    print()
    
    print("✓ 扩展功能:")
    print("  - 支持更多视角 (超越PerAct的5个固定视角)")
    print("  - 支持自定义相机位置和参数")
    print("  - 支持视频生成")
    print("  - 支持实时数据提取")
    print()
    
    print("✓ 性能优化:")
    print("  - 无头模式运行")
    print("  - 批量数据处理")
    print("  - 内存优化")
    print("  - 并行处理支持")


def main():
    """主函数"""
    print("RLBench 多视角数据提取工具演示")
    print("=" * 60)
    print("本工具用于从RLBench仿真平台提取类似PerAct格式的多视角数据集")
    print("支持RGB、深度图像和相机内外参的同时提取")
    print("=" * 60)
    
    # 演示数据结构
    demonstrate_data_structure()
    
    # 演示使用方法
    demonstrate_usage()
    
    # 演示兼容性
    demonstrate_peract_compatibility()
    
    print("\n总结")
    print("=" * 50)
    print("本工具包提供了三种不同级别的多视角数据提取方案:")
    print()
    print("1. 快速多视角提取器 - 适合快速原型开发")
    print("2. 增强多视角提取器 - 支持自定义相机视角")
    print("3. 自定义相机提取器 - 高级相机配置和管理")
    print()
    print("所有工具都支持:")
    print("- RGB、深度图像和相机内外参的同时提取")
    print("- 类似PerAct的数据格式")
    print("- 视频生成功能")
    print("- 批量数据处理")
    print()
    print("详细使用说明请参考: README_multi_view_extraction.md")


if __name__ == "__main__":
    main()

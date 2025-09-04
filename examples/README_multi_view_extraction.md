# RLBench 多视角数据提取工具

本工具包提供了三种不同级别的多视角数据提取方案，用于从RLBench仿真平台提取类似PerAct格式的多视角数据集。

## 工具概述

### 1. 快速多视角提取器 (`quick_multi_view_extraction.py`)
- **用途**: 快速提取RLBench默认5个相机视角的数据
- **特点**: 简单易用，适合快速原型开发
- **输出**: RGB图像、深度图像、相机内外参、低维状态数据

### 2. 增强多视角提取器 (`enhanced_multi_view_extractor.py`)
- **用途**: 支持自定义相机视角的高级数据提取
- **特点**: 可添加任意数量的自定义相机，支持视频生成
- **输出**: 完整的多视角数据集，包括自定义视角

### 3. 自定义相机提取器 (`custom_camera_extractor.py`)
- **用途**: 专门用于添加和管理自定义相机视角
- **特点**: 灵活的相机配置，支持复杂的相机布局
- **输出**: 增强的多视角数据，包含更多视角信息

## 安装要求

```bash
pip install rlbench pyrep numpy pillow opencv-python
```

## 使用方法

### 快速开始 - 使用默认5个相机视角

```python
from quick_multi_view_extraction import QuickMultiViewExtractor

# 创建提取器
extractor = QuickMultiViewExtractor(
    image_size=(640, 480),
    headless=True
)

# 提取单个任务的数据
extractor.extract_task_data(
    task_name="ReachTarget",
    num_episodes=5,
    save_dir="/tmp/rlbench_data"
)

# 提取多个任务的数据
extractor.extract_all_tasks(
    task_names=["ReachTarget", "PickUpCup", "StackBlocks"],
    num_episodes=3,
    save_dir="/tmp/rlbench_data"
)

extractor.shutdown()
```

### 添加自定义相机视角

```python
from enhanced_multi_view_extractor import EnhancedMultiViewExtractor, CameraView

# 定义自定义相机视角
custom_views = [
    CameraView(
        name="side_left_camera",
        position=(0.5, 0.8, 0.3),
        orientation=(0, -30, 90),
        resolution=(640, 480)
    ),
    CameraView(
        name="top_down_camera",
        position=(0, 0, 1.2),
        orientation=(0, -90, 0),
        resolution=(640, 480)
    )
]

# 创建增强提取器
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

# 创建视频
if multi_view_data:
    episode_data = multi_view_data[0]
    extractor.create_video(episode_data, "/tmp/video.mp4", "side_left_camera")

extractor.shutdown()
```

### 高级自定义相机配置

```python
from custom_camera_extractor import CustomCameraExtractor, CameraConfig

# 创建复杂的相机配置
camera_configs = [
    CameraConfig(
        name="side_left_camera",
        position=(0.6, 0.8, 0.4),
        orientation=(0, -20, 90),
        resolution=(640, 480),
        fov=60.0,
        near_plane=0.01,
        far_plane=10.0
    ),
    CameraConfig(
        name="close_up_camera",
        position=(0, 0.4, 0.1),
        orientation=(0, 0, 0),
        resolution=(1280, 720)
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
```

## 数据格式

### 目录结构
```
task_name/
├── episode_000/
│   ├── descriptions.json          # 任务描述
│   ├── episode.pkl               # 完整episode数据（pickle格式）
│   ├── episode.json              # 完整episode数据（JSON格式）
│   ├── left_shoulder_camera/
│   │   ├── rgb/                  # RGB图像序列
│   │   │   ├── frame_000000.png
│   │   │   └── ...
│   │   ├── depth/                # 深度图像序列
│   │   │   ├── frame_000000.npy
│   │   │   └── ...
│   │   ├── mask/                 # 掩码图像序列
│   │   │   ├── frame_000000.png
│   │   │   └── ...
│   │   └── camera_params.json    # 相机内外参
│   ├── right_shoulder_camera/
│   ├── overhead_camera/
│   ├── wrist_camera/
│   ├── front_camera/
│   ├── custom_camera_1/          # 自定义相机
│   └── videos/                   # 生成的视频
│       ├── left_shoulder_camera.mp4
│       └── ...
└── episode_001/
    └── ...
```

### 相机参数格式
```json
{
  "intrinsics": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ],
  "extrinsics": [
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0, 0, 0, 1]
  ],
  "near_plane": 0.01,
  "far_plane": 10.0
}
```

## 与PerAct的兼容性

本工具生成的数据格式与PerAct兼容：

1. **图像数据**: RGB和深度图像以相同格式保存
2. **相机参数**: 内外参矩阵格式一致
3. **数据结构**: 支持PerAct的数据加载方式
4. **扩展性**: 支持更多视角，超越PerAct的5个固定视角

## 性能优化建议

1. **图像尺寸**: 根据需求调整图像尺寸，较小尺寸可提高处理速度
2. **无头模式**: 使用`headless=True`可显著提高性能
3. **批量处理**: 一次提取多个episode比单独提取更高效
4. **存储优化**: 深度图像使用`.npy`格式，RGB图像使用`.png`格式

## 常见问题

### Q: 如何添加更多相机视角？
A: 使用`CameraView`或`CameraConfig`类定义新的相机位置和参数，然后添加到提取器中。

### Q: 深度图像的范围是什么？
A: 深度图像值范围是0-1，可以通过`depth_in_meters=False`设置。

### Q: 如何生成视频？
A: 使用`create_video`方法或`_create_episode_videos`方法自动生成视频。

### Q: 支持哪些任务？
A: 支持RLBench中的所有任务，可以通过`task_file_to_task_class`获取任务类。

## 示例任务列表

```python
# 常用任务
common_tasks = [
    "ReachTarget",
    "PickUpCup",
    "PutIntoDrawer", 
    "StackBlocks",
    "OpenDoor",
    "CloseDoor",
    "PutIntoMicrowave",
    "TakeOutOfMicrowave",
    "PutIntoDishwasher",
    "TakeOutOfDishwasher"
]

# 复杂任务
complex_tasks = [
    "RearrangePillars",
    "SetupCheckers",
    "PutGroceriesInCupboard",
    "PutItemInDrawer",
    "SlideBlockToTarget"
]
```

## 扩展功能

### 添加新的相机运动模式
```python
class CustomCameraMotion:
    def __init__(self, camera):
        self.camera = camera
    
    def update_pose(self, step):
        # 实现自定义相机运动
        new_position = calculate_position(step)
        new_orientation = calculate_orientation(step)
        self.camera.set_position(new_position)
        self.camera.set_orientation(new_orientation)
```

### 添加数据增强
```python
def apply_data_augmentation(rgb_image, depth_image):
    # 实现数据增强
    augmented_rgb = augment_rgb(rgb_image)
    augmented_depth = augment_depth(depth_image)
    return augmented_rgb, augmented_depth
```

## 许可证

本工具遵循RLBench的许可证条款。

# RLBench 视频生成工具

本工具包提供了多种方式来生成RLBench任务的视频，包括静态相机视角和动态相机运动。

## 工具概述

### 1. 原始cinematic_recorder.py
- **用途**: 生成电影风格的视频，相机围绕场景做圆形运动
- **特点**: 动态相机运动，适合演示和展示
- **输出**: 高质量MP4视频

### 2. 快速视频生成器 (quick_video_generator.py)
- **用途**: 快速生成多视角视频
- **特点**: 简单易用，支持自定义相机位置
- **输出**: 多个视角的MP4视频文件

### 3. 完整视频生成器 (generate_videos.py)
- **用途**: 高级视频生成，支持多视角合成
- **特点**: 功能完整，支持视频合成
- **输出**: 单视角视频 + 多视角合成视频

## 使用方法

### 方法1: 使用原始cinematic_recorder

```bash
# 生成单个任务的电影风格视频
python tools/cinematic_recorder.py \
    --save_dir /tmp/rlbench_videos/ \
    --tasks ReachTarget \
    --individual True \
    --camera_resolution 1280,720 \
    --headless True

# 生成所有任务的视频
python tools/cinematic_recorder.py \
    --save_dir /tmp/rlbench_videos/ \
    --individual True \
    --camera_resolution 1280,720
```

### 方法2: 使用快速视频生成器

```bash
# 运行快速视频生成器
python examples/quick_video_generator.py
```

这会生成：
- 4个自定义视角的视频
- 5个默认RLBench相机视角的视频
- 总共9个视频文件

### 方法3: 使用完整视频生成器

```bash
# 运行完整视频生成器
python examples/generate_videos.py
```

这会生成：
- 单视角视频
- 多视角合成视频（2x2网格布局）

## 视频类型说明

### 1. 静态相机视频
- **特点**: 相机位置固定
- **用途**: 分析任务执行过程
- **视角**: 侧面、俯视、近距离等

### 2. 动态相机视频
- **特点**: 相机围绕场景运动
- **用途**: 演示和展示
- **运动**: 圆形运动、线性运动等

### 3. 多视角合成视频
- **特点**: 同时显示多个视角
- **用途**: 全面分析任务执行
- **布局**: 2x2网格或其他布局

## 自定义相机配置

### 相机位置设置
```python
# 侧面视角
position=(0.5, 0.8, 0.3)  # (x, y, z)
orientation=(0, -30, 90)  # (roll, pitch, yaw) 度

# 俯视视角
position=(0, 0, 1.2)
orientation=(0, -90, 0)

# 近距离视角
position=(0, 0.3, 0.1)
orientation=(0, 0, 0)
```

### 相机参数设置
```python
# 分辨率
resolution=(640, 480)  # 或 (1280, 720)

# 帧率
fps=30

# 视频编码
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
```

## 输出文件说明

### 文件命名规则
```
{任务名}_{相机名}.mp4
例如: ReachTarget_side_left.mp4
```

### 文件大小
- 640x480分辨率: 约1-5MB per video
- 1280x720分辨率: 约3-15MB per video
- 取决于任务长度和复杂度

### 视频质量
- 格式: MP4 (H.264编码)
- 质量: 高质量RGB图像
- 兼容性: 支持所有主流播放器

## 性能优化建议

### 1. 无头模式运行
```python
env = Environment(
    action_mode=action_mode,
    obs_config=obs_config,
    headless=True  # 重要：提高性能
)
```

### 2. 调整图像分辨率
```python
# 较低分辨率，更快处理
resolution=(320, 240)

# 较高分辨率，更好质量
resolution=(1280, 720)
```

### 3. 批量处理
```python
# 一次处理多个任务
task_names = ["ReachTarget", "PickUpCup", "StackBlocks"]
for task_name in task_names:
    generate_video(task_name)
```

## 常见问题

### Q: 视频文件很大怎么办？
A: 可以降低分辨率或使用更高效的编码格式。

### Q: 如何添加更多相机视角？
A: 在相机配置列表中添加新的位置和朝向。

### Q: 视频播放有问题怎么办？
A: 确保使用支持H.264的播放器，如VLC或QuickTime。

### Q: 如何生成更长的视频？
A: 增加演示数据的数量或使用更复杂的任务。

## 示例任务列表

```python
# 简单任务（适合快速测试）
simple_tasks = [
    "ReachTarget",
    "PickUpCup",
    "PutIntoDrawer"
]

# 复杂任务（生成更长的视频）
complex_tasks = [
    "RearrangePillars",
    "SetupCheckers",
    "PutGroceriesInCupboard"
]
```

## 扩展功能

### 添加相机运动
```python
class CameraMotion:
    def __init__(self, camera):
        self.camera = camera
    
    def update_pose(self, step):
        # 实现自定义运动
        new_position = calculate_position(step)
        self.camera.set_position(new_position)
```

### 添加视频特效
```python
def add_effects(frame):
    # 添加文字标签
    cv2.putText(frame, "RLBench Demo", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame
```

## 总结

RLBench提供了强大的视频生成功能：

1. **cinematic_recorder.py** - 电影风格视频
2. **多视角提取工具** - 静态多视角视频
3. **自定义视频生成器** - 灵活的视频生成

所有工具都支持：
- 高质量视频输出
- 自定义相机配置
- 批量处理
- 无头模式运行

选择适合你需求的工具开始生成视频吧！

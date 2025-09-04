#!/usr/bin/env python3
"""
测试多视角数据提取工具
验证工具是否正常工作
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加RLBench路径
sys.path.append(str(Path(__file__).parent.parent))

def test_quick_extractor():
    """测试快速提取器"""
    print("测试快速多视角提取器...")
    
    try:
        from quick_multi_view_extraction import QuickMultiViewExtractor
        
        # 创建提取器
        extractor = QuickMultiViewExtractor(
            image_size=(128, 128),  # 使用较小尺寸进行快速测试
            headless=True
        )
        
        # 设置环境
        extractor.setup_environment()
        
        # 提取少量数据进行测试
        extractor.extract_task_data(
            task_name="ReachTarget",
            num_episodes=1,
            save_dir="/tmp/test_quick_data"
        )
        
        extractor.shutdown()
        print("✓ 快速提取器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 快速提取器测试失败: {e}")
        return False

def test_custom_camera_extractor():
    """测试自定义相机提取器"""
    print("测试自定义相机提取器...")
    
    try:
        from custom_camera_extractor import CustomCameraExtractor, CameraConfig
        
        # 创建简单的相机配置
        camera_configs = [
            CameraConfig(
                name="test_camera",
                position=(0.5, 0.5, 0.5),
                orientation=(0, -30, 45),
                resolution=(128, 128)
            )
        ]
        
        # 创建提取器
        extractor = CustomCameraExtractor(headless=True)
        extractor.setup_environment()
        extractor.add_custom_cameras(camera_configs)
        
        # 测试相机数据获取
        camera_data = extractor.get_camera_data("test_camera")
        if camera_data is not None:
            print("✓ 自定义相机数据获取成功")
        else:
            print("✗ 自定义相机数据获取失败")
            return False
        
        extractor.shutdown()
        print("✓ 自定义相机提取器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 自定义相机提取器测试失败: {e}")
        return False

def test_enhanced_extractor():
    """测试增强提取器"""
    print("测试增强多视角提取器...")
    
    try:
        from enhanced_multi_view_extractor import EnhancedMultiViewExtractor, CameraView
        
        # 创建简单的自定义视角
        custom_views = [
            CameraView(
                name="test_view",
                position=(0.3, 0.3, 0.3),
                orientation=(0, -20, 30),
                resolution=(128, 128)
            )
        ]
        
        # 创建提取器
        extractor = EnhancedMultiViewExtractor(
            custom_views=custom_views,
            image_size=(128, 128),
            headless=True
        )
        
        # 设置环境
        extractor.setup_environment()
        
        # 测试数据提取（只提取1个episode）
        from rlbench.tasks import ReachTarget
        multi_view_data = extractor.extract_multi_view_data(
            task_class=ReachTarget,
            num_episodes=1
        )
        
        if multi_view_data and len(multi_view_data) > 0:
            print("✓ 增强提取器数据提取成功")
            
            # 测试数据保存
            extractor.save_data(multi_view_data, "/tmp/test_enhanced_data", "ReachTarget")
            print("✓ 增强提取器数据保存成功")
        else:
            print("✗ 增强提取器数据提取失败")
            return False
        
        extractor.shutdown()
        print("✓ 增强多视角提取器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 增强多视角提取器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_format():
    """测试数据格式"""
    print("测试数据格式...")
    
    try:
        # 检查生成的数据文件
        test_dirs = [
            "/tmp/test_quick_data",
            "/tmp/test_enhanced_data"
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                # 检查目录结构
                task_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                if task_dirs:
                    task_dir = os.path.join(test_dir, task_dirs[0])
                    episode_dirs = [d for d in os.listdir(task_dir) if d.startswith("episode_")]
                    
                    if episode_dirs:
                        episode_dir = os.path.join(task_dir, episode_dirs[0])
                        
                        # 检查必要文件
                        required_files = ["episode.json", "descriptions.json"]
                        for file in required_files:
                            if os.path.exists(os.path.join(episode_dir, file)):
                                print(f"✓ 找到文件: {file}")
                            else:
                                print(f"✗ 缺少文件: {file}")
                                return False
                        
                        # 检查相机目录
                        camera_dirs = [d for d in os.listdir(episode_dir) 
                                     if os.path.isdir(os.path.join(episode_dir, d)) and d.endswith("_camera")]
                        
                        if len(camera_dirs) >= 5:  # 至少应该有5个默认相机
                            print(f"✓ 找到 {len(camera_dirs)} 个相机目录")
                        else:
                            print(f"✗ 相机目录数量不足: {len(camera_dirs)}")
                            return False
        
        print("✓ 数据格式测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 数据格式测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试多视角数据提取工具...")
    print("=" * 50)
    
    test_results = []
    
    # 运行测试
    test_results.append(test_quick_extractor())
    print()
    
    test_results.append(test_custom_camera_extractor())
    print()
    
    test_results.append(test_enhanced_extractor())
    print()
    
    test_results.append(test_data_format())
    print()
    
    # 总结结果
    print("=" * 50)
    print("测试结果总结:")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！工具可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    # 清理测试文件
    import shutil
    test_dirs = [
        "/tmp/test_quick_data",
        "/tmp/test_enhanced_data"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"清理测试目录: {test_dir}")
            except Exception as e:
                print(f"清理目录失败: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
将SuperGlue模型导出为TorchScript格式，以便在C++中使用
支持动态批次大小和CPU/CUDA两种版本
"""

import sys
import torch
from pathlib import Path
import argparse

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.superglue import SuperGlue


def export_superglue(weight_type='outdoor', output_path=None, device='cpu', 
                    enable_dynamic_batch=True):
    """
    导出SuperGlue模型为TorchScript格式
    
    Args:
        weight_type: 'indoor' 或 'outdoor'
        output_path: 输出文件路径
        device: 'cpu' 或 'cuda'，决定模型在哪个设备上运行
        enable_dynamic_batch: 是否启用动态批次大小
    """
    # 设置输出路径
    if output_path is None:
        device_suffix = device if device in ['cpu', 'cuda'] else 'cpu'
        output_path = f'superglue_{weight_type}_{device_suffix}.torchscript'
    
    print(f"=" * 60)
    print(f"导出 SuperGlue 模型配置:")
    print(f"  权重类型: {weight_type}")
    print(f"  目标设备: {device}")
    print(f"  动态批次: {enable_dynamic_batch}")
    print(f"  输出路径: {output_path}")
    print(f"=" * 60)
    
    # 创建配置
    config = {
        'weights': weight_type,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    
    # 加载模型
    print(f"\n加载 SuperGlue 模型 ({weight_type} 权重)...")
    model = SuperGlue(config)
    model.eval()
    
    # 移动到指定设备
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"模型已移动到 CUDA 设备")
    else:
        model = model.cpu()
        print(f"模型在 CPU 上运行")
    
    print(f"模型加载成功")
    print(f"配置: {model.config}")
    
    # 创建示例输入用于测试和trace
    print(f"\n创建测试输入数据...")
    
    # 测试不同批次大小
    test_batch_sizes = [1, 2, 4] if enable_dynamic_batch else [1]
    num_keypoints0 = 100
    num_keypoints1 = 120
    descriptor_dim = 256
    
    # 将模型移到正确的设备
    device_type = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    def create_example_inputs(batch_size, device_type):
        """创建示例输入数据"""
        return {
            'keypoints0': torch.randn(batch_size, num_keypoints0, 2, device=device_type),
            'keypoints1': torch.randn(batch_size, num_keypoints1, 2, device=device_type),
            'scores0': torch.randn(batch_size, num_keypoints0, device=device_type),
            'scores1': torch.randn(batch_size, num_keypoints1, device=device_type),
            'descriptors0': torch.randn(batch_size, descriptor_dim, num_keypoints0, device=device_type),
            'descriptors1': torch.randn(batch_size, descriptor_dim, num_keypoints1, device=device_type),
            'image0': torch.randn(batch_size, 1, 480, 640, device=device_type),
            'image1': torch.randn(batch_size, 1, 480, 640, device=device_type),
        }
    
    # 使用batch_size=1的示例输入进行trace
    example_inputs = create_example_inputs(1, device_type)
    
    print(f"\n测试模型前向传播...")
    with torch.no_grad():
        output = model(example_inputs)
    
    print(f"输出键: {output.keys()}")
    print(f"matches0 形状: {output['matches0'].shape}")
    print(f"matches1 形状: {output['matches1'].shape}")
    
    # 导出模型为TorchScript
    print(f"\n导出模型为 TorchScript 格式...")
    
    try:
        # 方法1: 尝试使用 torch.jit.script
        print("尝试使用 torch.jit.script...")
        scripted_model = torch.jit.script(model)
        export_method = "script"
        print("✓ 使用 torch.jit.script 导出成功")
    except Exception as e:
        print(f"✗ torch.jit.script 失败: {e}")
        print("\n尝试使用 torch.jit.trace...")
        
        # 方法2: 使用 torch.jit.trace
        try:
            scripted_model = torch.jit.trace(model, example_inputs, 
                                            strict=False)
            export_method = "trace"
            print("✓ 使用 torch.jit.trace 导出成功")
        except Exception as e2:
            print(f"✗ torch.jit.trace 也失败: {e2}")
            raise RuntimeError("无法导出模型")
    
    # 保存模型
    scripted_model.save(output_path)
    print(f"\n✓ 模型已保存到: {output_path}")
    
    # 验证导出的模型
    print(f"\n验证导出的模型...")
    loaded_model = torch.jit.load(output_path, map_location=device_type)
    loaded_model.eval()
    
    # 测试不同批次大小
    all_tests_passed = True
    for batch_size in test_batch_sizes:
        print(f"\n  测试批次大小 {batch_size}...")
        test_inputs = create_example_inputs(batch_size, device_type)
        
        with torch.no_grad():
            try:
                output_loaded = loaded_model(test_inputs)
                print(f"    ✓ 批次大小 {batch_size} 测试通过")
                print(f"      matches0 形状: {output_loaded['matches0'].shape}")
            except Exception as e:
                print(f"    ✗ 批次大小 {batch_size} 测试失败: {e}")
                all_tests_passed = False
    
    # 额外验证：比较batch_size=1的输出
    print(f"\n数值精度验证...")
    with torch.no_grad():
        output_original = model(example_inputs)
        output_loaded = loaded_model(example_inputs)
    
    matches_diff = (output_original['matches0'] != output_loaded['matches0']).sum().item()
    scores_diff = (output_original['matching_scores0'] - output_loaded['matching_scores0']).abs().max().item()
    
    print(f"  匹配索引差异: {matches_diff}")
    print(f"  最大分数差异: {scores_diff:.6f}")
    
    if matches_diff == 0 and scores_diff < 1e-5:
        print(f"  ✓ 数值精度验证通过")
    else:
        print(f"  ⚠ 警告: 输出存在轻微差异")
    
    # 最终结果
    print(f"\n" + "=" * 60)
    if all_tests_passed:
        print(f"✓ 模型导出成功!")
        print(f"  文件: {output_path}")
        print(f"  方法: {export_method}")
        print(f"  设备: {device}")
        print(f"  动态批次: {'启用' if enable_dynamic_batch else '禁用'}")
    else:
        print(f"⚠ 模型导出完成，但部分测试失败")
    print(f"=" * 60)
    
    return output_path


def export_all_variants():
    """
    导出所有变体：indoor/outdoor x cpu/cuda
    """
    variants = [
        ('indoor', 'cpu'),
        ('indoor', 'cuda'),
        ('outdoor', 'cpu'),
        ('outdoor', 'cuda'),
    ]
    
    print("\n" + "=" * 60)
    print("批量导出所有 SuperGlue 模型变体")
    print("=" * 60)
    
    results = []
    for weight_type, device in variants:
        try:
            output_path = export_superglue(
                weight_type=weight_type,
                device=device,
                enable_dynamic_batch=True
            )
            results.append((weight_type, device, output_path, True))
            print(f"\n✓ {weight_type}-{device} 导出成功\n")
        except Exception as e:
            print(f"\n✗ {weight_type}-{device} 导出失败: {e}\n")
            results.append((weight_type, device, None, False))
    
    # 最终总结
    print("\n" + "=" * 60)
    print("导出总结:")
    print("=" * 60)
    for weight_type, device, path, success in results:
        status = "✓" if success else "✗"
        path_str = path if path else "失败"
        print(f"{status} {weight_type:8s} | {device:4s} | {path_str}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='导出 SuperGlue 模型为 TorchScript 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出 outdoor CPU 版本
  python export_torchscript.py --weights outdoor --device cpu
  
  # 导出 indoor CUDA 版本
  python export_torchscript.py --weights indoor --device cuda
  
  # 导出所有变体
  python export_torchscript.py --all
  
  # 指定输出路径
  python export_torchscript.py --weights outdoor --device cpu --output my_model.torchscript
        """
    )
    
    parser.add_argument('--weights', type=str, default='outdoor',
                       choices=['indoor', 'outdoor'],
                       help='SuperGlue 权重类型 (默认: outdoor)')
    
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='目标设备 (默认: cpu)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认: 自动生成)')
    
    parser.add_argument('--no-dynamic-batch', action='store_true',
                       help='禁用动态批次大小支持')
    
    parser.add_argument('--all', action='store_true',
                       help='导出所有变体 (indoor/outdoor x cpu/cuda)')
    
    args = parser.parse_args()
    
    if args.all:
        # 导出所有变体
        export_all_variants()
    else:
        # 导出单个模型
        export_superglue(
            weight_type=args.weights,
            output_path=args.output,
            device=args.device,
            enable_dynamic_batch=not args.no_dynamic_batch
        )

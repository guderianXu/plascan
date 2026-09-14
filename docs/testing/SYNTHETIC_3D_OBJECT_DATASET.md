# 合成三维物体端到端测试数据

`scripts/validation/generate_synthetic_object_dataset.py` 生成一个闭合的不规则小天体/岩石物体，并从环绕相机独立渲染多视影像。物体具有明显轮廓变化、遮挡、非凸陨坑、连续表面纹理和多高度视角，不调用 PlaScan 的 SfM、MVS、网格或地形实现生成真值。

## 生成数据

生成器使用仓库根目录 `.venv` 中的 NumPy 和 OpenCV：

```powershell
.\.venv\Scripts\python.exe scripts\validation\generate_synthetic_object_dataset.py `
  --output-dir build\tmp\synthetic-object-medium `
  --quality medium
```

输出目录必须不存在或为空。`--quality` 支持以下固定等级，所有等级都使用固定随机种子并可通过显式参数覆盖单项配置：

| 等级 | 视图 | 影像 | 网格（纬线×经线） | 径向真值 | 纹理层 | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `coarse` | 20 | 640×480 | 24×48 | 256×128 | 2 | 快速 smoke、接口和失败路径 |
| `medium` | 28 | 800×608 | 32×64 | 512×256 | 3 | 日常回归与算法比较 |
| `fine` | 40 | 960×736 | 64×128 | 1024×512 | 5 | 发布前精度和性能验收 |

默认等级为 `medium`：

```text
synthetic-object-medium/
├── image_camera.lis
├── manifest.json
├── images/
│   └── view_*.png
├── cameras/
│   └── view_*.tsai
├── mvs_masks/
│   └── view_*_mask.png
└── ground_truth/
    ├── object.ply
    ├── radial_albedo.png
    ├── radial_radius.npy
    ├── radial_elevation.npy
    ├── radial_dom.png
    ├── depth/
    │   ├── view_*.npy
    │   └── view_*_preview.png
    └── masks/
        └── view_*.png
```

`object.ply` 是闭合真值网格；径向图采用经纬展开，`radial_radius.npy` 存储相对物体中心的绝对半径，`radial_elevation.npy` 存储相对参考半径的高程。逐视图深度采用正向 camera-z。`manifest.json` 记录相机内外参、物体尺度、轮廓覆盖率和每个文件的 SHA-256。

`ground_truth/masks` 使用非零表示真值有效区；`mvs_masks` 按 PlaScan 工程蒙版约定使用非零表示排除区，不能互换。

随机种子只影响物体纹理和可复现的传感器噪声，不影响几何或相机。三个等级的噪声标准差依次为 0、0.35 和 0.7 灰度级；当前仍不包含镜头畸变和运动模糊，这两项应作为独立干扰变量，而不是暗含在精细度等级中。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest -v tests.test_generate_synthetic_object_dataset
```

单元测试检查闭合物体真值、三维深度变化、环绕相机、相同种子的逐字节确定性、纹理与几何的随机性隔离、文件校验和以及非空输出目录保护。真实 CLI 全流程、质量门限、退出码和报告格式见 [合成数据全流程测试](SYNTHETIC_E2E_TESTING.md)。

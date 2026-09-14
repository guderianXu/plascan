# 合成地形端到端测试数据

`scripts/validation/generate_synthetic_terrain_dataset.py` 生成米制局部坐标系中的航测地形、多视影像和独立真值。场景包含山脊、沟谷、撞击坑、连续起伏与多尺度微地形，使用航摄相机网格，不复用 PlaScan 的 SfM、MVS、DEM 或 DOM 实现。

## 精细度等级

| 等级 | 视图 | 影像 | DEM | 真值网格 | 地形频段 | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `coarse` | 12 | 512×384 | 256×192 | 96×72 | 2 | 快速 smoke |
| `medium` | 20 | 640×480 | 512×384 | 192×144 | 4 | 日常全流程回归 |
| `fine` | 30 | 960×736 | 1024×768 | 384×288 | 6 | DEM/DOM 精度与性能验收 |

```powershell
.\.venv\Scripts\python.exe scripts\validation\generate_synthetic_terrain_dataset.py `
  --output-dir build\tmp\synthetic-terrain-medium `
  --quality medium
```

输出包括：

```text
synthetic-terrain-medium/
├── image_camera.lis
├── manifest.json
├── images/view_*.png
├── cameras/view_*.tsai
├── mvs_masks/view_*_mask.png
└── ground_truth/
    ├── dem.npy
    ├── dem.tif
    ├── dem.tfw
    ├── dem.prj
    ├── dem_preview.png
    ├── dem_color.png
    ├── dom.png
    ├── dom.tif
    ├── dom.tfw
    ├── dom.prj
    ├── terrain.ply
    ├── depth/view_*.npy
    └── masks/view_*.png
```

DEM 和 DOM 都带米制世界文件与本地坐标系说明。逐视图深度采用正向 camera-z。`manifest.json` 保存仿真等级、地形范围、高程范围、像元大小、相机真值、覆盖率与全部文件的 SHA-256。

`ground_truth/masks` 使用非零表示真值有效区；`mvs_masks` 按 PlaScan 工程蒙版约定使用非零表示排除区，供完整流程直接使用。

## 评价边界

对象场景评价闭合点云/网格的三维距离、完整率和轮廓；地形场景评价 DEM 高程 RMSE、NMAD、空洞率、DOM 平面偏差和有效覆盖率。两类报告不共用“单一总分”，避免闭合物体与 2.5D 地形的几何差异污染结论。

```powershell
.\.venv\Scripts\python.exe -m unittest -v tests.test_generate_synthetic_terrain_dataset
```

真实 CLI 全流程、质量门限、退出码和报告格式见 [合成数据全流程测试](SYNTHETIC_E2E_TESTING.md)。

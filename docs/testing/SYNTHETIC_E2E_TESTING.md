# 合成数据全流程测试

PlaScan 的合成全流程测试使用独立生成的三维真值，不依赖“程序能跑完”作为唯一标准。测试覆盖两类场景：闭合三维物体，以及局部米制地形。两类场景分别使用适合自身几何的指标和门限，不合并成一个不透明总分。

## 推荐执行顺序

先生成一个固定精细度的数据集：

```powershell
.\.venv\Scripts\python.exe scripts\validation\generate_synthetic_object_dataset.py `
  --output-dir build\tmp\synthetic-object-coarse `
  --quality coarse

.\.venv\Scripts\python.exe scripts\validation\generate_synthetic_terrain_dataset.py `
  --output-dir build\tmp\synthetic-terrain-coarse `
  --quality coarse
```

再运行 PlaScan 的真实 CLI 全流程并生成报告：

```powershell
.\.venv\Scripts\python.exe scripts\validation\run_synthetic_e2e.py `
  --dataset-dir build\tmp\synthetic-object-coarse `
  --output-dir build\tmp\synthetic-e2e-object-coarse `
  --build-dir build\windows-source-release `
  --device cpu `
  --mvs-backend auto `
  --point-cloud-backend auto `
  --threads 8

.\.venv\Scripts\python.exe scripts\validation\run_synthetic_e2e.py `
  --dataset-dir build\tmp\synthetic-terrain-coarse `
  --output-dir build\tmp\synthetic-e2e-terrain-coarse `
  --build-dir build\windows-source-release `
  --device cpu `
  --mvs-backend auto `
  --point-cloud-backend auto `
  --threads 8
```

`--device cpu` 控制 SfM/通用处理设备；`--mvs-backend auto` 仍会在当前构建支持时选择 Recovered CUDA MVS。报告中的 `mvs_backend_actual` 和各点云处理阶段的 `actual` 字段记录真实后端。

## 仿真等级与处理等级

仿真精细度和算法处理强度是两个独立概念。数据生成器的 `coarse`、`medium`、`fine` 增加视图、影像、真值网格和纹理细节；全流程运行器同时使用下表的默认处理预算：

| 等级 | SfM quality | MVS quality | 网格分辨率 | 地形 DEM 分辨率 | 指标采样数 |
| --- | ---: | --- | ---: | ---: | ---: |
| `coarse` | 2 | `low` | 64 | 0.18 m | 8,000 |
| `medium` | 2 | `medium` | 100 | 0.10 m | 20,000 |
| `fine` | 3 | `high` | 160 | 0.06 m | 50,000 |

命令行可以覆盖单项处理参数，但数据集的精细度和对应质量门限始终从 `manifest.json` 读取。

## 评价内容

共同检查包括流程状态、全部影像注册、SfM 稀疏点数、平均重投影误差、稠密点云和网格产物存在性。

闭合物体另外检查：

- 稠密点云相对径向真值的 RMSE、P95 和球面覆盖率；
- 重建网格与闭合真值网格的双向 Chamfer-L1 和对称 P95；
- 小天体径向 DEM 的高程 RMSE 和全球有效覆盖率。

地形另外检查：

- 稠密点云对真值 DEM 的高程 RMSE、P95 和网格覆盖率；
- 在实际重建 XY 范围内的双向网格距离，并由覆盖指标单独约束缺失区域；
- PlaScan XYZA GeoTIFF 的 Z 波段 DEM RMSE、NMAD，以及相对完整真值范围的有效面积覆盖率；
- DEM 和 DOM 产物存在性。

门限定义在 `scripts/validation/synthetic_e2e_metrics.py`。修改门限时应以稳定的多次基线为依据，并在评审中说明测量噪声、算法变更或业务要求；不要为了让单次失败变绿而放宽门限。

## 输出与退出码

每次运行保留下列关键文件：

```text
synthetic-e2e-*/
├── run_config.json
├── pipeline.log
├── pipeline/
├── small_body_pipeline.log       # 仅闭合物体
├── small_body_report.json        # 仅闭合物体
├── synthetic_e2e_report.json
└── synthetic_e2e_report.html
```

退出码语义：

- `0`：流程和全部质量门禁通过；
- `1`：流程完成并生成报告，但至少一项质量门禁失败；
- `2`：配置、依赖、外部 CLI、产物解析或流程执行失败。

已有 PlaScan 产物可用 `--evaluate-only --pipeline-report <report.json>` 重新评价；闭合物体还需传入 `--small-body-report <small_body_report.json>`。

## 自动化测试

```powershell
.\.venv\Scripts\python.exe -m unittest -v `
  tests.test_generate_synthetic_object_dataset `
  tests.test_generate_synthetic_terrain_dataset `
  tests.test_synthetic_e2e
```

这些测试验证生成确定性、真值和蒙版约定、三档精细度、非空目录保护、场景专用命令、XYZA DEM 读取、指标计算及 HTML 报告生成。它们不替代真实 CLI 全流程运行。

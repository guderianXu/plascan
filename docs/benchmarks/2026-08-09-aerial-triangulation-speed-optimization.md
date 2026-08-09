# 空中三角测量速度优化验证（2026-08-09）

## 优化范围

- 无相机先验焦距搜索由 18 个固定尺度全扫改为 8 个锚点粗搜，再细化排序前两名的相邻尺度；
  粗搜未形成满足 MVS 质量门的完整模型时仍回退到完整尺度表。
- 焦距候选排序只生成稀疏质量摘要，不再构造逐点、逐观测和逐相机 JSON；正式结果 sidecar 改用
  compact JSON。
- 已知位姿 PnP 只加载一次输入相机，并通过一次全局轨迹倒排构造各影像 2D-3D 对应。
- `CorrespondenceGraph` 维护邻接计数索引，并以只读 `span` 返回对应关系，避免热点查询复制和扫描全部像对。
- OpenCV RANSAC 使用线程局部 RNG 的保存/恢复，不再用全局互斥锁串行化候选求解。
- 匹配缓存采用“原始匹配指纹 + 几何指纹”两级键。完整命中时不准备 TensorRT engine、不创建
  matcher context、不运行 USAC；只改变几何参数时复用原始对应并重跑 USAC。
- `.pimatch` 分片内容未变化时不重写，读取端使用最多 16 个、总 payload 256 MiB、单项 64 MiB 的
  有界 LRU。

## Middlebury dino 16 视角冷/暖实测

环境沿用 Windows Release、CUDA 13.1、RTX 5080、Ryzen 9 9950X3D。输入为：

`testData/photogrammetry_benchmarks/middlebury_dino_sparse_ring/prepared/plascan/image_camera.lis`

命令：

```powershell
build/windows-vcpkg-cuda-release/bin/reconstruct_pipeline_cli.exe `
  testData/photogrammetry_benchmarks/middlebury_dino_sparse_ring/prepared/plascan/image_camera.lis `
  --output-dir build/benchmark_runs/aerial_speed_20260809_dino `
  --device cuda --quality 0 --threads 32 --cuda-parallel-pairs 1 `
  --feature-max-image-dim 1024 --stop-after-sfm --force
```

同一命令连续运行两次：

| 指标 | 冷运行 | 暖运行 | 加速 |
|---|---:|---:|---:|
| CLI 报告的 SFM 耗时 | 7.555 s | 0.943 s | 8.01x |
| 进程墙钟时间 | 11.322 s | 4.416 s | 2.56x |
| 注册影像 | 16/16 | 16/16 | 一致 |
| 稀疏点 | 610 | 610 | 一致 |
| 平均重投影误差 | 0.670289 px | 0.670289 px | 一致 |

暖运行精确复用 77 个像对、2695 个原始匹配；几何阶段复用 77 个像对和 1843 个内点，未准备
matcher context、未运行 USAC。输出位于
`build/benchmark_runs/aerial_speed_20260809_dino`。

## 无先验焦距搜索回退验证

移除同一组 16 视角数据的相机先验后，以已有特征/匹配资产运行高质量自动焦距搜索。该弱输入只注册
6/16 幅影像、生成 358 个稀疏点，未通过 MVS 质量门，因此流程先评估 8 个锚点，再自动补齐其余
10 个尺度：共评估 18 个候选，`exhaustive_fallback=true`，最终选择焦距尺度 5.2。该结果验证了分层搜索
不会在粗搜模型质量不足时牺牲原有完整搜索的安全性；这组弱输入不用于衡量焦距搜索加速比。

## 构建与测试

```powershell
cmake --build build/windows-vcpkg-cuda-release --parallel 12
$env:PROJ_DATA = "build/windows-vcpkg-cuda-release/vcpkg_installed/x64-windows/share/proj"
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -j 8
```

- Windows MSVC、CUDA、Qt Release 全量构建通过。
- CTest 2090 项中 2089 项被执行并全部通过；1 项显式 disabled，另 10 项因缺少可选数据或运行条件跳过。

## 尚存空间

Generic 影像对预选仍依赖本轮内存 SIFT，因此暖运行会先提取特征再确认匹配分片全命中。若要进一步
降低大型工程的恢复时间，需要持久化候选像对计划，或在不依赖 Generic 预选的模式下把缓存探测前移到
`FeatureStage` 之前。

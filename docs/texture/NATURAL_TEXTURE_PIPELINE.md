# 生成 Natural 纹理

当前默认相机纹理流程使用 `plascan_recovered_texture_core` 的 clean-room
Natural CPU 内核处理 winner 掩膜的小连通域，并保留 PlaScan 的相机输入、选图、图集和 PNG
输出边界。其余选图、投影图集和多频带烘焙仍是 PlaScan CPU 工程适配；它不是参考程序的逐像素复刻。
参考 Vulkan 路径需要五个连续 `RGBA32F` atlas band、专用 shader runner、项目 float32 中间 ABI
和固定 TIFF/JPEG 页面编解码。PlaScan GUI/工作流目前不生成或验证这些输入，因此不会把该路径标为
可用，也不会在 Vulkan/ABI/codec 缺失时静默降级为“exact”。

## 使用条件与参数

先完成相机定向、深度图计算及模型生成，再打开“工作流程 → 生成纹理”。只有具备有效针孔相机、
彩色影像、深度、置信度和支持掩膜的视图才能参与。当前相机输入须已去畸变；本操作不重建模型几何。

| 参数 | 默认值 | 实际作用 |
| --- | --- | --- |
| 纹理大小 | 8192 | 单页 PNG 边长；GUI 提供 1024、2048、4096、8192 |
| 影像下采样 | x2 | 源颜色与融合金字塔分辨率；x4 降低内存占用 |
| 抗锯齿 | 1x | 1、2×2 或 4×4 子像素采样，在线性颜色空间累计 |
| 邻视角空洞恢复 | 开 | 严格候选失败后先放宽深度容差再次筛选；仍无有效采样时仅沿网格邻接传播已支持的颜色，并为每个面保留独立 UV，不把所有洞面折叠为同一常量色 |
| 重影过滤 | 开 | 面内光度一致性参与代价，逐纹素以实样本 medoid 剔除颜色离群样本 |
| 局部失焦抑制 | 关 | 以局部 Laplacian 能量图参与候选代价，而非整幅影像清晰度阈值 |
| 重叠区域曝光校正 | 关 | 沿用共同可见三维区域的有界曝光增益估计 |
| 锐化强度 | 1.0 | 烘焙后各 chart 内部的局部反遮罩锐化；0 表示关闭 |

UV 岛保留两像素边距并始终做边缘扩展。无有效相机的面保留顶点色回退；没有顶点色时使用安全常量色。
估算工作内存超过 3 GiB 会报错，应降低图集大小或提高影像下采样倍数。估算包含图集、准备后的影像及
金字塔，不代表进程 RSS 上限；输入工作区、网格和图割也占用内存。

## 实现对应关系

参考来源为用户提供目录中的 `cleanroom/src/candidate_unary.cpp`、`graph_partition.cpp`、
`natural_uv_*.cpp` 和 `texture_pipeline.cpp`。只借鉴算法与参数含义；运行时不读取该目录、不依赖
原程序捕获的 winner、代价数组、PSX 工程、Vulkan shader fixture 或固定场景数据。

| 阶段 | 当前代码 | 替换与保留边界 |
| --- | --- | --- |
| 候选相机 | `TextureVisibilityEvaluator.cpp`、`TextureCandidateCost.cpp`、`plascan_recovered_texture_core` | 保留深度/掩膜/网格遮挡检查；输入校验后由 recovered `build_face_camera_unary` 生成二元代价 |
| 相机标签 | `TextureLabelOptimizer.cpp` | 整数容量 alpha-expansion 取代 ICM 和事后小孤岛合并；仅接受降低 Potts 能量的更新 |
| UV | `TextureChartBuilder.cpp` | 相机标签连通域形成投影岛，按最小面积矩形旋转；保留有预算的 MaxRects/shelf 单页打包器 |
| 多频段 | `TextureSourcePyramids.cpp`、`TextureNaturalBlender.cpp`、`plascan_recovered_texture_core` | 每张相机独立构建五层、逐层四倍降采样金字塔；winner 掩膜先经过 recovered 8 邻域小连通域内核，再按距离权重融合差分频带 |
| 输出 | `TextureAtlasBaker.cpp` | 透视正确反投影与子像素采样，使用显式 support mask 的网格邻接补洞，保留遮挡复核、共享边校正、边缘扩展和 OBJ/MTL/PNG 提交契约 |

`NeighborViewRecovery` 的未映射面补洞以几何顶点场为边界：已映射面分别在三个几何顶点采样线性
颜色，沿顶点邻接的 wavefront 传播到无种子顶点，再把每个未映射三角形的三个顶点颜色烘焙为渐变
tile。该机制保留合法黑色样本，且不改写已映射面；它是 PlaScan CPU 补洞质量修复，不是参考 Vulkan
的 `interpolate_holes_on_mesh` ABI 或完整 exact pipeline 的接入。

与参考代码的差异：局部清晰度使用 OpenCV Laplacian；光度质量使用投影样本的中位数/MAD；
标签图割当前处理全图而非参考实现的空间分区/halo；UV 打包不是其定制 xatlas；金字塔使用
OpenCV 浮点掩膜归一化滤波与距离场，而非 Vulkan 半精度 Jacobi 和捕获 shader 的精确常量。
低频权重为非主视角保留小幅贡献，高频则由 winner 区域主导。因而不承诺原程序标签、UV 排布或像素一致。
大型网格的全图优化性能及真实数据上的接缝质量仍需结合代表性工程评估。

## 旧设置迁移

对话框保存 `textureMappingSettingsRevision=3`。打开并确认旧设置后，固定使用
`pipeline=recovered_natural_v1`、`mappingMode=natural_mapping`、`blendMode=natural`、`padding=2`、
`keepUnmapped=true`。移除混合模式、内部边距及丢弃未映射面的 UI 选项；可支持的图集大小、
下采样、空洞恢复、重影、失焦和曝光开关继续回显。原 16384 图集选项不再提供。

未标版本的隐式锐化默认值 0.35 迁移为 1.0；标记 revision 2 的显式 0.35 保留。
核心 API 的最佳视角和加权平均仍保留；相机纹理入口固定走 Natural 管线，旧相机图集实现不再可由配置、GUI 或 CLI 选择。
CLI 从设置 JSON 读取同一配置；结果中新增实际生效的 `effective_texture_anti_aliasing`。

## 验证入口

在已初始化的原生编译器环境中构建项目：

```powershell
python scripts/env/configure_with_env.py --source-deps --build
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure -R 'Texture|WorkflowParameterDialogStyleTest|BinaryGridMinCutSolver'
```

相关测试覆盖二标签图割的穷举最优能量、禁止标签、取消、非有限质量输入、旋转 UV 坐标、单相机
多频带颜色还原、掩膜边界、主视图细节、重影过滤、逐纹素遮挡与深度复核及对话框设置迁移。
人工验收应选同一工程的细纹理、曝光接缝、失焦影像、遮挡边界分别比较旧版和新版产物；
自动化夹具通过不能代替真实数据的视觉验收。

验证记录必须以当前候选对应的实际 CMake/CTest 和 Dino 表面采样报告为准；过去的“106 项测试”
计数与“未做真实视觉对比”说明不再作为当前实现的验收证据。当前 Windows 构建若 Qt `uic.exe`
缺少其运行时 DLL，会在 CMake 生成阶段明确失败；这不是纹理流程允许忽略的回退。

# 模型影像回归验收实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 GUI/CLI 共用的模型质量验收服务，并用留出相机视角和可选 Metashape 参考点云实际评估 Dino 与无人机九图模型。

**Architecture:** 在 `src/core/qc` 中实现无 GUI 依赖的 CPU z-buffer 渲染、影像指标和参考点云指标；CLI 只负责读取影像-相机列表、选择留出视角和序列化报告。模型读取复用 `TriMesh`，相机投影统一使用 `PositiveDepthCameraModel`，几何最近邻查询复用 PlaPoint。

**Tech Stack:** C++17、Qt6 Core/Gui、OpenCV、PlaPoint、GoogleTest、CLI11、CMake。

## Global Constraints

- 不结束或覆盖当前正在运行且可能含未保存状态的 GUI。
- 当前工作区已有大量用户改动；只增量修改本计划列出的文件，不回滚或格式化无关文件。
- CLI 与未来 GUI 必须调用同一个 `ModelImageQualityEvaluator`，不能维护两套判定逻辑。
- 验收渲染使用 CPU；不把磁盘读取或普通缩放迁移到 CUDA。
- 缺颜色、无有效相机、模型不在相机前方、参考点云坐标不一致都必须显式报告。
- 本计划不创建 Git commit，除非用户后续明确要求。

---

## File Structure

- Modify: `src/core/mesh/MeshTypes.h`、`src/core/mesh/MeshIO.cpp`：增加共享 PLY 网格读取接口。
- Create: `src/core/qc/ModelImageQualityTypes.h`：验收配置、逐视图指标和汇总结果。
- Create: `src/core/qc/ModelMeshRenderer.h/.cpp`：tile-based CPU z-buffer 与顶点颜色插值。
- Create: `src/core/qc/ModelImageMetrics.h/.cpp`：掩膜、覆盖率、IoU、边缘距离、SSIM/PSNR。
- Create: `src/core/qc/ModelGeometryComparator.h/.cpp`：网格连通性和参考点云双向距离。
- Create: `src/core/qc/ModelImageQualityEvaluator.h/.cpp`：统一编排、门控和报告数据。
- Create: `src/cli/cli_model_quality.cpp`：CLI 参数、影像相机加载、留出划分和输出文件。
- Modify: `src/core/qc/CMakeLists.txt`、`src/cli/CMakeLists.txt`、`tests/CMakeLists.txt`：目标注册。
- Create: `src/core/qc/tests/test_model_image_quality.cpp`：核心单元和合成回归测试。
- Modify: `tests/test_cli_contracts.cpp`：CLI 退出码与输出契约。
- Modify: `docs/PROJECT_ARCHITECTURE.md`：记录质量验收模块和 CLI。

---

### Task 1: 共享 PLY 三角网格读取

**Files:**
- Modify: `src/core/mesh/MeshTypes.h`
- Modify: `src/core/mesh/MeshIO.cpp`
- Test: `src/core/qc/tests/test_model_image_quality.cpp`

**Interfaces:**
- Produces: `static bool TriMesh::loadPLY(const std::string &, TriMesh *, std::string *)`。
- Consumes: PlaPoint `readPly<float>()` 返回的顶点、法线、颜色和 faces。

- [ ] **Step 1: 写失败测试**

构造带 3 个顶点、RGB、法线和 1 个面的临时二进制 PLY，调用：

```cpp
TriMesh loaded;
std::string error;
ASSERT_TRUE(TriMesh::loadPLY(path, &loaded, &error)) << error;
ASSERT_EQ(loaded.vertices.size(), 3U);
ASSERT_EQ(loaded.faces.size(), 1U);
EXPECT_EQ(loaded.vertices[1].g, 255);
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_model_image_quality --config Release -j 4
```

Expected: 编译失败，提示 `TriMesh::loadPLY` 尚未定义。

- [ ] **Step 3: 实现最小读取接口**

`loadPLY()` 必须：

```cpp
static bool loadPLY(const std::string &path,
                    TriMesh *mesh,
                    std::string *errorMsg = nullptr);
```

检查空指针、异常、faces 是否为三角形、索引是否越界；缺颜色时保留默认灰色，缺法线时保留零法线。

- [ ] **Step 4: 重新运行测试**

Expected: PLY round-trip 测试通过。

---

### Task 2: CPU z-buffer 模型渲染器

**Files:**
- Create: `src/core/qc/ModelImageQualityTypes.h`
- Create: `src/core/qc/ModelMeshRenderer.h`
- Create: `src/core/qc/ModelMeshRenderer.cpp`
- Modify: `src/core/qc/CMakeLists.txt`
- Test: `src/core/qc/tests/test_model_image_quality.cpp`

**Interfaces:**
- Consumes: `TriMesh`、`PositiveDepthCameraModel`、目标图像尺寸。
- Produces: `ModelRenderResult ModelMeshRenderer::render(...)`，包含 `CV_8UC3` BGR、`CV_8UC1` valid mask、`CV_32FC1` z-buffer 和统计。

- [ ] **Step 1: 写投影和遮挡失败测试**

创建前后两个重叠三角形，前面红色、后面蓝色：

```cpp
const ModelRenderResult render = renderer.render(mesh, camera, cv::Size(128, 128));
ASSERT_TRUE(render.ok);
EXPECT_EQ(render.validMask.at<std::uint8_t>(64, 64), 255);
EXPECT_GT(render.color.at<cv::Vec3b>(64, 64)[2], 200);
EXPECT_LT(render.color.at<cv::Vec3b>(64, 64)[0], 20);
```

- [ ] **Step 2: 运行测试确认失败**

Expected: 头文件或渲染实现不存在。

- [ ] **Step 3: 实现 tile-based 光栅化**

实现流程：

```cpp
for each triangle:
    project vertices with projectWithDepth()
    reject invalid/back-camera/degenerate triangles
    append triangle id to every overlapped 32x32 tile
parallel_for each tile:
    barycentric rasterization
    depth test
    interpolate RGB and depth
```

要求每个 tile 由单个 worker 写入，避免像素级锁；线程数限制在 2–8，并按输出缓冲区内存估算收紧。

- [ ] **Step 4: 增加边界测试**

覆盖相机后方三角形、跨图像边缘三角形、退化三角形、无颜色模型和完全不可见模型。

- [ ] **Step 5: 运行核心测试**

Expected: z-buffer、颜色插值和无效路径全部通过。

---

### Task 3: 影像指标与可视化产物

**Files:**
- Create: `src/core/qc/ModelImageMetrics.h`
- Create: `src/core/qc/ModelImageMetrics.cpp`
- Modify: `src/core/qc/ModelImageQualityTypes.h`
- Test: `src/core/qc/tests/test_model_image_quality.cpp`

**Interfaces:**
- Produces: `ModelViewQuality evaluateModelView(...)`。
- Produces: `cv::Mat buildDinoForegroundMask(...)`、`cv::Mat buildOverlay(...)`、`cv::Mat buildErrorHeatmap(...)`。

- [ ] **Step 1: 写指标失败测试**

使用两个已知矩形掩膜验证 IoU、覆盖率和漂浮面率；使用平移 2 像素的矩形验证边缘 P90；
使用相同彩色区域验证 SSIM 为 1。

- [ ] **Step 2: 运行测试确认失败**

Expected: 指标函数不存在。

- [ ] **Step 3: 实现掩膜和结构指标**

- Dino 掩膜：灰度阈值 + 形态学闭运算 + 最大连通分量 + 孔洞填充。
- 无人机：valid mask 作为覆盖范围；Canny/Scharr 梯度提取稳定结构边缘。
- 边缘距离：双方 distance transform，采样对方边缘并合并为对称分布。

- [ ] **Step 4: 实现外观指标**

只在参考与渲染交集区域计算；先按通道做稳健线性亮度/对比度拟合，再计算 SSIM/PSNR。
交集少于参考区域 20% 时外观指标无效，不能判通过。

- [ ] **Step 5: 验证可视化输出**

测试 overlay 尺寸、热图只在有效区域着色、透明背景不会被黑色当作高相似度。

---

### Task 4: 网格连通性和参考点云 C 验收

**Files:**
- Create: `src/core/qc/ModelGeometryComparator.h`
- Create: `src/core/qc/ModelGeometryComparator.cpp`
- Modify: `src/core/qc/CMakeLists.txt`
- Test: `src/core/qc/tests/test_model_image_quality.cpp`

**Interfaces:**
- Consumes: `TriMesh` 和可选参考 PLY。
- Produces: `ModelGeometryQuality compareGeometry(...)`。

- [ ] **Step 1: 写连通性失败测试**

构造一个大四面体组件和一个小漂浮三角形，验证主要组件占比和漂浮组件包围盒比例。

- [ ] **Step 2: 写双向距离失败测试**

构造两组已知偏移 0.1 的点，验证双向 P50/P95 与覆盖率。

- [ ] **Step 3: 实现 PlaPoint KDTree 比较**

仅在 `--align-reference-cloud` 时调用现有 `PointCloudAlignment`；默认要求坐标直接一致。
点数过大时采用确定性体素抽样，并把抽样步长写入结果。

- [ ] **Step 4: 运行几何测试**

Expected: 连通性、漂浮分量和双向距离测试通过。

---

### Task 5: 统一评估服务和质量门控

**Files:**
- Create: `src/core/qc/ModelImageQualityEvaluator.h`
- Create: `src/core/qc/ModelImageQualityEvaluator.cpp`
- Modify: `src/core/qc/CMakeLists.txt`
- Test: `src/core/qc/tests/test_model_image_quality.cpp`

**Interfaces:**
- Consumes: `ModelImageQualityOptions`。
- Produces: `ModelImageQualityResult evaluate(const ModelImageQualityOptions &) const`。

- [ ] **Step 1: 写 Dino 和无人机判定失败测试**

使用人工指标对象验证：任一硬性几何门控失败时总结果失败；缺颜色时 SSIM 不可用并失败；
所有阈值满足时通过。

- [ ] **Step 2: 实现编排**

加载模型一次，逐视图加载原图、缩放相机、渲染、计算指标并写 PNG；汇总使用中位数，
最终判定严格使用设计文档阈值。

- [ ] **Step 3: 输出 JSON/CSV/contact sheet**

JSON 包含输入哈希、训练/留出列表、逐视图结果、汇总、阈值和失败原因；contact sheet 每行一个视图，
列顺序固定为 source/render/overlay/error。

- [ ] **Step 4: 运行服务测试**

Expected: 合成模型生成完整报告和四联图，失败模型返回 `ok=false` 但保留诊断产物。

---

### Task 6: model_quality_cli

**Files:**
- Create: `src/cli/cli_model_quality.cpp`
- Modify: `src/cli/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Modify: `tests/test_cli_contracts.cpp`

**Interfaces:**
- Consumes: `--mesh`、`--image-camera-list`、`--validation-split`、可选 `--reference-cloud`。
- Produces: 输出目录和退出码 0/2/3。

- [ ] **Step 1: 写 CLI 契约失败测试**

验证 `--help` 包含全部参数；无 mesh 返回 2；质量失败返回 3 且 JSON 仍存在。

- [ ] **Step 2: 实现列表和留出划分**

复用 `readPhotogrammetryImageList()` 加载图像和 TSAI；Dino `angular-20` 按相机方位角每五张留一张；
无人机 `grid-center-edge` 按相机中心 PCA 主平面分格。

- [ ] **Step 3: 实现 CLI 调用和退出码**

CLI 不复制指标逻辑，只构造 `ModelImageQualityOptions` 并调用 evaluator。

- [ ] **Step 4: 编译与契约测试**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target model_quality_cli test_model_image_quality test_cli_contracts --config Release -j 4
E:\code\plascan\build\windows-vcpkg-cuda-release\src\core\qc\Release\test_model_image_quality.exe
```

Expected: 新目标编译成功，核心测试全部通过，CLI 契约测试通过。

---

### Task 7: Dino 留出视角真实验收

**Files:**
- Runtime output only: `E:/code/test/dino/model_quality_20260713/`

**Interfaces:**
- Consumes: 当前 Dino 模型、Middlebury image-camera list、留出视角。
- Produces: Dino JSON/CSV/contact sheet 和明确通过/失败结论。

- [x] **Step 1: 对当前模型运行 A 验收**

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\model_quality_cli.exe `
  --mesh E:\code\test\dino\mvs_output\products\model_from_mesh.ply `
  --mvs-workspace E:\code\test\dino\mvs_output `
  --validation-split angular-20 `
  --output-dir E:\code\test\dino\model_quality_20260713 `
  --max-render-dim 1600
```

Expected: 无论达标与否都生成完整诊断；退出 3 表示当前模型失败，不视为工具失败。

- [x] **Step 2: 人工检查 contact sheet**

使用 `view_image` 检查所有留出视角，确认指标未被背景、裁剪或错误相机欺骗。

- [x] **Step 3: 定位首个失败阶段**

若轮廓错位，优先检查空三/相机；轮廓正确但内部破碎，检查深度一致性和网格；几何正确但颜色错位，检查 TextureMapper。

---

### Task 8: 无人机九图 A+C 真实验收

**Files:**
- Runtime output only: `E:/code/test/small_test/model_quality_20260713/`

**Interfaces:**
- Consumes: 九图模型、`agisoft_aerial_gcps_small` image-camera list、可选 Metashape 参考点云。
- Produces: 无人机 JSON/CSV/contact sheet 和几何比较报告。

- [x] **Step 1: 对九图模型运行 A 验收**

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\model_quality_cli.exe `
  --mesh E:\code\test\small_test\model_valid_normal_fix\products\model_from_mesh.ply `
  --mvs-workspace E:\code\test\small_test\mvs_output `
  --validation-split grid-center-edge `
  --output-dir E:\code\test\small_test\model_quality_20260713 `
  --max-render-dim 1600
```

- [x] **Step 2: 查找并验证 Metashape 参考点云坐标**

只有确认九图模型和参考点云处于同一局部坐标系，或显式启用对齐后，才追加 `--reference-cloud`。

- [x] **Step 3: 运行 C 验收并检查接触表**

通过 `--reference-camera-list` 使用同名 MVS/TSAI 相机中心估计完整 Sim3，随后把 Metashape
参考点云裁剪到九图模型局部范围再计算双向距离。2026-07-13 实测 P50=0.245 m、
P84=0.721 m、P95=1.154 m，模型/参考覆盖率分别为 17.7%/6.8%，未达标。

记录双向 P50/P84/P95、覆盖率和可视化；道路/田块边缘错位超过 3 像素时判失败。

- [x] **Step 4: 决定是否跑 444 张全量**

结论：九图硬门控失败，不启动 444 张全量。先修复 MVS 帧覆盖波动和网格分裂。

只有九图 A 的硬门控全部通过，且 C 没有显示系统性高程起伏或尺度错误时，才启动全量重建。

---

### Task 9: 文档与最终验证

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/qc/CMakeLists.txt`
- Modify: `src/cli/CMakeLists.txt`

- [x] **Step 1: 更新架构文档**

记录 `ModelImageQualityEvaluator`、`model_quality_cli`、A+C 指标和输出文件。

- [x] **Step 2: 运行相关全量测试**

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release `
  --target qc meshing model_quality_cli test_model_image_quality test_cli_contracts `
  --config Release -j 4
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release `
  -C Release --output-on-failure -R "ModelImageQuality|Mesh|Cli"
```

Expected: 新增测试全部通过；任何历史失败必须单独列出，不能描述为全量通过。

- [x] **Step 3: 核对工作区**

```powershell
git diff --check
git status --short
```

确认未覆盖用户现有改动，测试输出仅写入指定 `E:/code/test/.../model_quality_20260713` 目录。

# PlaScan 点云模块替换为 plapoint 库 — 设计规格

日期: 2026-05-08 | 状态: 已确认

## 背景

plapoint (github.com/guderianXu/plapoint) 是为 PlaScan 专门编写的 GPU 加速点云 C++17 库。它基于 plamatrix 矩阵后端，提供 PCL 风格的算法类模板（`<Scalar, Device::CPU/GPU>`）。

当前 PlaScan 有自己的点云实现（`src/core/pointcloud/`）、空间索引（`src/common/spatial/`）、及 mesh 模块中重复的 MarchingCubes/Poisson 重建。本次替换消除这些重复，统一到 plapoint。

## 目标

- 用 plapoint 替换 PlaScan 中所有点云相关代码，不做桥接层
- plapoint 通过可选属性扩展支持 PlaScan 的全部数据需求（颜色、纹理坐标、面片、OBJ）
- TDD 驱动：先写测试，再写实现
- 两个 repo（plapoint + plascan）并行推进

## 设计决策

### 决策 1: 可选属性扩展 plapoint PointCloud

plapoint 的 `PointCloud<Scalar, Dev>` 保持 Nx3 矩阵为唯一必选项。额外属性通过 `unique_ptr` 持有，不存在即为 null，不影响库的通用性：

```
PointCloud<Scalar, Dev>
├── _points (Nx3 DenseMatrix)          ← 已有，唯一必选项
├── _normals (Nx3 DenseMatrix)         ← 已有
├── _colors (Nx3 DenseMatrix, uint8)   ← 新增
├── _textureCoords (Nx2 DenseMatrix)   ← 新增
├── _faces (Fx3, vertices)             ← 新增
├── _faceTextureIndices (Fx3)          ← 新增
├── _materialLibraryFile (string)      ← 新增
└── _textureImageFile (string)         ← 新增
```

每个新增属性配套 `has*()` / `*()` / `set*()` 三件套，模式与已有 normals 一致。

### 决策 2: 点视图包装

给 plapoint 添加轻量零拷贝视图 `PointCloud::operator[](size_t i)`：

```cpp
auto pt = cloud[i];
pt.x();  pt.y();  pt.z();             // 始终可用
pt.r();  pt.g();  pt.b();             // hasColors() 时断言
pt.nx(); pt.ny(); pt.nz();            // hasNormals() 时断言
pt.u();  pt.v();                      // hasTextureCoords() 时断言
```

底层只是矩阵行索引，零拷贝。

### 决策 3: 摄影测量属性聚合结构（方案 B）

`PhotogrammetryPointAttributes`（pointId, trackLength, reprojectionError, confidence, isControlPoint）不进入 plapoint。在 PlaScan 中用轻量聚合结构：

```cpp
// 位置: src/core/sfm/common/SparsePointCloud.h
namespace xjw::sfm {

struct SparsePointCloud {
    plapoint::PointCloud<float, plapoint::CPU> geometry;
    std::vector<PhotogrammetryPointAttributes> attributes;

    size_t size() const;       // 断言 geometry.size() == attributes.size()
    bool empty() const;
    void clear();
};

} // namespace xjw::sfm
```

保证两端索引一致即可，不加额外抽象。

### 决策 4: OBJ/MTL I/O 放入 plapoint

plapoint 新增 `plapoint/io/obj_io.h`，支持：
- 读取 .obj（顶点、纹理坐标、法向量、面片）和 .mtl（材质名、纹理图路径）
- 写入 .obj（含面片和纹理引用）

### 决策 5: 删除而非废弃

`src/common/spatial/`（KDTree/KDTree2D/KDTree3D）直接删除，不保留兼容层。

---

## 变更清单

### plapoint 侧（新功能）

| 变更 | 文件 | 说明 |
|------|------|------|
| 可选属性 | `include/plapoint/core/point_cloud.h` | 添加 colors/textureCoords/faces/faceTextureIndices/material/texture 成员 |
| 点视图 | `include/plapoint/core/point_cloud.h` | 添加 `PointView` 内部类和 `operator[]` |
| OBJ I/O | `include/plapoint/io/obj_io.h` + `src/obj_io.cpp` | 读写 OBJ/MTL |
| 模板实例化 | `src/plapoint.cpp` | 新增属性和 OBJ I/O 的显式实例化 |
| 测试 | `test/unit/core/` + `test/unit/io/` | PointCloud 可选属性测试 + OBJ 读写测试 |

### PlaScan 侧 — 删除

| 删除 | 说明 |
|------|------|
| `src/core/pointcloud/` (全部) | 16 文件：data/PointCloud, PointCloudPoint, io/PointCloudIO, ObjMtlLoader, processing/PointCloudProcessor* |
| `src/common/spatial/` (全部) | KDTree.h, KDTree2D.h, KDTree3D.h + tests |
| `src/core/mesh/MarchingCubesTable.*` | plapoint::MarchingCubes 替代 |
| `src/core/mesh/poisson/` (全部) | plapoint::PoissonReconstruction 替代 |
| `src/core/mesh/SurfaceReconstructorIO.*` | 自有 PointXYZRGB 类型和解析器，plapoint I/O 替代 |

### PlaScan 侧 — 新增

| 新增 | 说明 |
|------|------|
| `src/core/sfm/common/SparsePointCloud.h` | 聚合 plapoint::PointCloud + vector<PhotogrammetryPointAttributes> |

### PlaScan 侧 — 修改

| 模块 | 文件 | 改动 |
|------|------|------|
| terrain | DemGenerator, DemDomIO, TerrainPipeline, AsteroidProjection | PointCloud → plapoint, 矩阵/视图访问 |
| mesh | TextureMapper | PointCloud → plapoint |
| mesh | SurfaceReconstructor | 拆除 MC/Poisson 子模块调用，改用 plapoint |
| mesh | ModelWorkflowService | PointCloudQualityReport 改用 plapoint |
| mvs | StereoDenseCloudPipelineOutput | PLY 写入改用 plapoint I/O |
| pipeline | SFMService | PointCloud + PLY 写入改用 plapoint |
| gui | CameraModel3DDialog | 成员 m_cloud 改为 plapoint::PointCloud，undo 栈适配 |
| sfm/filtering | SparsePointCloudProcessor | 使用新的 SparsePointCloud 聚合类型 |
| CMake | 所有 CMakeLists.txt | `pointcloud` → `plapoint::plapoint`，移除 common_spatial |
| tests | test_sfm_filter, test_sparse_point_cloud_processor, test_initial_sparse_triangulator, test_sfm_params | SparsePointCloudPoint → SparsePointCloud 聚合类型 |

### 不受影响的模块

- MVS 内部类型（DensePoint, SparseCloud, cv::Mat 网格）— 不经过 PointCloud
- 特征提取/匹配/BA/深度图估计 — 不涉及点云数据结构
- CLI 工具 — 直接使用 plapoint

---

## 执行顺序

按 TDD 铁律推进，两个 repo 可并行：

### Phase 1: plapoint 扩展（plapoint repo）
1. PointCloud 可选属性（colors, textureCoords, faces）— 测试 → 实现
2. PointView 视图包装 — 测试 → 实现
3. OBJ/MTL I/O — 测试 → 实现
4. 更新显式模板实例化

### Phase 2: PlaScan 删除（plascan repo）
5. 用 plapoint KdTree 替换 common/spatial 引用 → 删除 common/spatial
6. 用 plapoint::PointCloud 替换 xjw::pointcloud::PointCloud 引用 → 删除 pointcloud 模块
7. 用 plapoint MC/Poisson 替换 mesh 模块重复代码 → 删除 mesh 子模块

### Phase 3: PlaScan 集成（plascan repo）
8. 新增 SparsePointCloud 聚合类型
9. 适配所有消费者（terrain, mesh, mvs, pipeline, gui, sfm/filtering）
10. 更新 CMakeLists.txt 依赖
11. 全量编译 + 全量测试通过

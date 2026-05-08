# PointCloud Replacement with plapoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PlaScan's point cloud module (`src/core/pointcloud/`), spatial index (`src/common/spatial/`), and mesh duplicate code (MarchingCubes/Poisson) with the plapoint library, extending plapoint with optional attributes and OBJ I/O along the way.

**Architecture:** plapoint gets extended with optional PointCloud attributes (colors, textureCoords, faces, material refs) and a PointView accessor. PlaScan consumers switch from `xjw::pointcloud::PointCloud` (SoA vectors) to `plapoint::PointCloud<Scalar, CPU>` (Nx3 matrix-backed), with photogrammetry-specific attributes bundled in a lightweight `SparsePointCloud` aggregator. All common/spatial KD-Tree usage switches to `plapoint::search::KdTree`. Mesh MC/Poisson duplication is deleted, consumers use plapoint implementations directly.

**Tech Stack:** C++17, plapoint (depends on plamatrix), Google Test, CUDA (optional)

---

## File Structure

### plapoint repo — create/modify

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `include/plapoint/core/point_cloud.h` | Add optional colors/textureCoords/faces/material refs + PointView inner class |
| Create | `include/plapoint/io/obj_io.h` | `readObj<Scalar>(path)` and `writeObj<Scalar>(path, cloud)` declarations |
| Create | `src/obj_io.cpp` | OBJ/MTL reader/writer implementation |
| Create | `test/unit/core/point_cloud_attributes_test.cpp` | Tests for optional attributes and PointView |
| Create | `test/unit/io/obj_io_test.cpp` | Tests for OBJ read/write roundtrip |

### plascan repo — create/modify/delete

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/core/sfm/common/SparsePointCloud.h` | Aggregates `plapoint::PointCloud<float, CPU>` + `std::vector<PhotogrammetryPointAttributes>` |
| Delete | `src/core/pointcloud/` (all) | Replaced by plapoint |
| Delete | `src/common/spatial/` (all) | Replaced by plapoint::KdTree |
| Delete | `src/core/mesh/MarchingCubesTable.h/cpp` | Replaced by plapoint::MarchingCubes |
| Delete | `src/core/mesh/poisson/` (all) | Replaced by plapoint::PoissonReconstruction |
| Delete | `src/core/mesh/SurfaceReconstructorIO.h/cpp` | Its PointXYZRGB type replaced by plapoint::PointCloud |
| Modify | `src/core/terrain/DemGenerator.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/terrain/DemDomIO.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/terrain/TerrainPipeline.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/terrain/projection/AsteroidProjection.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/terrain/CMakeLists.txt` | pointcloud dep → plapoint |
| Modify | `src/core/mesh/TextureMapper.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/mesh/ModelWorkflowService.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/mesh/SurfaceReconstructor.h/cpp` | Replace MC/Poisson calls with plapoint |
| Modify | `src/core/mesh/CMakeLists.txt` | pointcloud dep → plapoint |
| Modify | `src/core/mvs/StereoDenseCloudPipelineOutput.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/mvs/CMakeLists.txt` | pointcloud dep → plapoint |
| Modify | `src/core/pipeline/SFMService.h/cpp` | PointCloud type → plapoint |
| Modify | `src/core/pipeline/CMakeLists.txt` | pointcloud dep → plapoint |
| Modify | `src/core/sfm/filtering/SparsePointCloudProcessor.h/cpp` | SparsePointCloudPoint → SparsePointCloud |
| Modify | `src/core/sfm/filtering/SfmPointCloudFilter.h/cpp` | PointCloud references → SparsePointCloud |
| Modify | `src/core/sfm/CMakeLists.txt` | pointcloud dep → plapoint |
| Modify | `src/gui/dialogs/CameraModel3DDialog.h/cpp` | m_cloud member → plapoint |
| Modify | `src/gui/CMakeLists.txt` | pointcloud dep → plapoint |
| Modify | `tests/CMakeLists.txt` + top-level SFM tests | PointCloud → plapoint/SparsePointCloud |
| Modify | Root `CMakeLists.txt` | Add `find_package(plapoint REQUIRED)` |

---

### Task 1: PointCloud optional colors attribute (plapoint repo)

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`
- Create: `test/unit/core/point_cloud_attributes_test.cpp`

- [ ] **Step 1: Write the failing test**

In `test/unit/core/point_cloud_attributes_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointCloudAttributesTest, NoColorsByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasColors());
    EXPECT_EQ(cloud.colors(), nullptr);
}

TEST(PointCloudAttributesTest, SetColorsCopy)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(10, 3);
    colors.fill(128);

    cloud.setColors(colors);

    ASSERT_TRUE(cloud.hasColors());
    EXPECT_EQ(cloud.colors()->getValue(0, 0), 128);
    EXPECT_EQ(cloud.colors()->getValue(0, 1), 128);
    EXPECT_EQ(cloud.colors()->getValue(0, 2), 128);
}

TEST(PointCloudAttributesTest, SetColorsMove)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(10, 3);
    colors.setValue(0, 0, 255);

    cloud.setColors(std::move(colors));

    ASSERT_TRUE(cloud.hasColors());
    EXPECT_EQ(cloud.colors()->getValue(0, 0), 255);
}

TEST(PointCloudAttributesTest, SetColorsRejectsWrongSize)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(5, 3);
    EXPECT_THROW(cloud.setColors(colors), std::runtime_error);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/guderian/code/plapoint/build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*"
```

Expected: FAIL — `hasColors`/`setColors`/`colors()` not defined on PointCloud.

- [ ] **Step 3: Implement colors attribute in point_cloud.h**

Add `#include <cstdint>` at the top, then add the following members (following the normals pattern):

In the public section, after normals methods:

```cpp
    /// Set optional RGB colors by copy (Nx3 uint8 matrix)
    void setColors(const plamatrix::DenseMatrix<uint8_t, Dev>& c)
    {
        if (c.rows() != _points.rows() || c.cols() != 3)
            throw std::runtime_error("Colors must match point count and be Nx3");
        _colors = std::make_unique<plamatrix::DenseMatrix<uint8_t, Dev>>(c.rows(), c.cols());
        for (plamatrix::Index r = 0; r < c.rows(); ++r)
            for (int col = 0; col < 3; ++col)
                _colors->setValue(r, col, pointGet(c, r, col));
    }

    void setColors(plamatrix::DenseMatrix<uint8_t, Dev>&& c)
    {
        if (c.rows() != _points.rows() || c.cols() != 3)
            throw std::runtime_error("Colors must match point count and be Nx3");
        _colors = std::make_unique<plamatrix::DenseMatrix<uint8_t, Dev>>(std::move(c));
    }

    bool hasColors() const { return _colors != nullptr; }

    const plamatrix::DenseMatrix<uint8_t, Dev>* colors() const { return _colors.get(); }

    plamatrix::DenseMatrix<uint8_t, Dev>* colors() { return _colors.get(); }
```

In the private section, after `_normals`:

```cpp
    std::unique_ptr<plamatrix::DenseMatrix<uint8_t, Dev>> _colors;
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*"
```

Expected: PASS (4/4).

- [ ] **Step 5: Commit**

```bash
cd /home/guderian/code/plapoint
git add include/plapoint/core/point_cloud.h test/unit/core/point_cloud_attributes_test.cpp
git commit -m "feat: add optional RGB colors to PointCloud"
```

---

### Task 2: PointCloud optional texture coordinates (plapoint repo)

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`
- Modify: `test/unit/core/point_cloud_attributes_test.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/unit/core/point_cloud_attributes_test.cpp`:

```cpp
TEST(PointCloudAttributesTest, NoTextureCoordsByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasTextureCoords());
    EXPECT_EQ(cloud.textureCoords(), nullptr);
}

TEST(PointCloudAttributesTest, SetTextureCoords)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(10, 2);
    tex.setValue(0, 0, 0.5f);
    tex.setValue(0, 1, 0.75f);

    cloud.setTextureCoords(std::move(tex));

    ASSERT_TRUE(cloud.hasTextureCoords());
    EXPECT_FLOAT_EQ(cloud.textureCoords()->getValue(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(cloud.textureCoords()->getValue(0, 1), 0.75f);
}

TEST(PointCloudAttributesTest, SetTextureCoordsRejectsWrongSize)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(5, 2);
    EXPECT_THROW(cloud.setTextureCoords(tex), std::runtime_error);
}
```

- [ ] **Step 2: Verify test fails**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*Texture*"
```

- [ ] **Step 3: Implement texture coords**

In `point_cloud.h` public section, after colors methods:

```cpp
    void setTextureCoords(const MatrixType& t)
    {
        if (t.rows() != _points.rows() || t.cols() != 2)
            throw std::runtime_error("Texture coords must match point count and be Nx2");
        _textureCoords = std::make_unique<MatrixType>(t.rows(), t.cols());
        for (plamatrix::Index r = 0; r < t.rows(); ++r)
            for (int col = 0; col < 2; ++col)
                _textureCoords->setValue(r, col, pointGet(t, r, col));
    }

    void setTextureCoords(MatrixType&& t)
    {
        if (t.rows() != _points.rows() || t.cols() != 2)
            throw std::runtime_error("Texture coords must match point count and be Nx2");
        _textureCoords = std::make_unique<MatrixType>(std::move(t));
    }

    bool hasTextureCoords() const { return _textureCoords != nullptr; }

    const MatrixType* textureCoords() const { return _textureCoords.get(); }

    MatrixType* textureCoords() { return _textureCoords.get(); }
```

In private section, after `_colors`:

```cpp
    std::unique_ptr<MatrixType> _textureCoords;
```

Note: texture coords use `Scalar` (float/double), same MatrixType as points.

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*"
```

- [ ] **Step 5: Commit**

```bash
cd /home/guderian/code/plapoint
git add include/plapoint/core/point_cloud.h test/unit/core/point_cloud_attributes_test.cpp
git commit -m "feat: add optional texture coordinates to PointCloud"
```

---

### Task 3: PointCloud optional face attributes (plapoint repo)

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`
- Modify: `test/unit/core/point_cloud_attributes_test.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/unit/core/point_cloud_attributes_test.cpp`:

```cpp
TEST(PointCloudAttributesTest, NoFacesByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasFaces());
    EXPECT_EQ(cloud.faces(), nullptr);
}

TEST(PointCloudAttributesTest, SetFaces)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces.setValue(0, 0, 0); faces.setValue(0, 1, 1); faces.setValue(0, 2, 2);
    faces.setValue(1, 0, 3); faces.setValue(1, 1, 4); faces.setValue(1, 2, 5);

    cloud.setFaces(std::move(faces));

    ASSERT_TRUE(cloud.hasFaces());
    EXPECT_EQ(cloud.faces()->rows(), 2);
    EXPECT_EQ(cloud.faces()->cols(), 3);
    EXPECT_EQ(cloud.faces()->getValue(0, 0), 0);
    EXPECT_EQ(cloud.faces()->getValue(1, 2), 5);
}

TEST(PointCloudAttributesTest, SetFacesWithTextureIndices)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0); faces.setValue(0, 1, 1); faces.setValue(0, 2, 2);
    cloud.setFaces(faces);

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> texFaces(1, 3);
    texFaces.setValue(0, 0, 5); texFaces.setValue(0, 1, 6); texFaces.setValue(0, 2, 7);
    cloud.setFaceTextureIndices(std::move(texFaces));

    ASSERT_TRUE(cloud.hasFaceTextureIndices());
    EXPECT_EQ(cloud.faceTextureIndices()->getValue(0, 0), 5);
}

TEST(PointCloudAttributesTest, SetFacesRejectsNonNx3)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 2);
    EXPECT_THROW(cloud.setFaces(faces), std::runtime_error);
}
```

- [ ] **Step 2: Verify test fails**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*Face*"
```

- [ ] **Step 3: Implement faces**

In `point_cloud.h` public section:

```cpp
    void setFaces(const plamatrix::DenseMatrix<int, Dev>& f)
    {
        if (f.cols() != 3)
            throw std::runtime_error("Faces must be Fx3");
        _faces = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(f.rows(), f.cols());
        for (plamatrix::Index r = 0; r < f.rows(); ++r)
            for (int col = 0; col < 3; ++col)
                _faces->setValue(r, col, pointGet(f, r, col));
    }

    void setFaces(plamatrix::DenseMatrix<int, Dev>&& f)
    {
        if (f.cols() != 3)
            throw std::runtime_error("Faces must be Fx3");
        _faces = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(std::move(f));
    }

    bool hasFaces() const { return _faces != nullptr; }

    const plamatrix::DenseMatrix<int, Dev>* faces() const { return _faces.get(); }
    plamatrix::DenseMatrix<int, Dev>* faces() { return _faces.get(); }

    void setFaceTextureIndices(const plamatrix::DenseMatrix<int, Dev>& ft)
    {
        if (ft.cols() != 3)
            throw std::runtime_error("Face texture indices must be Fx3");
        _faceTextureIndices = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(ft.rows(), ft.cols());
        for (plamatrix::Index r = 0; r < ft.rows(); ++r)
            for (int col = 0; col < 3; ++col)
                _faceTextureIndices->setValue(r, col, pointGet(ft, r, col));
    }

    void setFaceTextureIndices(plamatrix::DenseMatrix<int, Dev>&& ft)
    {
        if (ft.cols() != 3)
            throw std::runtime_error("Face texture indices must be Fx3");
        _faceTextureIndices = std::make_unique<plamatrix::DenseMatrix<int, Dev>>(std::move(ft));
    }

    bool hasFaceTextureIndices() const { return _faceTextureIndices != nullptr; }

    const plamatrix::DenseMatrix<int, Dev>* faceTextureIndices() const { return _faceTextureIndices.get(); }
    plamatrix::DenseMatrix<int, Dev>* faceTextureIndices() { return _faceTextureIndices.get(); }
```

In private section:

```cpp
    std::unique_ptr<plamatrix::DenseMatrix<int, Dev>> _faces;
    std::unique_ptr<plamatrix::DenseMatrix<int, Dev>> _faceTextureIndices;
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*"
```

- [ ] **Step 5: Commit**

```bash
cd /home/guderian/code/plapoint
git add include/plapoint/core/point_cloud.h test/unit/core/point_cloud_attributes_test.cpp
git commit -m "feat: add optional faces and face texture indices to PointCloud"
```

---

### Task 4: PointCloud material/texture file references (plapoint repo)

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`
- Modify: `test/unit/core/point_cloud_attributes_test.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/unit/core/point_cloud_attributes_test.cpp`:

```cpp
TEST(PointCloudAttributesTest, MaterialLibraryFileEmptyByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_TRUE(cloud.materialLibraryFile().empty());
}

TEST(PointCloudAttributesTest, SetMaterialLibraryFile)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    cloud.setMaterialLibraryFile("materials.mtl");
    EXPECT_EQ(cloud.materialLibraryFile(), "materials.mtl");
}

TEST(PointCloudAttributesTest, TextureImageFileEmptyByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_TRUE(cloud.textureImageFile().empty());
}

TEST(PointCloudAttributesTest, SetTextureImageFile)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    cloud.setTextureImageFile("diffuse.png");
    EXPECT_EQ(cloud.textureImageFile(), "diffuse.png");
}
```

- [ ] **Step 2: Verify test fails**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*Material*:PointCloudAttributesTest*TextureImage*"
```

- [ ] **Step 3: Implement string attributes**

In `point_cloud.h`, add `#include <string>`. In public section:

```cpp
    const std::string& materialLibraryFile() const { return _materialLibraryFile; }
    void setMaterialLibraryFile(const std::string& f) { _materialLibraryFile = f; }

    const std::string& textureImageFile() const { return _textureImageFile; }
    void setTextureImageFile(const std::string& f) { _textureImageFile = f; }
```

In private section:

```cpp
    std::string _materialLibraryFile;
    std::string _textureImageFile;
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointCloudAttributesTest*"
```

- [ ] **Step 5: Commit**

```bash
cd /home/guderian/code/plapoint
git add include/plapoint/core/point_cloud.h test/unit/core/point_cloud_attributes_test.cpp
git commit -m "feat: add material/texture file references to PointCloud"
```

---

### Task 5: PointView accessor (plapoint repo)

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`
- Create: `test/unit/core/point_view_test.cpp`

- [ ] **Step 1: Write the failing test**

In `test/unit/core/point_view_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointViewTest, AccessXYZ)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    cloud.points().setValue(0, 0, 1.0f);
    cloud.points().setValue(0, 1, 2.0f);
    cloud.points().setValue(0, 2, 3.0f);

    auto pt = cloud[0];
    EXPECT_FLOAT_EQ(pt.x(), 1.0f);
    EXPECT_FLOAT_EQ(pt.y(), 2.0f);
    EXPECT_FLOAT_EQ(pt.z(), 3.0f);
}

TEST(PointViewTest, AccessColors)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(3, 3);
    colors.setValue(0, 0, 10); colors.setValue(0, 1, 20); colors.setValue(0, 2, 30);
    cloud.setColors(std::move(colors));

    auto pt = cloud[0];
    EXPECT_EQ(pt.r(), 10);
    EXPECT_EQ(pt.g(), 20);
    EXPECT_EQ(pt.b(), 30);
}

TEST(PointViewTest, AccessNormals)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(3, 3);
    normals.setValue(0, 0, 0.0f); normals.setValue(0, 1, 0.0f); normals.setValue(0, 2, 1.0f);
    cloud.setNormals(std::move(normals));

    auto pt = cloud[0];
    EXPECT_FLOAT_EQ(pt.nx(), 0.0f);
    EXPECT_FLOAT_EQ(pt.ny(), 0.0f);
    EXPECT_FLOAT_EQ(pt.nz(), 1.0f);
}

TEST(PointViewTest, AccessTextureCoords)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(3, 2);
    tex.setValue(0, 0, 0.25f); tex.setValue(0, 1, 0.75f);
    cloud.setTextureCoords(std::move(tex));

    auto pt = cloud[0];
    EXPECT_FLOAT_EQ(pt.u(), 0.25f);
    EXPECT_FLOAT_EQ(pt.v(), 0.75f);
}
```

- [ ] **Step 2: Verify tests fail**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointViewTest*"
```

- [ ] **Step 3: Implement PointView**

In `point_cloud.h`, inside the `PointCloud` class, add as the first public member:

```cpp
    class PointView
    {
    public:
        Scalar x() const { return _cloud.points().getValue(static_cast<plamatrix::Index>(_idx), 0); }
        Scalar y() const { return _cloud.points().getValue(static_cast<plamatrix::Index>(_idx), 1); }
        Scalar z() const { return _cloud.points().getValue(static_cast<plamatrix::Index>(_idx), 2); }

        uint8_t r() const { return _cloud.colors()->getValue(static_cast<plamatrix::Index>(_idx), 0); }
        uint8_t g() const { return _cloud.colors()->getValue(static_cast<plamatrix::Index>(_idx), 1); }
        uint8_t b() const { return _cloud.colors()->getValue(static_cast<plamatrix::Index>(_idx), 2); }

        Scalar nx() const { return _cloud.normals()->getValue(static_cast<plamatrix::Index>(_idx), 0); }
        Scalar ny() const { return _cloud.normals()->getValue(static_cast<plamatrix::Index>(_idx), 1); }
        Scalar nz() const { return _cloud.normals()->getValue(static_cast<plamatrix::Index>(_idx), 2); }

        Scalar u() const { return _cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(_idx), 0); }
        Scalar v() const { return _cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(_idx), 1); }

    private:
        friend class PointCloud;
        PointView(const PointCloud& cloud, size_t idx) : _cloud(cloud), _idx(idx) {}
        const PointCloud& _cloud;
        size_t _idx;
    };

    PointView operator[](size_t idx) const { return PointView(*this, idx); }
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="PointViewTest*:PointCloudAttributesTest*"
```

- [ ] **Step 5: Commit**

```bash
cd /home/guderian/code/plapoint
git add include/plapoint/core/point_cloud.h test/unit/core/point_view_test.cpp
git commit -m "feat: add PointView accessor with named field access"
```

---

### Task 6: OBJ/MTL I/O (plapoint repo)

**Files:**
- Create: `include/plapoint/io/obj_io.h`
- Create: `test/unit/io/obj_io_test.cpp`

- [ ] **Step 1: Write the failing test**

In `test/unit/io/obj_io_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <plapoint/io/obj_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <fstream>
#include <cstdio>

TEST(ObjIoTest, WriteAndReadBackVertexOnly)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(4, 3);
    // Simple tetrahedron
    pts.setValue(0, 0, 0.0f); pts.setValue(0, 1, 0.0f); pts.setValue(0, 2, 0.0f);
    pts.setValue(1, 0, 1.0f); pts.setValue(1, 1, 0.0f); pts.setValue(1, 2, 0.0f);
    pts.setValue(2, 0, 0.5f); pts.setValue(2, 1, 1.0f); pts.setValue(2, 2, 0.0f);
    pts.setValue(3, 0, 0.5f); pts.setValue(3, 1, 0.5f); pts.setValue(3, 2, 1.0f);

    Cloud cloud(std::move(pts));

    std::string path = "/tmp/plapoint_test_vertex.obj";
    plapoint::io::writeObj<float>(path, cloud);

    auto read_back = plapoint::io::readObj<float>(path);
    ASSERT_NE(read_back, nullptr);
    EXPECT_EQ(read_back->size(), 4);
    EXPECT_FLOAT_EQ(read_back->points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(read_back->points().getValue(3, 2), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, WriteAndReadBackWithFaces)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    Cloud cloud(6);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces.setValue(0, 0, 0); faces.setValue(0, 1, 1); faces.setValue(0, 2, 2);
    faces.setValue(1, 0, 3); faces.setValue(1, 1, 4); faces.setValue(1, 2, 5);
    cloud.setFaces(std::move(faces));

    std::string path = "/tmp/plapoint_test_faces.obj";
    plapoint::io::writeObj<float>(path, cloud);

    auto read_back = plapoint::io::readObj<float>(path);
    ASSERT_NE(read_back, nullptr);
    EXPECT_EQ(read_back->size(), 6);
    ASSERT_TRUE(read_back->hasFaces());
    EXPECT_EQ(read_back->faces()->rows(), 2);
    EXPECT_EQ(read_back->faces()->getValue(0, 0), 0);

    std::remove(path.c_str());
}

TEST(ObjIoTest, WriteAndReadBackWithNormals)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    Cloud cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(3, 3);
    normals.setValue(0, 0, 0.0f); normals.setValue(0, 1, 0.0f); normals.setValue(0, 2, 1.0f);
    normals.setValue(1, 0, 1.0f); normals.setValue(1, 1, 0.0f); normals.setValue(1, 2, 0.0f);
    normals.setValue(2, 0, 0.0f); normals.setValue(2, 1, 1.0f); normals.setValue(2, 2, 0.0f);
    cloud.setNormals(std::move(normals));

    std::string path = "/tmp/plapoint_test_normals.obj";
    plapoint::io::writeObj<float>(path, cloud);

    auto read_back = plapoint::io::readObj<float>(path);
    ASSERT_NE(read_back, nullptr);
    EXPECT_EQ(read_back->size(), 3);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 2), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadNonExistentFileThrows)
{
    EXPECT_THROW(plapoint::io::readObj<float>("/tmp/nonexistent_plapoint.obj"), std::runtime_error);
}
```

- [ ] **Step 2: Verify tests fail (compile error — no obj_io.h yet)**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) 2>&1 | head -5
```

- [ ] **Step 3: Implement obj_io.h**

Since OBJ parsing is non-trivial (~200 lines), place the implementation in the header to follow plapoint's template pattern (same as ply_io.h). In `include/plapoint/io/obj_io.h`:

```cpp
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plapoint {
namespace io {

namespace detail {

struct ObjVertexIndices { int v, t, n; };

inline ObjVertexIndices parseFaceVertex(const std::string& s)
{
    ObjVertexIndices idx{-1, -1, -1};
    size_t slash1 = s.find('/');
    if (slash1 == std::string::npos) { idx.v = std::stoi(s) - 1; return idx; }
    idx.v = std::stoi(s.substr(0, slash1)) - 1;
    size_t slash2 = s.find('/', slash1 + 1);
    if (slash2 == std::string::npos || slash2 == slash1 + 1) { return idx; }
    if (slash2 > slash1 + 1) idx.t = std::stoi(s.substr(slash1 + 1, slash2 - slash1 - 1)) - 1;
    if (slash2 + 1 < s.size()) idx.n = std::stoi(s.substr(slash2 + 1)) - 1;
    return idx;
}

} // namespace detail

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readObj(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open OBJ file: " + path);

    std::vector<Scalar> vx, vy, vz;
    std::vector<Scalar> nx, ny, nz;
    std::vector<Scalar> tx, ty;
    std::vector<std::vector<int>> face_verts;
    std::vector<std::vector<int>> face_tex;
    std::string mtlLib;

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v")
        {
            Scalar x, y, z; iss >> x >> y >> z;
            vx.push_back(x); vy.push_back(y); vz.push_back(z);
        }
        else if (token == "vn")
        {
            Scalar x, y, z; iss >> x >> y >> z;
            nx.push_back(x); ny.push_back(y); nz.push_back(z);
        }
        else if (token == "vt")
        {
            Scalar u, v; iss >> u >> v;
            tx.push_back(u); ty.push_back(v);
        }
        else if (token == "f")
        {
            std::vector<int> fv, ft, fn;
            std::string s;
            while (iss >> s)
            {
                auto idx = detail::parseFaceVertex(s);
                fv.push_back(idx.v);
                if (idx.t >= 0) ft.push_back(idx.t);
                if (idx.n >= 0) fn.push_back(idx.n);
            }
            face_verts.push_back(fv);
            if (!ft.empty()) face_tex.push_back(ft);
        }
        else if (token == "mtllib")
        {
            iss >> mtlLib;
        }
    }

    size_t n = vx.size();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(static_cast<plamatrix::Index>(n), 3);
    for (size_t i = 0; i < n; ++i)
    {
        pts(static_cast<plamatrix::Index>(i), 0) = vx[i];
        pts(static_cast<plamatrix::Index>(i), 1) = vy[i];
        pts(static_cast<plamatrix::Index>(i), 2) = vz[i];
    }
    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));

    if (!nx.empty() && nx.size() == n)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(static_cast<plamatrix::Index>(n), 3);
        for (size_t i = 0; i < n; ++i)
        {
            nrm(static_cast<plamatrix::Index>(i), 0) = nx[i];
            nrm(static_cast<plamatrix::Index>(i), 1) = ny[i];
            nrm(static_cast<plamatrix::Index>(i), 2) = nz[i];
        }
        cloud->setNormals(std::move(nrm));
    }

    if (!tx.empty() && tx.size() == n)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> tex(static_cast<plamatrix::Index>(n), 2);
        for (size_t i = 0; i < n; ++i)
        {
            tex(static_cast<plamatrix::Index>(i), 0) = tx[i];
            tex(static_cast<plamatrix::Index>(i), 1) = ty[i];
        }
        cloud->setTextureCoords(std::move(tex));
    }

    if (!face_verts.empty())
    {
        plamatrix::Index nf = static_cast<plamatrix::Index>(face_verts.size());
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(nf, 3);
        for (plamatrix::Index fi = 0; fi < nf; ++fi)
            for (int c = 0; c < 3 && c < static_cast<int>(face_verts[fi].size()); ++c)
                faces(fi, c) = face_verts[fi][c];
        cloud->setFaces(std::move(faces));

        if (!face_tex.empty() && face_tex.size() == face_verts.size())
        {
            plamatrix::DenseMatrix<int, plamatrix::Device::CPU> ft(nf, 3);
            for (plamatrix::Index fi = 0; fi < nf; ++fi)
                for (int c = 0; c < 3 && c < static_cast<int>(face_tex[fi].size()); ++c)
                    ft(fi, c) = face_tex[fi][c];
            cloud->setFaceTextureIndices(std::move(ft));
        }
    }

    if (!mtlLib.empty()) cloud->setMaterialLibraryFile(mtlLib);

    return cloud;
}

template <typename Scalar>
void writeObj(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write OBJ file: " + path);

    const auto& mtlLib = cloud.materialLibraryFile();
    if (!mtlLib.empty()) f << "mtllib " << mtlLib << "\n";

    for (size_t i = 0; i < cloud.size(); ++i)
    {
        f << "v " << cloud.points()(static_cast<plamatrix::Index>(i), 0)
          << " " << cloud.points()(static_cast<plamatrix::Index>(i), 1)
          << " " << cloud.points()(static_cast<plamatrix::Index>(i), 2) << "\n";
    }

    if (cloud.hasNormals())
    {
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            f << "vn " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0)
              << " " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1)
              << " " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2) << "\n";
        }
    }

    if (cloud.hasTextureCoords())
    {
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            f << "vt " << cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(i), 0)
              << " " << cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(i), 1) << "\n";
        }
    }

    if (cloud.hasFaces())
    {
        bool use_tex = cloud.hasFaceTextureIndices();
        bool use_nrm = cloud.hasNormals();
        for (plamatrix::Index fi = 0; fi < cloud.faces()->rows(); ++fi)
        {
            f << "f";
            for (int c = 0; c < 3; ++c)
            {
                int vi = cloud.faces()->getValue(fi, c) + 1; // OBJ 1-indexed
                f << " " << vi;
                if (use_tex || use_nrm)
                {
                    f << "/";
                    if (use_tex) f << cloud.faceTextureIndices()->getValue(fi, c) + 1;
                    if (use_nrm) f << "/" << vi; // vn same index as v
                }
            }
            f << "\n";
        }
    }
}

} // namespace io
} // namespace plapoint
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /home/guderian/code/plapoint/build && cmake --build . -j$(nproc) && ./test/plapoint_tests --gtest_filter="ObjIoTest*"
```

Expected: PASS (4/4).

- [ ] **Step 5: Commit**

```bash
cd /home/guderian/code/plapoint
git add include/plapoint/io/obj_io.h test/unit/io/obj_io_test.cpp
git commit -m "feat: add OBJ/MTL read/write I/O"
```

---

### Task 7: Full plapoint build + test verification (plapoint repo)

**Files:** None new — verify all existing and new tests pass.

- [ ] **Step 1: Build entire plapoint**

```bash
cd /home/guderian/code/plapoint/build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)
```

Expected: zero warnings, zero errors.

- [ ] **Step 2: Run all plapoint tests**

```bash
cd /home/guderian/code/plapoint/build && ./test/plapoint_tests
```

Expected: all tests PASS (existing 29 + new ~15 = ~44 tests).

- [ ] **Step 3: Push plapoint**

```bash
cd /home/guderian/code/plapoint
git push origin main
```

---

### Task 8: Add plapoint dependency to plascan build (plascan repo)

**Files:**
- Modify: `CMakeLists.txt` (root)
- Modify: `cmake/PlascanPackages.cmake`

- [ ] **Step 1: Add find_package for plapoint**

In `cmake/PlascanPackages.cmake`, insert after the libzip block (line 85 `message(STATUS "plascan: found libzip...")`) and before the OpenMP section:

```cmake
# ── plapoint ──────────────────────────────────────────────────────────────────
find_package(plapoint REQUIRED)
message(STATUS "plascan: found plapoint")
```

This picks up plapoint from the system install prefix (where `plapointConfig.cmake` was installed by plapoint's build).

- [ ] **Step 2: Verify plapoint is findable**

```bash
cd /home/guderian/code/plascan/build && cmake .. 2>&1 | grep -i plapoint
```

Expected: prints something like "Found plapoint: ..." (no error).

- [ ] **Step 3: Commit**

```bash
cd /home/guderian/code/plascan
git add cmake/PlascanPackages.cmake
git commit -m "build: add plapoint dependency"
```

---

### Task 9: Add SparsePointCloud aggregator (plascan repo)

**Files:**
- Create: `src/core/sfm/common/SparsePointCloud.h`
- Modify: `src/core/sfm/common/SfmTypes.h` (if needed for includes)

- [ ] **Step 1: Create SparsePointCloud.h**

In `src/core/sfm/common/SparsePointCloud.h`:

```cpp
#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cassert>
#include <vector>
#include "PhotogrammetryPointAttributes.h"  // existing, move or include

namespace xjw {
namespace sfm {

struct SparsePointCloud
{
    using GeometryType = plapoint::PointCloud<float, plamatrix::Device::CPU>;

    GeometryType geometry;
    std::vector<PhotogrammetryPointAttributes> attributes;

    size_t size() const
    {
        assert(geometry.size() == attributes.size());
        return geometry.size();
    }

    bool empty() const { return size() == 0; }

    void clear()
    {
        geometry = GeometryType();
        attributes.clear();
    }
};

} // namespace sfm
} // namespace xjw
```

Note: `PhotogrammetryPointAttributes` currently lives in `src/core/pointcloud/data/PointCloudPoint.h`. Copy the struct definition into `src/core/sfm/common/PhotogrammetryPointAttributes.h` to decouple from the to-be-deleted pointcloud module.

- [ ] **Step 2: Move PhotogrammetryPointAttributes**

Read `src/core/pointcloud/data/PointCloudPoint.h`, extract the `PhotogrammetryPointAttributes` struct, write to `src/core/sfm/common/PhotogrammetryPointAttributes.h`:

```cpp
#pragma once

namespace xjw {
namespace sfm {

struct PhotogrammetryPointAttributes
{
    int pointId = -1;
    int trackLength = 0;
    float reprojectionError = 0.0f;
    float confidence = 0.0f;
    bool isControlPoint = false;
    bool isValid = true;
};

} // namespace sfm
} // namespace xjw
```

- [ ] **Step 3: Commit**

```bash
cd /home/guderian/code/plascan
git add src/core/sfm/common/SparsePointCloud.h src/core/sfm/common/PhotogrammetryPointAttributes.h
git commit -m "feat: add SparsePointCloud aggregator, extract photogrammetry attrs from pointcloud module"
```

---

### Task 10: Adapt terrain module (plascan repo)

**Files:**
- Modify: `src/core/terrain/DemGenerator.h`
- Modify: `src/core/terrain/DemGenerator.cpp`
- Modify: `src/core/terrain/DemDomIO.h`
- Modify: `src/core/terrain/DemDomIO.cpp`
- Modify: `src/core/terrain/TerrainPipeline.cpp`
- Modify: `src/core/terrain/projection/AsteroidProjection.h`
- Modify: `src/core/terrain/projection/AsteroidProjection.cpp`
- Modify: `src/core/terrain/CMakeLists.txt`

This task is a pure refactoring task — replace `xjw::pointcloud::PointCloud` with `plapoint::PointCloud<float, plamatrix::CPU>`. The existing terrain tests serve as regression tests.

- [ ] **Step 1: Ensure terrain tests pass before changes**

```bash
cd /home/guderian/code/plascan/build && cmake --build . -j$(nproc) --target terrain_tests 2>&1 | tail -5 && ./tests/terrain_tests
```

- [ ] **Step 2: Update DemGenerator.h**

Replace include `#include "data/PointCloud.h"` with `#include <plapoint/core/point_cloud.h>`. Replace all `const pointcloud::PointCloud&` with `const plapoint::PointCloud<float, plamatrix::Device::CPU>&`. Replace `pointcloud::PointCloud*` with `plapoint::PointCloud<float, plamatrix::Device::CPU>*`.

- [ ] **Step 3: Update DemGenerator.cpp**

Replace:
- `cloud.positions()[i]` → `cloud[i]` (PointView)
- `cloud.positions().size()` → `cloud.size()`
- `cloud.colors()[i]` → pass PointView to the rasterizer
- Access pattern: `pos.x` → `pt.x()`, `pos.y` → `pt.y()`, `pos.z` → `pt.z()`

- [ ] **Step 4: Update DemDomIO.h/cpp**

Replace `pointcloud::PointCloud` type references with `plapoint::PointCloud<float, plamatrix::Device::CPU>`. Replace `PointCloudIO::writePlyPointCloud()` calls with `plapoint::io::writePly<float>()`.

- [ ] **Step 5: Update TerrainPipeline.cpp**

Replace `readPointCloud(path, &pc, opts)` with `plapoint::io::readPly<float>(path)` (or the appropriate I/O function based on file extension).

- [ ] **Step 6: Update AsteroidProjection.h/cpp**

Replace `pointcloud::PointCloud&` params. Replace `cloud.positions()` access with matrix access.

- [ ] **Step 7: Update terrain/CMakeLists.txt**

Replace `target_link_libraries(terrain PUBLIC pointcloud)` with `target_link_libraries(terrain PUBLIC plapoint::plapoint)`.

- [ ] **Step 8: Verify terrain compiles**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target terrain 2>&1 | tail -20
```

- [ ] **Step 9: Commit**

```bash
cd /home/guderian/code/plascan
git add src/core/terrain/
git commit -m "refactor(terrain): switch from pointcloud module to plapoint"
```

---

### Task 11: Adapt mesh module (plascan repo)

**Files:**
- Modify: `src/core/mesh/TextureMapper.h/cpp`
- Modify: `src/core/mesh/ModelWorkflowService.h/cpp`
- Modify: `src/core/mesh/SurfaceReconstructor.h/cpp`
- Modify: `src/core/mesh/CMakeLists.txt`

- [ ] **Step 1: Update TextureMapper**

Replace `xjw::pointcloud::PointCloud` → `plapoint::PointCloud<float, plamatrix::Device::CPU>`. Replace:
- `cloud.positions()`, `cloud.normals()`, `cloud.colors()`, `cloud.textureCoordinates()` → plapoint PointView or matrix access
- `cloud.faces()` → `cloud.faces()` (same name, different type — now `DenseMatrix<int>`)
- `PointCloudIO::readPointCloud()` → `plapoint::io::readObj<float>()`
- `PointCloudIO::writePointCloud()` → `plapoint::io::writeObj<float>()`

- [ ] **Step 2: Update ModelWorkflowService**

Replace `PointCloudQualityReport` internals to use plapoint types.

- [ ] **Step 3: Update SurfaceReconstructor**

Replace:
- `SurfaceReconstructorIO::loadPointCloud()` → `plapoint::io::readPly<float>()` or `plapoint::io::readXyz<float>()` depending on extension
- MarchingCubes calls → `plapoint::mesh::MarchingCubes<Scalar>`
- Poisson reconstruction calls → `plapoint::mesh::PoissonReconstruction<Scalar>`

Remove includes of `MarchingCubesTable.h`, `poisson/*.h`, `SurfaceReconstructorIO.h`.

- [ ] **Step 4: Update mesh/CMakeLists.txt**

Replace `pointcloud` dep with `plapoint::plapoint`. Remove references to deleted Poisson/MC/SurfaceReconstructorIO source files.

- [ ] **Step 5: Verify mesh compiles and tests pass**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target meshing 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

---

### Task 12: Adapt mvs module (plascan repo)

**Files:**
- Modify: `src/core/mvs/StereoDenseCloudPipelineOutput.h/cpp`
- Modify: `src/core/mvs/CMakeLists.txt`

- [ ] **Step 1: Update StereoDenseCloudPipelineOutput**

Replace `pointcloud::PointCloud` local variable creation and population with `plapoint::PointCloud<float, CPU>`. Replace `writePlyPointCloud()` → `plapoint::io::writePly<float>()`.

- [ ] **Step 2: Update mvs/CMakeLists.txt**

Replace `PRIVATE pointcloud` with `PRIVATE plapoint::plapoint`.

- [ ] **Step 3: Verify mvs compiles**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target mvs 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

---

### Task 13: Adapt pipeline module (plascan repo)

**Files:**
- Modify: `src/core/pipeline/SFMService.h/cpp`
- Modify: `src/core/pipeline/CMakeLists.txt`

- [ ] **Step 1: Update SFMService**

Replace `xjw::pointcloud::PointCloud cloud` member and `writePointCloud()` calls with plapoint types and `plapoint::io::writePly<float>()`.

- [ ] **Step 2: Update pipeline/CMakeLists.txt** (if it links pointcloud)

- [ ] **Step 3: Verify pipeline compiles**

- [ ] **Step 4: Commit**

---

### Task 14: Adapt sfm filtering module (plascan repo)

**Files:**
- Modify: `src/core/sfm/filtering/SparsePointCloudProcessor.h/cpp`
- Modify: `src/core/sfm/filtering/SfmPointCloudFilter.h/cpp`
- Modify: `tests/test_sparse_point_cloud_processor.cpp`
- Modify: `tests/test_sfm_filter.cpp`

- [ ] **Step 1: Update SparsePointCloudProcessor**

Replace `SparsePointCloudPoint` struct usage with `SparsePointCloud` aggregator. The processor's `optimize(SparsePointCloud&)` method works on the new type. Internal logic that accesses `x, y, z, trackLen, rmsReprojPx` now goes through `cloud.geometry[i]` for position and `cloud.attributes[i].trackLength` etc. for metadata.

- [ ] **Step 2: Update SfmPointCloudFilter**

Replace any `PointCloud` references with `SparsePointCloud`.

- [ ] **Step 3: Update tests**

`test_sparse_point_cloud_processor.cpp`: Replace `SparsePointCloudPoint` construction with `SparsePointCloud` construction.

`test_sfm_filter.cpp`: Replace type references.

- [ ] **Step 4: Verify sfm filtering compiles and tests pass**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target sfm 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

---

### Task 15: Adapt GUI module (plascan repo)

**Files:**
- Modify: `src/gui/dialogs/CameraModel3DDialog.h`
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`
- Modify: `src/gui/CMakeLists.txt`

- [ ] **Step 1: Update CameraModel3DDialog header**

Replace `#include "data/PointCloud.h"` with `#include <plapoint/core/point_cloud.h>`. Replace `pointcloud::PointCloud m_cloud` member with `plapoint::PointCloud<float, plamatrix::Device::CPU> m_cloud`. Update undo stack `QStack<pointcloud::PointCloud>` → use the copy-semantics of plapoint::PointCloud (default copy works via DenseMatrix copy).

- [ ] **Step 2: Update CameraModel3DDialog.cpp**

Replace:
- `m_cloud.positions()`, `.normals()`, `.colors()` → matrix access + PointView
- `PointCloudIO::readPointCloud()` / `loadPointCloudFromXyz()` → plapoint I/O
- OpenGL rendering loops: `cloud.positions()[i]` → `cloud[i].x(), cloud[i].y(), cloud[i].z()`
- Manual pruning operations on `PointCloud` → same operations on plapoint::PointCloud (build new Nx3 matrix with kept points)

- [ ] **Step 3: Update gui/CMakeLists.txt**

Replace `pointcloud` link dep with `plapoint::plapoint`.

- [ ] **Step 4: Verify GUI compiles**

```bash
cd /home/guderian/code/plascan/build && cmake .. && cmake --build . -j$(nproc) --target plascan_gui 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

---

### Task 16: Delete common/spatial and pointcloud modules (plascan repo)

**Files to delete:**
- `src/common/spatial/KDTree.h`
- `src/common/spatial/KDTree2D.h`
- `src/common/spatial/KDTree3D.h`
- `src/common/spatial/tests/` (all)
- `src/core/pointcloud/` (entire directory — data/, io/, processing/, tests/)

**Files to modify:**
- `src/common/CMakeLists.txt` — remove spatial subdirectory
- `src/core/CMakeLists.txt` — remove pointcloud subdirectory
- Any remaining includes referencing these deleted files

- [ ] **Step 1: Remove spatial subdirectory from common/CMakeLists.txt**

Remove `add_subdirectory(spatial)` or equivalent.

- [ ] **Step 2: Remove pointcloud subdirectory from core/CMakeLists.txt**

Remove the `plascan_core_add_optional_module(pointcloud ...)` line.

- [ ] **Step 3: Delete the directories**

```bash
cd /home/guderian/code/plascan
git rm -r src/common/spatial/
git rm -r src/core/pointcloud/
git rm src/core/mesh/MarchingCubesTable.h
git rm src/core/mesh/MarchingCubesTable.cpp
git rm -r src/core/mesh/poisson/
git rm src/core/mesh/SurfaceReconstructorIO.h
git rm src/core/mesh/SurfaceReconstructorIO.cpp
```

- [ ] **Step 4: Fix any remaining includes**

```bash
grep -r "common/spatial" src/ tests/ || echo "none"
grep -r "pointcloud/" src/ tests/ | grep -v "SparsePointCloud.h" | grep -v "PhotogrammetryPointAttributes" || echo "none"
```

If there are remaining references, fix each one.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove pointcloud module, common/spatial, mesh MC/Poisson (replaced by plapoint)"
```

---

### Task 17: Update top-level tests and CMake (plascan repo)

**Files:**
- Modify: `tests/CMakeLists.txt`
- Modify: `tests/test_sparse_point_cloud_processor.cpp`
- Modify: `tests/test_sfm_filter.cpp`
- Modify: `tests/test_initial_sparse_triangulator.cpp` (if it uses PointCloud types)
- Modify: `tests/test_sfm_params.cpp` (if it uses PointCloud types)

- [ ] **Step 1: Update tests/CMakeLists.txt**

Remove `pointcloud` and `common_spatial` from `target_link_libraries`. Add `plapoint::plapoint` if needed by SFM tests (which it will be transitively through sfm).

- [ ] **Step 2: Update test source files**

Replace any `SparsePointCloudPoint` → `xjw::sfm::SparsePointCloud`. Replace any includes of pointcloud headers with appropriate plapoint or new sfm headers.

- [ ] **Step 3: Build and run all plascan tests**

```bash
cd /home/guderian/code/plascan/build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) && ctest --output-on-failure
```

Fix any compilation errors or test failures. Iterate until green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: update top-level tests for plapoint migration, fix all build issues"
```

---

### Task 18: Final verification — full build + full test suite (plascan repo)

- [ ] **Step 1: Clean rebuild**

```bash
cd /home/guderian/code/plascan/build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j$(nproc) 2>&1 | tail -30
```

Expected: zero errors, zero warnings.

- [ ] **Step 2: Run full test suite**

```bash
cd /home/guderian/code/plascan/build && ctest --output-on-failure
```

Expected: all tests pass (number may decrease from 162 since pointcloud/spatial tests were removed, offset by any new sfm tests).

- [ ] **Step 3: Git operations**

```bash
cd /home/guderian/code/plascan
git add -A
git status
git commit -m "refactor: finalize plapoint migration — all tests pass"
git push origin main
```

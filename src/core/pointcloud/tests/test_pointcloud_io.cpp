#include <gtest/gtest.h>

#include "data/PointCloud.h"
#include "io/PointCloudIO.h"

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace xjw::pointcloud;

namespace
{

/**
 * @brief 点云 IO 模块测试夹具。
 *
 * 每个测试用例都会创建一个临时目录，用于写出 PLY/OBJ 文件，
 * 在测试结束后自动删除，避免污染工作区。
 */
class PointCloudIoTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        _tempDir = std::filesystem::temp_directory_path() /
                   ("plascan_pointcloud_test_" + std::to_string(now));
        std::filesystem::create_directories(_tempDir);
    }

    void TearDown() override
    {
        std::error_code errorCode;
        std::filesystem::remove_all(_tempDir, errorCode);
    }

    /**
     * @brief 构造一个包含位置、法向量和颜色的样例点云。
     *
     * 该样例用于验证：
     * 1. `PointCloud` 基本属性保存是否正确。
     * 2. PLY/OBJ 写出后再读回时，坐标和附属属性是否保持一致。
     */
    PointCloud makeSampleCloud() const
    {
        PointCloud cloud;
        cloud.addPoint(Point3f{0.0f, 1.0f, 2.0f}, Point3f{0.0f, 0.0f, 1.0f}, ColorRGBA{255, 0, 0, 255});
        cloud.addPoint(Point3f{3.5f, 4.5f, 5.5f}, Point3f{0.0f, 1.0f, 0.0f}, ColorRGBA{0, 255, 0, 200});
        cloud.addPoint(Point3f{-1.0f, -2.0f, 7.0f}, Point3f{1.0f, 0.0f, 0.0f}, ColorRGBA{0, 0, 255, 128});
        return cloud;
    }

protected:
    std::filesystem::path _tempDir;
};

/**
 * @brief 比较两个三维向量是否在浮点误差范围内一致。
 */
void expectPointNear(const Point3f &lhs, const Point3f &rhs)
{
    EXPECT_NEAR(lhs.x, rhs.x, 1e-5f);
    EXPECT_NEAR(lhs.y, rhs.y, 1e-5f);
    EXPECT_NEAR(lhs.z, rhs.z, 1e-5f);
}

void expectTexCoordNear(const Point2f &lhs, const Point2f &rhs)
{
    EXPECT_NEAR(lhs.u, rhs.u, 1e-5f);
    EXPECT_NEAR(lhs.v, rhs.v, 1e-5f);
}

/**
 * @brief 比较两个颜色是否完全一致。
 */
void expectColorEq(const ColorRGBA &lhs, const ColorRGBA &rhs)
{
    EXPECT_EQ(lhs.r, rhs.r);
    EXPECT_EQ(lhs.g, rhs.g);
    EXPECT_EQ(lhs.b, rhs.b);
    EXPECT_EQ(lhs.a, rhs.a);
}

} // namespace

/**
 * @brief 验证 `PointCloud` 的基础操作。
 *
 * 覆盖点添加、颜色补齐、边界盒、质心、平移和缩放等行为。
 */
TEST_F(PointCloudIoTest, PointCloudBasicOperations)
{
    PointCloud cloud;
    cloud.addPoint(Point3f{1.0f, 2.0f, 3.0f});
    cloud.addPoint(Point3f{-1.0f, 4.0f, 5.0f}, ColorRGBA{10, 20, 30, 255});

    EXPECT_EQ(cloud.size(), 2U);
    EXPECT_TRUE(cloud.hasColors());
    EXPECT_TRUE(cloud.isConsistent());

    const PointCloudBounds bounds = cloud.computeBounds();
    EXPECT_TRUE(bounds.valid);
    expectPointNear(bounds.minCorner, Point3f{-1.0f, 2.0f, 3.0f});
    expectPointNear(bounds.maxCorner, Point3f{1.0f, 4.0f, 5.0f});

    const Point3f centroid = cloud.computeCentroid();
    expectPointNear(centroid, Point3f{0.0f, 3.0f, 4.0f});

    cloud.translate(Point3f{1.0f, -1.0f, 2.0f});
    expectPointNear(cloud.positions()[0], Point3f{2.0f, 1.0f, 5.0f});

    cloud.scale(0.5f);
    expectPointNear(cloud.positions()[1], Point3f{0.0f, 1.5f, 3.5f});
}

/**
 * @brief 验证单点属性修改、摄影测量属性和点云元属性。
 */
TEST_F(PointCloudIoTest, PointCloudSinglePointAndMetadataOperations)
{
    PointCloud cloud;
    cloud.addPoint(Point3f{0.0f, 0.0f, 0.0f});
    cloud.addPoint(Point3f{1.0f, 1.0f, 1.0f});

    ASSERT_TRUE(cloud.setPosition(1, Point3f{2.0f, 3.0f, 4.0f}));
    ASSERT_TRUE(cloud.setNormal(1, Point3f{0.0f, 0.0f, 1.0f}));
    ASSERT_TRUE(cloud.setColor(1, ColorRGBA{10, 20, 30, 40}));
    ASSERT_TRUE(cloud.setPhotogrammetryAttributes(1, PhotogrammetryPointAttributes{42, 5, 0.25f, 0.9f, true, true}));

    PointCloudMetadata metadata;
    metadata.name = "ba_sparse";
    metadata.sourcePath = "/tmp/ba_sparse.ply";
    metadata.coordinateSystem = "UTM";
    metadata.coordinateFrame = PointCloudCoordinateFrame::World;
    metadata.isRegistered = true;
    cloud.setMetadata(metadata);

    expectPointNear(cloud.positions()[1], Point3f{2.0f, 3.0f, 4.0f});
    ASSERT_NE(cloud.normalAt(1), nullptr);
    ASSERT_NE(cloud.colorAt(1), nullptr);
    ASSERT_NE(cloud.photogrammetryAttributesAt(1), nullptr);
    expectPointNear(*cloud.normalAt(1), Point3f{0.0f, 0.0f, 1.0f});
    expectColorEq(*cloud.colorAt(1), ColorRGBA{10, 20, 30, 40});
    EXPECT_EQ(cloud.photogrammetryAttributesAt(1)->pointId, 42);
    EXPECT_EQ(cloud.photogrammetryAttributesAt(1)->trackLength, 5);
    EXPECT_NEAR(cloud.photogrammetryAttributesAt(1)->reprojectionError, 0.25f, 1e-6f);
    EXPECT_NEAR(cloud.photogrammetryAttributesAt(1)->confidence, 0.9f, 1e-6f);
    EXPECT_TRUE(cloud.photogrammetryAttributesAt(1)->isControlPoint);
    EXPECT_EQ(cloud.metadata().name, "ba_sparse");
    EXPECT_EQ(cloud.metadata().coordinateFrame, PointCloudCoordinateFrame::World);
    EXPECT_TRUE(cloud.metadata().isRegistered);

    PointCloudPoint point = cloud.pointAt(1);
    EXPECT_TRUE(point.hasNormal);
    EXPECT_TRUE(point.hasColor);
    EXPECT_TRUE(point.hasPhotogrammetry);

    point.setTextureCoordinate(Point2f{0.25f, 0.75f});
    point.position = Point3f{5.0f, 6.0f, 7.0f};
    ASSERT_TRUE(cloud.setPoint(1, point));
    ASSERT_TRUE(cloud.hasTextureCoordinates());
    ASSERT_NE(cloud.textureCoordinateAt(1), nullptr);
    expectTexCoordNear(*cloud.textureCoordinateAt(1), Point2f{0.25f, 0.75f});
    expectPointNear(cloud.positions()[1], Point3f{5.0f, 6.0f, 7.0f});
}

/**
 * @brief 验证筛选、裁剪、中心缩放和坐标变换接口。
 */
TEST_F(PointCloudIoTest, PointCloudFilterCropAndTransformOperations)
{
    PointCloud cloud;
    cloud.addPoint(Point3f{0.0f, 0.0f, 0.0f}, Point3f{1.0f, 0.0f, 0.0f}, ColorRGBA{255, 0, 0, 255}, PhotogrammetryPointAttributes{1, 2, 0.2f, 0.2f, false, true});
    cloud.addPoint(Point3f{1.0f, 1.0f, 1.0f}, Point3f{0.0f, 1.0f, 0.0f}, ColorRGBA{0, 255, 0, 255}, PhotogrammetryPointAttributes{2, 3, 0.1f, 0.8f, false, true});
    cloud.addPoint(Point3f{5.0f, 5.0f, 5.0f}, Point3f{0.0f, 0.0f, 1.0f}, ColorRGBA{0, 0, 255, 255}, PhotogrammetryPointAttributes{3, 4, 0.6f, 0.4f, false, false});

    const std::vector<std::size_t> selected = cloud.selectIndices(
        [](std::size_t,
           const Point3f &,
           const Point3f *,
           const ColorRGBA *,
           const PhotogrammetryPointAttributes *photogrammetry) {
            return photogrammetry && photogrammetry->confidence >= 0.5f && photogrammetry->isValid;
        });
    ASSERT_EQ(selected.size(), 1U);
    EXPECT_EQ(selected[0], 1U);

    const PointCloud filtered = cloud.filter(
        [](std::size_t,
           const Point3f &,
           const Point3f *,
           const ColorRGBA *,
           const PhotogrammetryPointAttributes *photogrammetry) {
            return photogrammetry && photogrammetry->reprojectionError <= 0.2f;
        });
    ASSERT_EQ(filtered.size(), 2U);

    PointCloudBounds cropBounds;
    cropBounds.valid = true;
    cropBounds.minCorner = Point3f{-0.1f, -0.1f, -0.1f};
    cropBounds.maxCorner = Point3f{2.0f, 2.0f, 2.0f};
    const PointCloud cropped = cloud.crop(cropBounds);
    ASSERT_EQ(cropped.size(), 2U);

    cloud.scale(2.0f, Point3f{1.0f, 1.0f, 1.0f});
    expectPointNear(cloud.positions()[0], Point3f{-1.0f, -1.0f, -1.0f});
    expectPointNear(cloud.positions()[1], Point3f{1.0f, 1.0f, 1.0f});

    const PointCloudTransform transform = PointCloudTransform::fromRotationTranslation(
        std::array<float, 9>{1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 1.0f},
        Point3f{10.0f, 0.0f, -2.0f});
    const PointCloud transformed = cropped.transformed(transform);
    expectPointNear(transformed.positions()[0], Point3f{10.0f, 0.0f, -2.0f});
    expectPointNear(transformed.positions()[1], Point3f{11.0f, 1.0f, -1.0f});
}

/**
 * @brief 验证导出到 CUDA 紧凑布局时的数组组织是否正确。
 */
TEST_F(PointCloudIoTest, PointCloudCudaExport)
{
    PointCloud cloud = makeSampleCloud();
    cloud.setPhotogrammetryAttributes({
        PhotogrammetryPointAttributes{10, 2, 0.1f, 0.7f, false, true},
        PhotogrammetryPointAttributes{11, 3, 0.2f, 0.8f, false, true},
        PhotogrammetryPointAttributes{12, 4, 0.3f, 0.9f, true, true}});

    const PointCloudCudaExport exportData = cloud.exportToCuda();
    EXPECT_EQ(exportData.positionsXyz.size(), cloud.size() * 3U);
    EXPECT_EQ(exportData.normalsXyz.size(), cloud.size() * 3U);
    EXPECT_EQ(exportData.colorsRgba.size(), cloud.size() * 4U);
    EXPECT_EQ(exportData.confidences.size(), cloud.size());
    EXPECT_EQ(exportData.reprojectionErrors.size(), cloud.size());
    EXPECT_TRUE(exportData.hasNormals);
    EXPECT_TRUE(exportData.hasColors);
    EXPECT_TRUE(exportData.hasPhotogrammetryAttributes);
    EXPECT_FLOAT_EQ(exportData.positionsXyz[0], 0.0f);
    EXPECT_FLOAT_EQ(exportData.positionsXyz[1], 1.0f);
    EXPECT_FLOAT_EQ(exportData.positionsXyz[2], 2.0f);
    EXPECT_EQ(exportData.colorsRgba[0], 255);
    EXPECT_EQ(exportData.colorsRgba[1], 0);
    EXPECT_EQ(exportData.colorsRgba[2], 0);
    EXPECT_EQ(exportData.colorsRgba[3], 255);
    EXPECT_NEAR(exportData.confidences[2], 0.9f, 1e-6f);
    EXPECT_NEAR(exportData.reprojectionErrors[1], 0.2f, 1e-6f);
}

/**
 * @brief 验证 ASCII PLY 往返读写。
 */
TEST_F(PointCloudIoTest, PlyAsciiRoundTrip)
{
    const std::filesystem::path plyPath = _tempDir / "sample_ascii.ply";
    const PointCloud source = makeSampleCloud();

    PointCloudIOResult writeResult;
    PointCloudWriteOptions writeOptions;
    writeOptions.format = PointCloudFileFormat::PlyAscii;
    ASSERT_TRUE(writePointCloud(plyPath.string(), source, writeOptions, &writeResult))
        << writeResult.errorMessage;

    PointCloud loaded;
    PointCloudIOResult readResult;
    ASSERT_TRUE(readPointCloud(plyPath.string(), &loaded, {}, &readResult))
        << readResult.errorMessage;

    EXPECT_EQ(loaded.size(), source.size());
    EXPECT_TRUE(loaded.hasNormals());
    EXPECT_TRUE(loaded.hasColors());

    for (std::size_t index = 0; index < source.size(); ++index)
    {
        expectPointNear(loaded.positions()[index], source.positions()[index]);
        expectPointNear(loaded.normals()[index], source.normals()[index]);
        expectColorEq(loaded.colors()[index], source.colors()[index]);
    }
}

/**
 * @brief 验证二进制 PLY 往返读写。
 */
TEST_F(PointCloudIoTest, PlyBinaryRoundTrip)
{
    const std::filesystem::path plyPath = _tempDir / "sample_binary.ply";
    const PointCloud source = makeSampleCloud();

    PointCloudIOResult writeResult;
    PointCloudWriteOptions writeOptions;
    writeOptions.format = PointCloudFileFormat::PlyBinaryLittleEndian;
    ASSERT_TRUE(writePointCloud(plyPath.string(), source, writeOptions, &writeResult))
        << writeResult.errorMessage;

    PointCloud loaded;
    PointCloudIOResult readResult;
    ASSERT_TRUE(readPointCloud(plyPath.string(), &loaded, {}, &readResult))
        << readResult.errorMessage;

    EXPECT_EQ(readResult.detectedFormat, PointCloudFileFormat::PlyBinaryLittleEndian);
    EXPECT_EQ(loaded.size(), source.size());
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        expectPointNear(loaded.positions()[index], source.positions()[index]);
        expectPointNear(loaded.normals()[index], source.normals()[index]);
        expectColorEq(loaded.colors()[index], source.colors()[index]);
    }
}

/**
 * @brief 验证带三角面的二进制 PLY 往返读写。
 */
TEST_F(PointCloudIoTest, PlyBinaryMeshRoundTrip)
{
    const std::filesystem::path plyPath = _tempDir / "sample_mesh_binary.ply";

    PointCloud source;
    source.addPoint(Point3f{0.0f, 0.0f, 0.0f}, Point3f{0.0f, 0.0f, 1.0f}, ColorRGBA{255, 0, 0, 255});
    source.addPoint(Point3f{1.0f, 0.0f, 0.0f}, Point3f{0.0f, 0.0f, 1.0f}, ColorRGBA{0, 255, 0, 255});
    source.addPoint(Point3f{0.0f, 1.0f, 0.0f}, Point3f{0.0f, 0.0f, 1.0f}, ColorRGBA{0, 0, 255, 255});

    PointCloudFace face;
    face.vertexIndices = {0U, 1U, 2U};
    ASSERT_TRUE(source.addFace(face));

    PointCloudWriteOptions writeOptions;
    writeOptions.format = PointCloudFileFormat::PlyBinaryLittleEndian;

    PointCloudIOResult writeResult;
    ASSERT_TRUE(writePointCloud(plyPath.string(), source, writeOptions, &writeResult))
        << writeResult.errorMessage;

    PointCloud loaded;
    PointCloudIOResult readResult;
    ASSERT_TRUE(readPointCloud(plyPath.string(), &loaded, {}, &readResult))
        << readResult.errorMessage;

    ASSERT_TRUE(loaded.hasFaces());
    ASSERT_EQ(loaded.faces().size(), 1U);
    EXPECT_EQ(loaded.faces()[0].vertexIndices[0], 0U);
    EXPECT_EQ(loaded.faces()[0].vertexIndices[1], 1U);
    EXPECT_EQ(loaded.faces()[0].vertexIndices[2], 2U);
    EXPECT_EQ(loaded.size(), 3U);
}

/**
 * @brief 验证 OBJ 顶点云往返读写。
 *
 * OBJ 颜色一般按浮点值写出，再读取时存在量化误差，
 * 因此颜色断言允许 1 个量级以内的误差。
 */
TEST_F(PointCloudIoTest, ObjRoundTrip)
{
    const std::filesystem::path objPath = _tempDir / "sample.obj";
    const PointCloud source = makeSampleCloud();

    PointCloudIOResult writeResult;
    PointCloudWriteOptions writeOptions;
    writeOptions.format = PointCloudFileFormat::Obj;
    ASSERT_TRUE(writePointCloud(objPath.string(), source, writeOptions, &writeResult))
        << writeResult.errorMessage;

    PointCloud loaded;
    PointCloudIOResult readResult;
    ASSERT_TRUE(readPointCloud(objPath.string(), &loaded, {}, &readResult))
        << readResult.errorMessage;

    EXPECT_EQ(loaded.size(), source.size());
    EXPECT_TRUE(loaded.hasNormals());
    EXPECT_TRUE(loaded.hasColors());
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        expectPointNear(loaded.positions()[index], source.positions()[index]);
        expectPointNear(loaded.normals()[index], source.normals()[index]);
        EXPECT_NEAR(static_cast<float>(loaded.colors()[index].r), static_cast<float>(source.colors()[index].r), 1.0f);
        EXPECT_NEAR(static_cast<float>(loaded.colors()[index].g), static_cast<float>(source.colors()[index].g), 1.0f);
        EXPECT_NEAR(static_cast<float>(loaded.colors()[index].b), static_cast<float>(source.colors()[index].b), 1.0f);
    }
}

/**
 * @brief 验证 OBJ 纹理坐标、三角面与 MTL 输出。
 */
TEST_F(PointCloudIoTest, ObjTexturedMeshRoundTripAndMtl)
{
    const std::filesystem::path objPath = _tempDir / "sample_textured.obj";

    PointCloud source;
    source.addPoint(Point3f{0.0f, 0.0f, 0.0f});
    source.addPoint(Point3f{1.0f, 0.0f, 0.0f});
    source.addPoint(Point3f{0.0f, 1.0f, 0.0f});
    source.setTextureCoordinates({
        Point2f{0.0f, 0.0f},
        Point2f{1.0f, 0.0f},
        Point2f{0.0f, 1.0f}});

    PointCloudFace face;
    face.vertexIndices = {0U, 1U, 2U};
    face.textureIndices = {0U, 1U, 2U};
    face.hasTextureIndices = true;
    ASSERT_TRUE(source.addFace(face));

    PointCloudIOResult writeResult;
    PointCloudWriteOptions writeOptions;
    writeOptions.format = PointCloudFileFormat::Obj;
    writeOptions.materialLibraryFile = "sample_textured.mtl";
    writeOptions.materialName = "mesh_mat";
    writeOptions.textureImageFile = "textures/recon.png";
    ASSERT_TRUE(writePointCloud(objPath.string(), source, writeOptions, &writeResult))
        << writeResult.errorMessage;

    EXPECT_TRUE(std::filesystem::exists(_tempDir / "sample_textured.mtl"));

    PointCloud loaded;
    PointCloudIOResult readResult;
    ASSERT_TRUE(readPointCloud(objPath.string(), &loaded, {}, &readResult))
        << readResult.errorMessage;

    ASSERT_EQ(loaded.size(), 3U);
    ASSERT_TRUE(loaded.hasTextureCoordinates());
    ASSERT_TRUE(loaded.hasFaces());
    EXPECT_EQ(loaded.faces().size(), 1U);
    expectTexCoordNear(loaded.textureCoordinates()[0], Point2f{0.0f, 0.0f});
    expectTexCoordNear(loaded.textureCoordinates()[1], Point2f{1.0f, 0.0f});
    expectTexCoordNear(loaded.textureCoordinates()[2], Point2f{0.0f, 1.0f});
    EXPECT_EQ(loaded.faces()[0].vertexIndices[0], 0U);
    EXPECT_EQ(loaded.faces()[0].vertexIndices[1], 1U);
    EXPECT_EQ(loaded.faces()[0].vertexIndices[2], 2U);
    EXPECT_EQ(loaded.materialLibraryFile(), "sample_textured.mtl");
}

/**
 * @brief 验证 XYZ 文本往返读写。
 */
TEST_F(PointCloudIoTest, XyzRoundTrip)
{
        const std::filesystem::path xyzPath = _tempDir / "sample.xyz";
        const PointCloud source = makeSampleCloud();

        PointCloudIOResult writeResult;
        PointCloudWriteOptions writeOptions;
        writeOptions.format = PointCloudFileFormat::Xyz;
        ASSERT_TRUE(writePointCloud(xyzPath.string(), source, writeOptions, &writeResult))
                << writeResult.errorMessage;

        PointCloud loaded;
        PointCloudIOResult readResult;
        ASSERT_TRUE(readPointCloud(xyzPath.string(), &loaded, {}, &readResult))
                << readResult.errorMessage;

        EXPECT_EQ(readResult.detectedFormat, PointCloudFileFormat::Xyz);
        EXPECT_EQ(loaded.size(), source.size());
        EXPECT_TRUE(loaded.hasColors());
        for (std::size_t index = 0; index < source.size(); ++index)
        {
                expectPointNear(loaded.positions()[index], source.positions()[index]);
                expectColorEq(loaded.colors()[index], source.colors()[index]);
        }
}

/**
 * @brief 验证 CSV 与 ASCII PCD 风格文本可由 XYZ 读取器兼容加载。
 */
TEST_F(PointCloudIoTest, XyzReaderSupportsCsvAndAsciiPcdStyle)
{
    const std::filesystem::path csvPath = _tempDir / "sample.csv";
    {
        std::ofstream stream(csvPath);
        ASSERT_TRUE(stream.is_open());
        stream << "0.0,1.0,2.0,255,0,0\n";
        stream << "3.0,4.0,5.0,0,255,0\n";
    }

    PointCloud csvLoaded;
    PointCloudIOResult csvResult;
    ASSERT_TRUE(readPointCloud(csvPath.string(), &csvLoaded, {}, &csvResult))
        << csvResult.errorMessage;
    ASSERT_EQ(csvLoaded.size(), 2U);
    ASSERT_TRUE(csvLoaded.hasColors());
    expectPointNear(csvLoaded.positions()[0], Point3f{0.0f, 1.0f, 2.0f});
    expectColorEq(csvLoaded.colors()[1], ColorRGBA{0, 255, 0, 255});

    const std::filesystem::path pcdPath = _tempDir / "sample_ascii.pcd";
    {
        std::ofstream stream(pcdPath);
        ASSERT_TRUE(stream.is_open());
        stream << "# .PCD v0.7 - Point Cloud Data file format\n";
        stream << "VERSION 0.7\n";
        stream << "FIELDS x y z rgb\n";
        stream << "SIZE 4 4 4 4\n";
        stream << "TYPE F F F F\n";
        stream << "COUNT 1 1 1 1\n";
        stream << "WIDTH 2\n";
        stream << "HEIGHT 1\n";
        stream << "POINTS 2\n";
        stream << "DATA ascii\n";
        stream << "1.0 2.0 3.0 255 255 0\n";
        stream << "4.0 5.0 6.0 0 0 255\n";
    }

    PointCloud pcdLoaded;
    PointCloudIOResult pcdResult;
    ASSERT_TRUE(readPointCloud(pcdPath.string(), &pcdLoaded, {}, &pcdResult))
        << pcdResult.errorMessage;
    ASSERT_EQ(pcdLoaded.size(), 2U);
    expectPointNear(pcdLoaded.positions()[0], Point3f{1.0f, 2.0f, 3.0f});
    expectPointNear(pcdLoaded.positions()[1], Point3f{4.0f, 5.0f, 6.0f});
}

/**
 * @brief 验证可从 BA 运行摘要 JSON 中恢复点云与摄影测量属性。
 */
TEST_F(PointCloudIoTest, ReadBaRunJsonPointCloud)
{
        const std::filesystem::path jsonPath = _tempDir / "ba_run_summary.json";
        std::ofstream stream(jsonPath);
        ASSERT_TRUE(stream.is_open());
        stream << R"({
    "output_dir": "/tmp/ba_case",
    "points": [
        {
            "index": 7,
            "valid": true,
            "converged": true,
            "track_len": 5,
            "rms_after": 0.12,
            "point_xyz": [1.0, 2.0, 3.0]
        },
        {
            "index": 8,
            "valid": false,
            "converged": false,
            "track_len": 2,
            "rms_after": 1.5,
            "point_xyz": [-4.0, 0.5, 9.0]
        }
    ]
})";
        stream.close();

        PointCloud loaded;
        PointCloudIOResult readResult;
        ASSERT_TRUE(readBaRunJsonPointCloud(jsonPath.string(), &loaded, &readResult))
                << readResult.errorMessage;

        ASSERT_EQ(loaded.size(), 2U);
        ASSERT_TRUE(loaded.hasPhotogrammetryAttributes());
        expectPointNear(loaded.positions()[0], Point3f{1.0f, 2.0f, 3.0f});
        expectPointNear(loaded.positions()[1], Point3f{-4.0f, 0.5f, 9.0f});
        EXPECT_EQ(loaded.photogrammetryAttributes()[0].pointId, 7);
        EXPECT_EQ(loaded.photogrammetryAttributes()[0].trackLength, 5);
        EXPECT_NEAR(loaded.photogrammetryAttributes()[0].reprojectionError, 0.12f, 1e-6f);
        EXPECT_TRUE(loaded.photogrammetryAttributes()[0].isValid);
        EXPECT_EQ(loaded.photogrammetryAttributes()[1].pointId, 8);
        EXPECT_EQ(loaded.photogrammetryAttributes()[1].trackLength, 2);
        EXPECT_NEAR(loaded.photogrammetryAttributes()[1].reprojectionError, 1.5f, 1e-6f);
        EXPECT_FALSE(loaded.photogrammetryAttributes()[1].isValid);
        EXPECT_EQ(loaded.metadata().coordinateFrame, PointCloudCoordinateFrame::World);
        EXPECT_TRUE(loaded.metadata().isRegistered);
}
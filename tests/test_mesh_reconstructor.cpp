#include <gtest/gtest.h>

#include "DepthMapMeshBuilder.h"
#include "ModelWorkflowService.h"
#include "SurfaceReconstructor.h"
#include "SurfaceReconstructorPostprocess.h"
#include "TextureMapper.h"

#include <plapoint/filters/preprocessing.h>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{

std::filesystem::path writeNoNormalsPointCloud(const std::filesystem::path &root)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "dense_no_normals.ply";

    constexpr int N = 24;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(N * N, 3);
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            const int row = y * N + x;
            const float fx = (static_cast<float>(x) - N * 0.5f) / static_cast<float>(N);
            const float fy = (static_cast<float>(y) - N * 0.5f) / static_cast<float>(N);
            points(row, 0) = fx;
            points(row, 1) = fy;
            points(row, 2) = 0.08f * std::sin(fx * 7.0f) + 0.05f * std::cos(fy * 9.0f);
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

std::filesystem::path writeSpherePointCloudWithNormals(const std::filesystem::path &root,
                                                       bool alternateNormalDirection = false,
                                                       bool injectInvalidNormals = false,
                                                       bool varyColors = false)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "sphere_with_normals.ply";

    constexpr int rings = 16;
    constexpr int segments = 16;
    constexpr int pointCount = rings * segments;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(pointCount, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(pointCount, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(pointCount, 3);

    int row = 0;
    for (int ring = 0; ring < rings; ++ring)
    {
        const float phi = static_cast<float>(ring + 1) * static_cast<float>(M_PI) /
                          static_cast<float>(rings + 1);
        for (int segment = 0; segment < segments; ++segment)
        {
            const float theta = static_cast<float>(segment) * 2.0f * static_cast<float>(M_PI) /
                                static_cast<float>(segments);
            const float nx = std::sin(phi) * std::cos(theta);
            const float ny = std::sin(phi) * std::sin(theta);
            const float nz = std::cos(phi);
            points(row, 0) = 2.0f * nx;
            points(row, 1) = 2.0f * ny;
            points(row, 2) = 2.0f * nz;
            const float normal_sign = alternateNormalDirection && (row % 2 != 0) ? -1.0f : 1.0f;
            normals(row, 0) = normal_sign * nx;
            normals(row, 1) = normal_sign * ny;
            normals(row, 2) = normal_sign * nz;
            if (varyColors && nx < 0.0f)
            {
                colors(row, 0) = 240;
                colors(row, 1) = 40;
                colors(row, 2) = 30;
            }
            else if (varyColors)
            {
                colors(row, 0) = 30;
                colors(row, 1) = 60;
                colors(row, 2) = 240;
            }
            else
            {
                colors(row, 0) = 180;
                colors(row, 1) = 190;
                colors(row, 2) = 210;
            }
            ++row;
        }
    }

    if (injectInvalidNormals)
    {
        normals(0, 0) = 0.0f;
        normals(0, 1) = 0.0f;
        normals(0, 2) = 0.0f;
        normals(1, 0) = std::numeric_limits<float>::quiet_NaN();
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setNormals(std::move(normals));
    cloud.setColors(std::move(colors));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

std::filesystem::path writeDenseGridPointCloud(const std::filesystem::path &root)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "dense_grid.ply";

    constexpr int N = 32;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(N * N, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(N * N, 3);
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            const int row = y * N + x;
            const float fx = static_cast<float>(x) / static_cast<float>(N - 1);
            const float fy = static_cast<float>(y) / static_cast<float>(N - 1);
            points(row, 0) = fx;
            points(row, 1) = fy;
            points(row, 2) = 0.04f * std::sin(fx * 8.0f) + 0.03f * std::cos(fy * 6.0f);
            colors(row, 0) = static_cast<std::uint8_t>(50 + x * 4);
            colors(row, 1) = static_cast<std::uint8_t>(80 + y * 4);
            colors(row, 2) = 160;
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setColors(std::move(colors));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

std::filesystem::path writeFlatGridPointCloudWithSparseVerticalSpikes(const std::filesystem::path &root)
{
    namespace fs = std::filesystem;
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "flat_grid_with_spikes.ply";

    constexpr int N = 24;
    constexpr int baseSamples = 5;
    constexpr int spikeSamples = 1;
    constexpr int perCellSamples = baseSamples + spikeSamples;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(N * N * perCellSamples, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(N * N * perCellSamples, 3);

    int row = 0;
    for (int y = 0; y < N; ++y)
    {
        for (int x = 0; x < N; ++x)
        {
            const float fx = static_cast<float>(x) / static_cast<float>(N - 1);
            const float fy = static_cast<float>(y) / static_cast<float>(N - 1);
            for (int sample = 0; sample < baseSamples; ++sample)
            {
                points(row, 0) = fx + 0.0002f * static_cast<float>(sample);
                points(row, 1) = fy;
                points(row, 2) = 0.0f;
                colors(row, 0) = 80;
                colors(row, 1) = 160;
                colors(row, 2) = 80;
                ++row;
            }
            const bool hasSpike = (x % 6 == 0) && (y % 6 == 0);
            points(row, 0) = fx;
            points(row, 1) = fy;
            points(row, 2) = hasSpike ? 4.0f : 0.0f;
            colors(row, 0) = 220;
            colors(row, 1) = 80;
            colors(row, 2) = 80;
            ++row;
        }
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setColors(std::move(colors));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);
    return plyPath;
}

xjw::mesh::ReconstructionConfig fallbackMeshConfig()
{
    xjw::mesh::ReconstructionConfig config;
    config.forcePoisson = true;
    config.poissonDepth = 12;
    config.resolution = 48;
    config.enableDenoise = false;
    config.enableDownsample = false;
    config.preprocessingDevice = plapoint::ProcessingDevice::CPU;
    config.cleanSmallComponents = false;
    config.smoothIterations = 0;
    config.holeFillPasses = 2;
    return config;
}

} // namespace

TEST(MeshReconstructorTest, UsesStreamingHeightGridForOversizedPly)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_streaming_input_cap_test";
    const fs::path plyPath = writeDenseGridPointCloud(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.forcePoisson = false;
    config.resolution = 32;
    config.maxInputPointsForMeshing = 300;
    config.streamingThreads = 3;
    config.streamingChunkBytes = 256;
    std::vector<std::string> progressMessages;
    config.progressFn = [&](const std::string &stage, float) {
        progressMessages.push_back(stage);
    };

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "streaming_tiled_height_grid");
    EXPECT_GT(mesh.faceCount(), 0);
    const auto streamed = std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &msg) {
        return msg.find("并行分块") != std::string::npos &&
               msg.find("1024") != std::string::npos &&
               msg.find("3") != std::string::npos;
    });
    EXPECT_NE(streamed, progressMessages.end());
    const auto tiled = std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &msg) {
        return msg.find("瓦片") != std::string::npos &&
               msg.find("2x2") != std::string::npos;
    });
    EXPECT_NE(tiled, progressMessages.end());
}

TEST(MeshReconstructorTest, OversizedArbitraryCloudUsesBoundedPoissonSample)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_bounded_poisson_sample_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;
    config.orientNormalsForClosedSurface = true;
    config.maxInputPointsForMeshing = 240;
    std::vector<std::string> progressMessages;
    config.progressFn = [&](const std::string &stage, float) {
        progressMessages.push_back(stage);
    };

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "poisson");
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_NE(std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &message) {
                  return message.find("Poisson") != std::string::npos
                      && message.find("抽样") != std::string::npos;
              }),
              progressMessages.end());
}

TEST(MeshReconstructorTest, StreamingHeightGridSuppressesSparseVerticalSpikes)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_streaming_spike_test";
    const fs::path plyPath = writeFlatGridPointCloudWithSparseVerticalSpikes(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.forcePoisson = false;
    config.resolution = 32;
    config.maxInputPointsForMeshing = 300;
    config.streamingThreads = 2;
    config.streamingChunkBytes = 512;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    ASSERT_FALSE(mesh.vertices.empty());
    EXPECT_EQ(algorithmUsed, "streaming_tiled_height_grid");

    float maxZ = -std::numeric_limits<float>::max();
    for (const auto &vertex : mesh.vertices)
    {
        maxZ = std::max(maxZ, vertex.z);
    }
    EXPECT_LT(maxZ, 0.4f)
        << "Sparse vertical outliers should not become visible terrain spikes.";
}

TEST(MeshReconstructorTest, ForcePoissonUsesInputNormalsWhenPresent)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_normals_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.vertexCount(), 0);
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "poisson");
}

TEST(MeshReconstructorTest, PoissonTransfersSourceColorsToMeshVertices)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_color_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root, false, false, true);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    ASSERT_FALSE(mesh.vertices.empty());
    EXPECT_EQ(algorithmUsed, "poisson");

    int red_dominant = 0;
    int blue_dominant = 0;
    for (const auto &vertex : mesh.vertices)
    {
        red_dominant += vertex.r > vertex.b ? 1 : 0;
        blue_dominant += vertex.b > vertex.r ? 1 : 0;
    }
    EXPECT_GT(red_dominant, 0);
    EXPECT_GT(blue_dominant, 0);
}

TEST(MeshReconstructorTest, PoissonDepthAdaptsToPointCloudDensity)
{
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(50000, 8), 6);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(250000, 8), 7);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(1000000, 8), 8);
    EXPECT_EQ(xjw::mesh::SurfaceReconstructor::recommendedPoissonDepth(50000, 5), 5);
}

TEST(MeshReconstructorTest, PoissonOrientsMixedNormalsForClosedSurface)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_mixed_normals_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root, true);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;
    config.orientNormalsForClosedSurface = true;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "poisson");
}

TEST(MeshReconstructorTest, PoissonRepairsInvalidNormalsBeforeReconstruction)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_invalid_normals_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root, false, true);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.poissonDepth = 4;
    config.allowHeightGridFallback = false;
    config.orientNormalsForClosedSurface = true;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "poisson");
}

TEST(MeshReconstructorTest, SmallComponentCleanupKeepsLargestComponent)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(7);
    mesh.vertices[0].x = 0.0f; mesh.vertices[0].y = 0.0f; mesh.vertices[0].z = 0.0f;
    mesh.vertices[1].x = 1.0f; mesh.vertices[1].y = 0.0f; mesh.vertices[1].z = 0.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 1.0f; mesh.vertices[2].z = 0.0f;
    mesh.vertices[3].x = 1.0f; mesh.vertices[3].y = 1.0f; mesh.vertices[3].z = 0.0f;
    mesh.vertices[4].x = 10.0f; mesh.vertices[4].y = 0.0f; mesh.vertices[4].z = 0.0f;
    mesh.vertices[5].x = 11.0f; mesh.vertices[5].y = 0.0f; mesh.vertices[5].z = 0.0f;
    mesh.vertices[6].x = 10.0f; mesh.vertices[6].y = 1.0f; mesh.vertices[6].z = 0.0f;

    xjw::mesh::Triangle firstA;
    firstA.v[0] = 0; firstA.v[1] = 1; firstA.v[2] = 2;
    xjw::mesh::Triangle secondA;
    secondA.v[0] = 1; secondA.v[1] = 3; secondA.v[2] = 2;
    xjw::mesh::Triangle firstB;
    firstB.v[0] = 4; firstB.v[1] = 5; firstB.v[2] = 6;
    mesh.faces = {firstA, secondA, firstB};

    xjw::mesh::detail::removeSmallConnectedComponents(&mesh, 10);

    EXPECT_EQ(mesh.faceCount(), 2);
    EXPECT_EQ(mesh.vertexCount(), 4);
}

TEST(MeshReconstructorTest, SmallBoundaryHoleFillingClosesTetrahedronOpening)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(4);
    mesh.vertices[0].x = 0.0f; mesh.vertices[0].y = 0.0f; mesh.vertices[0].z = 0.0f;
    mesh.vertices[1].x = 1.0f; mesh.vertices[1].y = 0.0f; mesh.vertices[1].z = 0.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 1.0f; mesh.vertices[2].z = 0.0f;
    mesh.vertices[3].x = 0.0f; mesh.vertices[3].y = 0.0f; mesh.vertices[3].z = 1.0f;
    mesh.faces = {
        xjw::mesh::Triangle{{0, 1, 3}},
        xjw::mesh::Triangle{{1, 2, 3}},
        xjw::mesh::Triangle{{2, 0, 3}},
    };

    const int filled = xjw::mesh::detail::fillSmallBoundaryHoles(&mesh, 8);

    EXPECT_EQ(filled, 1);
    EXPECT_EQ(mesh.vertexCount(), 5);
    EXPECT_EQ(mesh.faceCount(), 6);
}

TEST(MeshPostprocessTest, WeldsCoincidentVerticesBeforeComponentFiltering)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(6);
    mesh.vertices[0].x = 0.0f; mesh.vertices[0].y = 0.0f;
    mesh.vertices[1].x = 1.0f; mesh.vertices[1].y = 0.0f;
    mesh.vertices[2].x = 0.0f; mesh.vertices[2].y = 1.0f;
    mesh.vertices[3].x = 1.0f; mesh.vertices[3].y = 0.0f;
    mesh.vertices[4].x = 1.0f; mesh.vertices[4].y = 1.0f;
    mesh.vertices[5].x = 0.0f; mesh.vertices[5].y = 1.0f;

    xjw::mesh::Triangle first;
    first.v[0] = 0; first.v[1] = 1; first.v[2] = 2;
    xjw::mesh::Triangle second;
    second.v[0] = 3; second.v[1] = 4; second.v[2] = 5;
    mesh.faces = {first, second};

    xjw::mesh::detail::weldCoincidentVertices(&mesh, 1.0e-6f);
    xjw::mesh::detail::removeSmallConnectedComponents(&mesh, 2);

    EXPECT_EQ(mesh.vertexCount(), 4);
    EXPECT_EQ(mesh.faceCount(), 2);
}

TEST(MeshReconstructorTest, WeldedPoissonMeshSurvivesOversizedComponentThreshold)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_tiny_cleanup_test";
    const fs::path plyPath = writeSpherePointCloudWithNormals(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.cleanSmallComponents = true;
    config.minComponentFaces = 100000;
    config.poissonDepth = 4;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "poisson");
    EXPECT_GT(mesh.faceCount(), 100);
}

TEST(MeshReconstructorTest, ForcePoissonFallsBackWhenCloudHasNoNormals)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_no_normals_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root);

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    bool ok = false;
    ASSERT_NO_THROW({
        ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                            fallbackMeshConfig(),
                                                                            mesh,
                                                                            &error,
                                                                            &algorithmUsed);
    });

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.vertexCount(), 0);
    EXPECT_GT(mesh.faceCount(), 0);
    EXPECT_EQ(algorithmUsed, "height_grid");
}

TEST(MeshReconstructorTest, Arbitrary3dRejectsHeightGridFallbackWhenPoissonCannotRun)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_no_arbitrary_fallback_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    config.allowHeightGridFallback = false;

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(mesh.empty());
    EXPECT_TRUE(algorithmUsed.empty());
    EXPECT_NE(error.find("Poisson"), std::string::npos);
    EXPECT_NE(error.find("高度格网"), std::string::npos);
}

TEST(MeshReconstructorTest, PoissonFallbackProgressIncludesFailureReason)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_poisson_reason_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root);

    xjw::mesh::ReconstructionConfig config = fallbackMeshConfig();
    std::vector<std::string> progressMessages;
    config.progressFn = [&](const std::string &stage, float) {
        progressMessages.push_back(stage);
    };

    xjw::mesh::TriMesh mesh;
    std::string error;
    std::string algorithmUsed;
    const bool ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                                   config,
                                                                                   mesh,
                                                                                   &error,
                                                                                   &algorithmUsed);

    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(algorithmUsed, "height_grid");
    const auto it = std::find_if(progressMessages.begin(), progressMessages.end(), [](const std::string &message) {
        return message.find("Poisson 重建失败(") != std::string::npos &&
               message.find("改用高度格网") != std::string::npos;
    });
    EXPECT_NE(it, progressMessages.end());
}

TEST(MeshWorkflowServiceTest, RecordsActualFallbackAlgorithmInPayload)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_workflow_algorithm_test";
    const fs::path plyPath = writeNoNormalsPointCloud(root / "input");

    xjw::mesh::workflow::MeshBuildRequest request;
    request.pointCloudPath = QString::fromStdString(plyPath.string());
    request.outputRoot = QString::fromStdString((root / "model").string());
    request.reconstruction = fallbackMeshConfig();
    request.exportObj = false;

    const auto result = xjw::mesh::workflow::buildMeshAndOptionalTexture(request);
    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    EXPECT_EQ(result.payload.value(QStringLiteral("mesh_algorithm")).toString().toStdString(), "height_grid");
    EXPECT_TRUE(fs::exists(result.payload.value(QStringLiteral("model_ply")).toString().toStdString()));
}

TEST(MeshWorkflowServiceTest, SharedModelEntryMapsSettingsAndBuildsPointCloudSource)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_shared_model_entry_test";
    const fs::path plyPath = writeDenseGridPointCloud(root);

    xjw::mesh::workflow::ModelBuildRequest request;
    request.sourceData = QStringLiteral("point_cloud");
    request.requestedSourcePath = QString::fromStdString(plyPath.string());
    request.sourcePointCloudPath = request.requestedSourcePath;
    request.outputRoot = QString::fromStdString((root / "model").string());
    request.settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    request.settings[QStringLiteral("method")] = QStringLiteral("Height Grid");
    request.settings[QStringLiteral("octreeDepth")] = 7;
    request.settings[QStringLiteral("cleanSmall")] = false;
    request.settings[QStringLiteral("smoothIter")] = 0;
    request.settings[QStringLiteral("depthFiltering")] = QStringLiteral("disabled");
    request.settings[QStringLiteral("qualityProfile")] = QStringLiteral("detail");

    const auto result = xjw::mesh::workflow::buildModel(request);

    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    EXPECT_EQ(result.payload.value(QStringLiteral("source_data")).toString(),
              QStringLiteral("point_cloud"));
    EXPECT_EQ(result.payload.value(QStringLiteral("source_point_cloud_path")).toString(),
              request.sourcePointCloudPath);
    EXPECT_TRUE(fs::exists(result.payload.value(QStringLiteral("mesh_ply")).toString().toStdString()));
}

TEST(MeshWorkflowSettingsTest, HeightFieldSourceDisablesPoissonAndMapsFiltering)
{
    QJsonObject settings;
    settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    settings[QStringLiteral("quality")] = QStringLiteral("ultra");
    settings[QStringLiteral("qualityProfile")] = QStringLiteral("detail");
    settings[QStringLiteral("meshResolution")] = 384;
    settings[QStringLiteral("octreeDepth")] = 11;
    settings[QStringLiteral("targetFaces")] = 0;
    settings[QStringLiteral("interpolation")] = QStringLiteral("disabled");
    settings[QStringLiteral("depthFiltering")] = QStringLiteral("aggressive");

    const auto config = xjw::mesh::workflow::reconstructionConfigFromModelSettings(settings);

    EXPECT_FALSE(config.forcePoisson);
    EXPECT_GE(config.resolution, 384);
    EXPECT_FALSE(config.fillHoles);
    EXPECT_TRUE(config.enableDenoise);
    EXPECT_LE(config.denoiseStdMul, 0.95f);
    EXPECT_EQ(config.simplifyTargetFaces, 0);
}

TEST(MeshWorkflowSettingsTest, ArbitrarySourceKeepsPoissonAndHonorsFaceBudget)
{
    QJsonObject settings;
    settings[QStringLiteral("surface_type")] = QStringLiteral("arbitrary_3d");
    settings[QStringLiteral("quality")] = QStringLiteral("high");
    settings[QStringLiteral("qualityProfile")] = QStringLiteral("detail");
    settings[QStringLiteral("meshResolution")] = 320;
    settings[QStringLiteral("octreeDepth")] = 10;
    settings[QStringLiteral("targetFaces")] = 240000;
    settings[QStringLiteral("interpolation")] = QStringLiteral("extrapolated");
    settings[QStringLiteral("depthFiltering")] = QStringLiteral("mild");

    const auto config = xjw::mesh::workflow::reconstructionConfigFromModelSettings(settings);

    EXPECT_TRUE(config.forcePoisson);
    EXPECT_FALSE(config.allowHeightGridFallback);
    EXPECT_TRUE(config.orientNormalsForClosedSurface);
    EXPECT_GE(config.poissonDepth, 10);
    EXPECT_TRUE(config.fillHoles);
    EXPECT_GE(config.holeFillPasses, 16);
    EXPECT_EQ(config.simplifyTargetFaces, 240000);
    EXPECT_EQ(config.maxInputPointsForMeshing, 400000);
    EXPECT_FALSE(config.enableDownsample);
    EXPECT_GE(config.kNormals, 18);
    EXPECT_GE(config.smoothIterations, 4);
}

TEST(MeshWorkflowSettingsTest, DepthMapMeshRequestPreservesSourceAndSettings)
{
    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QStringLiteral("E:/tmp/mvs_output");
    request.reusableDenseCloudPath = QStringLiteral("E:/tmp/mvs_output/dense_cloud.ply");
    request.outputRoot = QStringLiteral("E:/tmp/model");
    request.settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    request.settings[QStringLiteral("quality")] = QStringLiteral("high");
    request.settings[QStringLiteral("reuseDepthMaps")] = true;

    EXPECT_EQ(request.depthMapSourcePath, QStringLiteral("E:/tmp/mvs_output"));
    EXPECT_EQ(request.reusableDenseCloudPath, QStringLiteral("E:/tmp/mvs_output/dense_cloud.ply"));
    EXPECT_EQ(request.outputRoot, QStringLiteral("E:/tmp/model"));
    EXPECT_TRUE(request.settings.value(QStringLiteral("reuseDepthMaps")).toBool());
}

TEST(DepthMapMeshBuilderTest, DiscoversDepthFramesFromOutputDirectory)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_discovery_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "depth_001.bin").put('\0');
    std::ofstream(root / "depth_001_conf.bin").put('\0');
    std::ofstream(root / "depth_001.png").put('\0');
    std::ofstream(root / "depth_001_mask.png").put('\0');
    std::ofstream(root / "depth_002.bin").put('\0');

    const auto frames =
        xjw::mesh::DepthMapMeshBuilder::discoverDepthFrames(QString::fromStdString(root.string()));

    ASSERT_EQ(frames.size(), 2);
    EXPECT_TRUE(frames.at(0).depthPath.endsWith(QStringLiteral("depth_001.bin")));
    EXPECT_TRUE(frames.at(0).confidencePath.endsWith(QStringLiteral("depth_001_conf.bin")));
    EXPECT_TRUE(frames.at(0).previewPath.endsWith(QStringLiteral("depth_001.png")));
    EXPECT_TRUE(frames.at(0).validMaskPath.endsWith(QStringLiteral("depth_001_mask.png")));
    EXPECT_TRUE(frames.at(1).depthPath.endsWith(QStringLiteral("depth_002.bin")));
}

TEST(DepthMapMeshBuilderTest, UsesExistingDenseCloudWhenPresent)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_existing_dense_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path densePath = writeDenseGridPointCloud(root);
    fs::rename(densePath, root / "dense_cloud.ply");

    QString error;
    const QString resolved = xjw::mesh::DepthMapMeshBuilder::resolveReusableDenseCloud(
        QString::fromStdString(root.string()),
        &error);

    EXPECT_TRUE(error.isEmpty());
    EXPECT_TRUE(resolved.endsWith(QStringLiteral("dense_cloud.ply")));
}

TEST(DepthMapMeshBuilderTest, ReportsActionableErrorWhenDepthFramesCannotBeFused)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_no_dense_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "depth_001.bin").put('\0');

    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QString::fromStdString(root.string());
    request.outputRoot = QString::fromStdString(root.string());
    request.reconstruction = fallbackMeshConfig();

    const auto result = xjw::mesh::workflow::buildMeshFromDepthMaps(request);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("缺少深度图 metadata")));
}

TEST(TextureMapperTest, ReadsPlyMeshFacesForTextureMapping)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_texture_mapper_ply_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path plyPath = root / "mesh_with_faces.ply";

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points(0, 0) = 0.0f; points(0, 1) = 0.0f; points(0, 2) = 0.0f;
    points(1, 0) = 1.0f; points(1, 1) = 0.0f; points(1, 2) = 0.0f;
    points(2, 0) = 0.0f; points(2, 1) = 1.0f; points(2, 2) = 0.0f;
    points(3, 0) = 1.0f; points(3, 1) = 1.0f; points(3, 2) = 0.0f;

    plapoint::PointCloud<float, plamatrix::Device::CPU> meshCloud(std::move(points));

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(4, 3);
    colors(0, 0) = 255; colors(0, 1) = 0;   colors(0, 2) = 0;
    colors(1, 0) = 0;   colors(1, 1) = 255; colors(1, 2) = 0;
    colors(2, 0) = 0;   colors(2, 1) = 0;   colors(2, 2) = 255;
    colors(3, 0) = 255; colors(3, 1) = 255; colors(3, 2) = 255;
    meshCloud.setColors(std::move(colors));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces(0, 0) = 0; faces(0, 1) = 1; faces(0, 2) = 2;
    faces(1, 0) = 1; faces(1, 1) = 3; faces(1, 2) = 2;
    meshCloud.setFaces(std::move(faces));
    plapoint::io::writePly<float>(plyPath.string(), meshCloud, plapoint::io::PlyFormat::BinaryLE);

    xjw::mesh::TextureMappingConfig config;
    config.textureSize = 512;
    xjw::mesh::TextureMappingResult result;
    std::string error;
    ASSERT_TRUE(xjw::mesh::TextureMapper::generateTexturedModelFromMeshFile(plyPath.string(),
                                                                            root.string(),
                                                                            config,
                                                                            &result,
                                                                            &error))
        << error;
    EXPECT_TRUE(fs::exists(result.modelObjPath));
    EXPECT_TRUE(fs::exists(result.modelMtlPath));
    EXPECT_TRUE(fs::exists(result.texturePngPath));

    std::ifstream objFile(result.modelObjPath);
    std::stringstream objBuffer;
    objBuffer << objFile.rdbuf();
    EXPECT_NE(objBuffer.str().find("usemtl material0"), std::string::npos);

    std::ifstream mtlFile(result.modelMtlPath);
    std::stringstream mtlBuffer;
    mtlBuffer << mtlFile.rdbuf();
    EXPECT_NE(mtlBuffer.str().find("map_Kd textures/model_texture.png"), std::string::npos);
}

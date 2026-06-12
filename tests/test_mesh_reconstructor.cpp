#include <gtest/gtest.h>

#include "SurfaceReconstructor.h"
#include "TextureMapper.h"

#include <plapoint/filters/preprocessing.h>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

TEST(MeshReconstructorTest, ForcePoissonFallsBackWhenCloudHasNoNormals)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mesh_no_normals_test";
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

    xjw::mesh::TriMesh mesh;
    std::string error;
    bool ok = false;
    ASSERT_NO_THROW({
        ok = xjw::mesh::SurfaceReconstructor::reconstructFromPointCloudFile(plyPath.string(),
                                                                            config,
                                                                            mesh,
                                                                            &error);
    });

    ASSERT_TRUE(ok) << error;
    EXPECT_GT(mesh.vertexCount(), 0);
    EXPECT_GT(mesh.faceCount(), 0);
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

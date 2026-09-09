#include <gtest/gtest.h>

#include "RecoveredModelInput.h"
#include "RecoveredDepthScene.h"
#include "RecoveredModelBuilder.h"
#include "DepthFrameUtils.h"
#include "recovered_model/VertexColorOptions.h"
#include "recovered_model/RecoveredModelColorizer.h"
#include "ModelWorkflowService.h"
#include "io/PathIO.h"
#include "MeshTypes.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QTemporaryDir>

#include <bit>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#if defined(MVS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace
{
    class RecoveredModelInputTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_TRUE(QDir().mkpath(QString::fromUtf8(RECOVERED_MODEL_TEST_ROOT)));
            _directory =
                std::make_unique<QTemporaryDir>(QString::fromUtf8(RECOVERED_MODEL_TEST_ROOT) + "/input-XXXXXX");
            ASSERT_TRUE(_directory->isValid());
            _path = _directory->filePath("input");
            _scene.region.specified = true;
            _scene.region.center = {0.25, -0.5, 4.0};
            _scene.region.size = {8, 10, 2};
            _scene.cameras.resize(2);
            _depth.cameras.resize(2);
            for (std::size_t i = 0; i < 2; ++i)
            {
                auto& camera = _scene.cameras[i];
                camera.index = i;
                camera.aligned = true;
                camera.image.width = 32;
                camera.image.height = 32;
                camera.model.f = 32;
                camera.model.cx = 16;
                camera.model.cy = 16;
                camera.model.k1 = 0.00123456789;
                camera.model.b1 = 0.75;
                camera.pose.translation = {static_cast<double>(i), 0.25, -0.125};
                camera.center = metalign::camera_center(camera.pose);
                camera.pose.center = camera.center;
                auto& output = _depth.cameras[i];
                output.patchmatch.camera_index = i;
                output.voting.depth_after_components = {std::vector<float>(64, 4.0F),
                                                        std::vector<float>(16, std::bit_cast<float>(0x40800001U)),
                                                        std::vector<float>(4, 4.25F)};
                output.voting.depth_after_components[0][3] = 0;
                output.public_depth.assign(64, 99.0F);
            }
        }
        std::unique_ptr<QTemporaryDir> _directory;
        QString _path;
        metmodel::Scene _scene;
        metmodel::RecoveredPatchMatchD4SceneOutput _depth;
    };

    TEST_F(RecoveredModelInputTest, PublicD4CameraUsesReferenceCalibrationWithoutHalfPixelShift)
    {
        xjw::FramePinholeCamera original;
        original.setIntrinsics(402, 400, 131.5, 125.25);
        original.setDistortion({0.1, 0.2, 0.3, 0.01, 0.02});
        original.setCameraCenter({1, 2, 3});
        const auto output = xjw::mvs::recoveredPublicD4Camera(original);
        const auto intrinsics = output.intrinsics();
        EXPECT_DOUBLE_EQ(intrinsics.focalX, 100);
        EXPECT_DOUBLE_EQ(intrinsics.focalY, 100);
        EXPECT_DOUBLE_EQ(intrinsics.principalX, 32.875);
        EXPECT_DOUBLE_EQ(intrinsics.principalY, 31.3125);
        EXPECT_EQ(output.cameraCenter(), original.cameraCenter());
        const auto distortion = output.distortion();
        EXPECT_EQ(distortion.radialK1, 0);
        EXPECT_EQ(distortion.radialK2, 0);
        EXPECT_EQ(distortion.radialK3, 0);
        EXPECT_EQ(distortion.tangentialP1, 0);
        EXPECT_EQ(distortion.tangentialP2, 0);
    }

    TEST_F(RecoveredModelInputTest, RoundTripPreservesVotedLevelsAndBrownCamera)
    {
        _scene.cameras[1].path = xjw::common::io::toFilesystemPath(_directory->filePath("原始影像.jpg"));
        ASSERT_NO_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth));
        metmodel::Scene loaded;
        metmodel::RecoveredPatchMatchD4SceneOutput depth;
        ASSERT_NO_THROW(xjw::mvs::readRecoveredModelInput(_path, loaded, depth));
        ASSERT_EQ(loaded.cameras.size(), 2);
        EXPECT_EQ(loaded.cameras[1].model.k1, _scene.cameras[1].model.k1);
        EXPECT_EQ(loaded.cameras[1].model.b1, _scene.cameras[1].model.b1);
        EXPECT_EQ(loaded.cameras[1].path, _scene.cameras[1].path);
        EXPECT_EQ(loaded.cameras[1].pose.translation.x, 1.0);
        EXPECT_EQ(loaded.region.center.y, -0.5);
        for (std::size_t i = 0; i < 2; ++i)
        {
            EXPECT_EQ(depth.cameras[i].voting.depth_after_components, _depth.cameras[i].voting.depth_after_components);
            EXPECT_TRUE(depth.cameras[i].public_depth.empty());
        }
    }

    TEST_F(RecoveredModelInputTest, RejectsMissingLevelAndDoesNotPublish)
    {
        _depth.cameras[1].voting.depth_after_components[2].clear();
        EXPECT_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth), std::runtime_error);
        EXPECT_FALSE(QFileInfo::exists(_path));
    }

    TEST_F(RecoveredModelInputTest, RejectsOverwrite)
    {
        ASSERT_NO_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth));
        EXPECT_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth), std::runtime_error);
    }

    TEST_F(RecoveredModelInputTest, ReplacesExistingInputAfterCompleteStaging)
    {
        ASSERT_NO_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth));
        _depth.cameras[0].voting.depth_after_components[0][1] = 7.25F;
        ASSERT_NO_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth, true));

        metmodel::Scene loaded;
        metmodel::RecoveredPatchMatchD4SceneOutput loaded_depth;
        ASSERT_NO_THROW(xjw::mvs::readRecoveredModelInput(_path, loaded, loaded_depth));
        EXPECT_EQ(loaded_depth.cameras[0].voting.depth_after_components[0][1], 7.25F);
        const QDir parent(_directory->path());
        EXPECT_TRUE(parent
                        .entryList({QStringLiteral("input.staging-*"), QStringLiteral("input.backup-*")},
                                   QDir::Dirs | QDir::NoDotAndDotDot)
                        .isEmpty());
    }

    TEST_F(RecoveredModelInputTest, FailedReplacementPreservesPublishedInput)
    {
        ASSERT_NO_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth));
        const auto original = _depth.cameras[0].voting.depth_after_components[0];
        _depth.cameras[1].voting.depth_after_components[2].clear();
        EXPECT_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth, true), std::runtime_error);

        metmodel::Scene loaded;
        metmodel::RecoveredPatchMatchD4SceneOutput loaded_depth;
        ASSERT_NO_THROW(xjw::mvs::readRecoveredModelInput(_path, loaded, loaded_depth));
        EXPECT_EQ(loaded_depth.cameras[0].voting.depth_after_components[0], original);
    }

    TEST(RecoveredModelSupportLevelsTest, DefaultScheduleEndsAtActualTreeMaximum)
    {
        EXPECT_EQ(xjw::mesh::planRecoveredSupportLevels(4), (std::vector<std::uint32_t>{4}));
        EXPECT_EQ(xjw::mesh::planRecoveredSupportLevels(6), (std::vector<std::uint32_t>{6}));
        EXPECT_EQ(xjw::mesh::planRecoveredSupportLevels(8), (std::vector<std::uint32_t>{6, 8}));
        EXPECT_EQ(xjw::mesh::planRecoveredSupportLevels(9), (std::vector<std::uint32_t>{5, 7, 9}));
        EXPECT_EQ(xjw::mesh::planRecoveredSupportLevels(12), (std::vector<std::uint32_t>{6, 8, 10, 12}));
    }

    TEST(RecoveredModelSupportLevelsTest, ExplicitScheduleIsValidatedBeforeGpuWork)
    {
        EXPECT_EQ(xjw::mesh::planRecoveredSupportLevels(8, QJsonArray{6, 8}), (std::vector<std::uint32_t>{6, 8}));
        EXPECT_THROW(xjw::mesh::planRecoveredSupportLevels(8, QJsonArray{6}), std::invalid_argument);
        EXPECT_THROW(xjw::mesh::planRecoveredSupportLevels(8, QJsonArray{5, 8}), std::invalid_argument);
        EXPECT_THROW(xjw::mesh::planRecoveredSupportLevels(8, QStringLiteral("6,8")), std::invalid_argument);
    }

    TEST_F(RecoveredModelInputTest, RejectsCorruptBytesWithoutPublishingPartialResult)
    {
        ASSERT_NO_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth));
        QFile file(QDir(_path).filePath("camera_1_d8.bin"));
        ASSERT_TRUE(file.open(QIODevice::ReadWrite));
        ASSERT_EQ(file.write("X", 1), 1);
        file.close();
        metmodel::Scene loaded;
        metmodel::RecoveredPatchMatchD4SceneOutput depth;
        EXPECT_THROW(xjw::mvs::readRecoveredModelInput(_path, loaded, depth), std::runtime_error);
        EXPECT_TRUE(loaded.cameras.empty());
        EXPECT_TRUE(depth.cameras.empty());
    }

    TEST_F(RecoveredModelInputTest, RejectsTruncationAndNonfiniteSamples)
    {
        _depth.cameras[0].voting.depth_after_components[0][0] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth), std::runtime_error);
        _depth.cameras[0].voting.depth_after_components[0][0] = 1;
        ASSERT_NO_THROW(xjw::mvs::writeRecoveredModelInput(_path, _scene, _depth));
        QFile file(QDir(_path).filePath("camera_0_d4.bin"));
        ASSERT_TRUE(file.resize(8));
        metmodel::Scene loaded;
        metmodel::RecoveredPatchMatchD4SceneOutput depth;
        EXPECT_THROW(xjw::mvs::readRecoveredModelInput(_path, loaded, depth), std::runtime_error);
    }

    TEST_F(RecoveredModelInputTest, ModelRejectsCancelledRequestBeforeLoading)
    {
        EXPECT_THROW(xjw::mesh::buildRecoveredModel(_path, {}, 100, [] { return true; }, {}), std::runtime_error);
        EXPECT_FALSE(QFileInfo::exists(_path));
    }

    TEST_F(RecoveredModelInputTest, RejectsUnsupportedModelOptionsBeforeGpuWork)
    {
        EXPECT_THROW(xjw::mesh::buildRecoveredModel(_path, {{"compute_mode", "cpu"}}, 100, {}, {}), std::runtime_error);
        EXPECT_THROW(xjw::mesh::buildRecoveredModel(_path, {{"interpolation", "disabled"}}, 100, {}, {}),
                     std::runtime_error);
        EXPECT_THROW(xjw::mesh::buildRecoveredModel(_path, {{"strictVolumetricMasks", true}}, 100, {}, {}),
                     std::runtime_error);
        EXPECT_THROW(xjw::mesh::buildRecoveredModel(_path, {{"splitIntoBlocks", true}}, 100, {}, {}),
                     std::runtime_error);
    }

    TEST_F(RecoveredModelInputTest, PlyRoundTripPreservesConfidence)
    {
        xjw::mesh::TriMesh mesh;
        mesh.hasVertexConfidence = true;
        mesh.vertices.resize(3);
        mesh.vertices[0].confidence = 12.5F;
        mesh.vertices[1].x = 1;
        mesh.vertices[1].confidence = 3.25F;
        mesh.vertices[2].y = 1;
        mesh.vertices[2].confidence = 0;
        mesh.faces.push_back({{0, 1, 2}});
        const auto path = _directory->filePath("confidence.ply").toStdString();
        std::string error;
        ASSERT_TRUE(mesh.savePLY(path, &error)) << error;
        xjw::mesh::TriMesh loaded;
        ASSERT_TRUE(xjw::mesh::TriMesh::loadPLY(path, &loaded, &error)) << error;
        ASSERT_TRUE(loaded.hasVertexConfidence);
        ASSERT_EQ(loaded.vertices.size(), 3);
        EXPECT_FLOAT_EQ(loaded.vertices[0].confidence, 12.5F);
        EXPECT_FLOAT_EQ(loaded.vertices[1].confidence, 3.25F);
        EXPECT_FLOAT_EQ(loaded.vertices[2].confidence, 0);
    }

    TEST_F(RecoveredModelInputTest, ReferenceRgbLoaderPreservesChannels)
    {
        const auto color_path = _directory->filePath("color.png");
        ASSERT_TRUE(cv::imwrite(color_path.toStdString(), cv::Mat(8, 8, CV_8UC3, cv::Scalar(30, 60, 90))));
        const auto image = metalign::load_rgb_image(xjw::common::io::toFilesystemPath(color_path));
        ASSERT_EQ(image.rgb.size(), 8 * 8 * 3);
        EXPECT_EQ(image.rgb[0], 90);
        EXPECT_EQ(image.rgb[1], 60);
        EXPECT_EQ(image.rgb[2], 30);
        EXPECT_TRUE(image.gray.empty());
        EXPECT_THROW(metalign::load_rgb_image("missing-recovered-image.jpg"), std::runtime_error);
    }

    TEST_F(RecoveredModelInputTest, ReferencePlyPreservesDoubleCoordinatesAndSchema)
    {
        xjw::mesh::RecoveredModelResult result;
        result.mesh.vertices.resize(3);
        result.mesh.faces.push_back({{0, 1, 2}});
        result.precisePositions = {{{1.0000000001, 0, 0}}, {{2, 0, 0}}, {{1, 1, 0}}};
        result.mesh.vertices[0].confidence = 3.25F;
        const auto path = _directory->filePath("double.ply");
        std::string error;
        ASSERT_TRUE(xjw::mesh::writeRecoveredModelPly(result, path, &error)) << error;
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::ReadOnly));
        const auto bytes = file.readAll();
        EXPECT_TRUE(bytes.contains("property double x\n"));
        EXPECT_TRUE(bytes.contains("1.0000000001 0 0"));
        EXPECT_TRUE(bytes.contains("property float confidence\n"));
        EXPECT_FALSE(bytes.contains("property float nx"));
        file.close();
        result.mesh.faces[0].v[0] = 7;
        EXPECT_FALSE(xjw::mesh::writeRecoveredModelPly(result, path, &error));
        ASSERT_TRUE(file.open(QIODevice::ReadOnly));
        EXPECT_EQ(file.readAll(), bytes);
    }

    TEST_F(RecoveredModelInputTest, ReferenceModelRejectsImplicitLegacyTextureFallback)
    {
        xjw::mesh::workflow::DepthMapMeshBuildRequest request;
        request.depthMapSourcePath = _directory->path();
        request.settings = {{"reconstruction_mode", "recovered_ooc"}};
        request.exportObj = true;
        const auto result = xjw::mesh::workflow::buildMeshFromDepthMaps(request);
        EXPECT_FALSE(result.ok);
        EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("UV/纹理")));
    }

#if defined(MVS_ENABLE_CUDA)
    TEST_F(RecoveredModelInputTest, VulkanColorsSmallRgbFrameUsingEmbeddedShaders)
    {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "A CUDA/Vulkan device is required for the recovered color integration test";
        const auto image_path = _directory->filePath("原始颜色.png");
        std::vector<uchar> encoded;
        ASSERT_TRUE(cv::imencode(".png", cv::Mat(32, 32, CV_8UC3, cv::Scalar(32, 64, 96)), encoded));
        QFile file(image_path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_EQ(file.write(reinterpret_cast<const char*>(encoded.data()), static_cast<qint64>(encoded.size())),
                  static_cast<qint64>(encoded.size()));
        file.close();
        _scene.cameras.resize(1);
        auto& camera = _scene.cameras[0];
        camera.pose = {};
        camera.model.k1 = 0;
        camera.model.b1 = 0;
        camera.center = {};
        metmodel::Mesh mesh;
        // Interior triangles are required: boundary-only vertices can all be
        // rejected by the reference half-resolution visibility raster.
        for (std::size_t y = 0; y < 5; ++y)
            for (std::size_t x = 0; x < 5; ++x)
            {
                metmodel::Vertex vertex;
                vertex.position = {(static_cast<double>(x) - 2) * 0.3,
                                   (static_cast<double>(y) - 2) * 0.3,
                                   4.0 + static_cast<double>(x) * 0.005};
                mesh.vertices.push_back(vertex);
                if (x < 4 && y < 4)
                {
                    const auto i = y * 5 + x;
                    mesh.faces.push_back({{i, i + 5, i + 1}});
                    mesh.faces.push_back({{i + 1, i + 5, i + 6}});
                }
            }
        const QJsonObject settings{{"recovered_source_images", QJsonArray{image_path}}};
        const auto result = xjw::mesh::colorizeRecoveredModel(mesh, _scene, settings, 0, {}, {});
        EXPECT_EQ(result.value("color_source_view_count").toInteger(), 1);
        EXPECT_GT(result.value("reliably_colored_vertex_count").toInteger(), 0);
        EXPECT_EQ(camera.model.cx_offset, 0);
        EXPECT_EQ(camera.model.cy_offset, 0);
        for (const auto& vertex : mesh.vertices)
        {
            EXPECT_NEAR(vertex.color[0], 96, 1);
            EXPECT_NEAR(vertex.color[1], 64, 1);
            EXPECT_NEAR(vertex.color[2], 32, 1);
        }
    }

    TEST_F(RecoveredModelInputTest, VulkanRejectsCancellationAndMalformedImagesBeforeDeviceWork)
    {
        metmodel::Mesh mesh;
        metmodel::RecoveredVertexColorVulkanOptions options;
        options.isCancelled = [] { return true; };
        EXPECT_THROW(metmodel::colorize_mesh_recovered_vulkan(mesh, {}, {}, options), std::runtime_error);
        options.isCancelled = {};
        mesh.vertices.resize(3);
        mesh.faces.resize(1);
        _scene.cameras[0].image.width = 31;
        EXPECT_THROW(metmodel::colorize_mesh_recovered_vulkan(
                         mesh, _scene.cameras, xjw::common::io::toFilesystemPath(_directory->path()), options),
                     std::invalid_argument);
    }
#endif
} // namespace

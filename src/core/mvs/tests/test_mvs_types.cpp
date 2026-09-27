// ============================================================
// test_mvs_types.cpp — MVS 类型与参数配置单元测试
//
// 测试内容：
//   1. PatchMatchConfig 默认值验证（优化后的参数）
//   2. FusionConfig 默认值验证
//   3. PlaCamera 正深度归一化与投影
//   4. PlaCamera 投影-反投影一致性
//   5. 深度图后处理参数验证
// ============================================================

#include <gtest/gtest.h>
#include "MvsQualityReport.h"
#include "MvsTypes.h"
#include "MvsViewSelection.h"
#include <placamera/tsai.h>

#include <QJsonObject>

#include <algorithm>
#include <cmath>
#include <array>
#include <atomic>
#include <memory>
#include <string>

using namespace xjw::mvs;

namespace
{

    placamera::FramePinholeModel makeCamera(double fu,
                                            double fv,
                                            double cu,
                                            double cv,
                                            int uDir,
                                            int vDir,
                                            const double r_wc[9],
                                            const double center[3],
                                            bool depthAxisFlipped)
    {
        static std::atomic_size_t next_camera_id{0};
        const std::string id = std::to_string(next_camera_id.fetch_add(1));
        const placamera::FrameId frame("mvs-selection-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("mvs-selection-definition-" + id),
                                                      placamera::FrameIntrinsics{fu, fv, cu, cv, 1.0, uDir, vDir},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame,
                                                      depthAxisFlipped);
        return placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("mvs-selection-camera-" + id),
            placamera::ImageId("mvs-selection-image-" + id),
            definition,
            placamera::ImageSize{std::max(100, static_cast<int>(cu * 2.0)), std::max(100, static_cast<int>(cv * 2.0))},
            placamera::Pose::create(frame,
                                    {center[0], center[1], center[2]},
                                    {r_wc[0], r_wc[1], r_wc[2], r_wc[3], r_wc[4], r_wc[5], r_wc[6], r_wc[7], r_wc[8]}));
    }

    std::shared_ptr<const placamera::FramePinholeModel> makeViewCamera(double fu,
                                                                       double fv,
                                                                       double cu,
                                                                       double cv,
                                                                       int uDir,
                                                                       int vDir,
                                                                       const double r_wc[9],
                                                                       const double center[3],
                                                                       bool depthAxisFlipped)
    {
        return std::make_shared<const placamera::FramePinholeModel>(
            makeCamera(fu, fv, cu, cv, uDir, vDir, r_wc, center, depthAxisFlipped));
    }

} // namespace

// ─── PatchMatchConfig 参数验证 ─────────────────────────────────

TEST(PatchMatchConfigTest, DefaultParametersOptimized)
{
    PatchMatchConfig cfg;

    // 生产点云默认应偏质量，避免把整幅低置信深度图直接用于融合。
    EXPECT_EQ(cfg.patchHalf, 7) << "patchHalf should be 7 (15x15 window) for robust matching";
    EXPECT_EQ(cfg.numIterations, 16);
    EXPECT_GE(cfg.numSourceViews, 5) << "Aerial production MVS should use enough source views for consensus";
    EXPECT_FLOAT_EQ(cfg.confidenceThresh, 0.60f)
        << "Production PatchMatch confidence threshold should reject low-confidence full-frame depths";
    EXPECT_EQ(cfg.backend, PatchMatchBackend::Auto);
    EXPECT_FALSE(cfg.cudaFallbackToCpu) << "CUDA failures must remain visible after the run selects CUDA";
    EXPECT_FALSE(cfg.openClFallbackToCpu)
        << "OpenCL failures must remain visible instead of running a GPU-tagged CPU fallback";
    EXPECT_EQ(cfg.cudaDeviceIndex, -1);
    EXPECT_EQ(cfg.downsampleFactor, 2);
    EXPECT_FALSE(cfg.returnNativeResolution)
        << "Standalone PatchMatch callers must retain the historical full-size output contract";
    EXPECT_TRUE(cfg.enablePerPixelSourceSelection);
    EXPECT_TRUE(cfg.enableAsymmetricPropagation);
    EXPECT_TRUE(cfg.enableFinalPropagationPass);
    EXPECT_TRUE(cfg.enableGeometricGuidancePass);
    EXPECT_GT(cfg.geometricGuidanceWeight, 0.0f);
}

TEST(PatchMatchConfigTest, PostProcessingEnabled)
{
    PatchMatchConfig cfg;

    // 中值滤波应启用
    EXPECT_TRUE(cfg.doMedianBlur);
    EXPECT_EQ(cfg.medianKernelSize, 5) << "Median kernel should be 5 for effective noise removal";

    // 双边滤波应启用
    EXPECT_TRUE(cfg.doBilateralFilter);
    EXPECT_EQ(cfg.bilateralD, 9);
    EXPECT_GT(cfg.bilateralSigmaColor, 0.f);
    EXPECT_GT(cfg.bilateralSigmaSpace, 0.f);
    EXPECT_FALSE(cfg.enableReferenceGuidedFilter);
    EXPECT_GT(cfg.bilateralSigmaGuidance, 0.f);
}

TEST(PatchMatchConfigTest, BackendIdsAreStableAndUnambiguous)
{
    EXPECT_STREQ(patchMatchBackendId(PatchMatchBackend::Auto), "auto");
    EXPECT_STREQ(patchMatchBackendId(PatchMatchBackend::Cpu), "cpu");
    EXPECT_STREQ(patchMatchBackendId(PatchMatchBackend::Cuda), "cuda");
    EXPECT_STREQ(patchMatchBackendId(PatchMatchBackend::OpenCl), "opencl");
}

TEST(DepthGenConfigTest, PointCloudProcessingDefaultsToOrderedAutoSelection)
{
    const DepthGenConfig config;

    EXPECT_EQ(config.pointCloudProcessingDevice, plapoint::ProcessingDevice::Auto);
}

TEST(PatchMatchConfigTest, GeometricConsistencyEnabled)
{
    PatchMatchConfig cfg;

    EXPECT_TRUE(cfg.geomConsistency) << "Geometric consistency should be enabled by default";
    EXPECT_FLOAT_EQ(cfg.geomConsistencyMaxErr, 1.0f) << "Max geometric consistency error should be 1.0 pixel";
}

// ─── MVS 源视图 / 稀疏 hint 可见性测试 ───────────────────────────

TEST(MvsSourceViewSelectionTest, SelectsSparseOverlapInsteadOfNearestIndex)
{
    const double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double badLeft[3] = {-100, 0, 0};
    const double badRight[3] = {100, 0, 0};
    const double refCenter[3] = {0, 0, 0};
    const double goodCenter[3] = {0.1, 0, 0};

    std::vector<CameraView> views(6);
    views[0].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badLeft, false);
    views[1].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badLeft, false);
    views[2].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, refCenter, false);
    views[3].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badRight, false);
    views[4].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badRight, false);
    views[5].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, goodCenter, false);
    for (auto& view : views)
    {
        view.imageWidth = 100;
        view.imageHeight = 100;
    }

    SparseCloud sparse;
    sparse.points = {{{0.00f, 0.00f, 10.0f}, {0.02f, 0.01f, 10.0f}, {-0.01f, 0.03f, 10.0f}}};

    const std::vector<int> selected = selectMvsSourceViewIndices(views, sparse, 2, 1);

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front(), 5);
}

TEST(MvsSourceViewSelectionTest, DoesNotPadScoredSourcesWithZeroOverlapNeighbors)
{
    const double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double badLeft[3] = {-100, 0, 0};
    const double badRight[3] = {100, 0, 0};
    const double refCenter[3] = {0, 0, 0};
    const double goodCenter[3] = {0.1, 0, 0};

    std::vector<CameraView> views(4);
    views[0].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badLeft, false);
    views[1].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, refCenter, false);
    views[2].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badRight, false);
    views[3].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, goodCenter, false);
    for (auto& view : views)
    {
        view.imageWidth = 100;
        view.imageHeight = 100;
    }

    SparseCloud sparse;
    sparse.points = {{{0.00f, 0.00f, 10.0f}, {0.02f, 0.01f, 10.0f}}};

    const std::vector<int> selected = selectMvsSourceViewIndices(views, sparse, 1, 3);

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front(), 3);
}

TEST(MvsSourceViewSelectionTest, RejectsPointBehindEitherCameraForBaselineGeometry)
{
    const double rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double firstCenter[3] = {0, 0, 0};
    const double secondCenter[3] = {1, 0, 0};

    const auto first = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, rotation, firstCenter, false);
    const auto second = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, rotation, secondCenter, false);
    const placamera::CameraBaseline front = placamera::CameraBaseline::evaluate(first, second, {{0.0, 0.0, 10.0}});
    ASSERT_TRUE(front.triangulationAngleDeg());
    EXPECT_NEAR(*front.triangulationAngleDeg(), 5.710593, 1e-5);
    const placamera::CameraBaseline behind = placamera::CameraBaseline::evaluate(first, second, {{0.0, 0.0, -10.0}});
    EXPECT_FALSE(behind.isPointInFrontOfBothCameras());
}

TEST(MvsSparseHintVisibilityTest, RequiresReferenceAndSelectedSourceVisibility)
{
    const double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double refCenter[3] = {0, 0, 0};
    const double sourceCenter[3] = {0.1, 0, 0};

    std::vector<CameraView> views(2);
    views[0].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, refCenter, false);
    views[1].camera = makeViewCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, sourceCenter, false);
    for (auto& view : views)
    {
        view.imageWidth = 100;
        view.imageHeight = 100;
    }

    SparseCloud sparse;
    sparse.points = {{
        {0.00f, 0.00f, 10.0f},  // ref/source 都可见
        {-0.49f, 0.00f, 10.0f}, // ref 可见，source 中落到左边界外
        {0.00f, 1.00f, 10.0f}   // ref/source 都不可见
    }};

    const std::vector<size_t> visible = collectMvsVisibleSparsePointIndices(views, sparse, 0, std::vector<int>{1}, 1);

    ASSERT_EQ(visible.size(), 1u);
    EXPECT_EQ(visible.front(), 0u);
}

// ─── FusionConfig 参数验证 ──────────────────────────────────────

TEST(FusionConfigTest, DefaultParametersOptimized)
{
    FusionConfig cfg;

    // 正式 dense cloud 要求多视一致；快速预览应由 GUI/profile 显式放宽。
    EXPECT_EQ(cfg.minConsistentViews, 3);
    EXPECT_FLOAT_EQ(cfg.relDepthThresh, 0.03f)
        << "Relative depth threshold should be strict enough to suppress vertical spikes";
    EXPECT_FLOAT_EQ(cfg.pixelThresh, 1.5f) << "Pixel threshold should be strict for production fusion";
    EXPECT_FLOAT_EQ(cfg.confidenceThresh, 0.65f)
        << "Fusion confidence threshold should reject low-confidence near-full depth maps";
    EXPECT_TRUE(cfg.enableAdaptiveConfidenceFilter);
    EXPECT_FLOAT_EQ(cfg.adaptiveFullCoverageThreshold, 0.95f);
    EXPECT_FLOAT_EQ(cfg.adaptiveLowMeanConfidenceThreshold, 0.65f);
    EXPECT_FLOAT_EQ(cfg.adaptiveStrictConfidenceThreshold, 0.65f);
}

TEST(FusionConfigTest, InpaintEnabled)
{
    FusionConfig cfg;
    EXPECT_TRUE(cfg.doInpaint);
    EXPECT_GT(cfg.inpaintRadius, 0);
}

TEST(FusionConfigTest, SigmaFusionEnabled)
{
    FusionConfig cfg;
    EXPECT_TRUE(cfg.doSigmaFusion);
    EXPECT_GT(cfg.sigmaMultiplier, 0.f);
}

TEST(MvsQualityReportTest, DetectsLocalDepthSpikesEvenWithHighConfidence)
{
    cv::Mat depth(8, 8, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(8, 8, CV_32F, cv::Scalar(0.90f));
    depth.at<float>(2, 2) = 40.0f;
    depth.at<float>(5, 5) = 42.0f;

    const DepthMapQualityMetrics metrics = analyzeDepthMapQuality(depth, confidence, 5);

    EXPECT_EQ(metrics.validPixelCount, 64);
    EXPECT_FLOAT_EQ(metrics.validCoverage, 1.0f);
    EXPECT_EQ(metrics.localDepthOutlierCount, 2);
    EXPECT_GT(metrics.localDepthOutlierRatio, 0.02f);
    EXPECT_TRUE(metrics.hasLocalDepthOutliers);

    const QJsonObject json = depthMapQualityMetricsToJson(metrics);
    EXPECT_EQ(json.value(QStringLiteral("local_depth_outlier_count")).toInt(), 2);
    EXPECT_TRUE(json.value(QStringLiteral("has_local_depth_outliers")).toBool(false));
}

// ─── PlaCamera 正深度归一化测试 ────────────────────────────────

// 从显式 ASP/Tsai 语义参数构造正深度模型
TEST(CameraPositiveDepthTest, FromCameraBasic)
{
    // 简单单位相机：焦距 1000px，主点 (512,384)，无翻转
    double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // identity
    double C[3] = {0, 0, 0};

    const auto camera = makeCamera(1000.0, 1000.0, 512.0, 384.0, 1, 1, R_wc, C, false);
    const auto cam = camera.normalizedForPositiveDepth(placamera::CameraDefinitionId("mvs-positive-depth-definition"),
                                                       placamera::CameraInstanceId("mvs-positive-depth-instance"));

    EXPECT_FALSE(cam.pinholeDefinition().depthAxisFlipped());
    EXPECT_DOUBLE_EQ(cam.pinholeDefinition().intrinsics().focalX, 1000.0);
    EXPECT_DOUBLE_EQ(cam.pinholeDefinition().intrinsics().focalY, 1000.0);
    EXPECT_DOUBLE_EQ(cam.pinholeDefinition().intrinsics().principalX, 512.0);
    EXPECT_DOUBLE_EQ(cam.pinholeDefinition().intrinsics().principalY, 384.0);
}

// 投影测试：光轴上的点应投影到主点
TEST(CameraPositiveDepthTest, ProjectOnAxis)
{
    double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double C[3] = {0, 0, 0};

    const auto cam = makeCamera(1000.0, 1000.0, 512.0, 384.0, 1, 1, R_wc, C, false);
    const auto projected = cam.groundToImage({cam.groundFrame(), {0.0, 0.0, 10.0}});
    ASSERT_TRUE(projected) << "Point on optical axis should project successfully";
    EXPECT_NEAR(projected.value().image.sample, 512.0, 0.01);
    EXPECT_NEAR(projected.value().image.line, 384.0, 0.01);
}

// 投影-反投影往返一致性
TEST(CameraPositiveDepthTest, ProjectUnprojectRoundtrip)
{
    double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double C[3] = {5, 3, -2};

    const auto cam = makeCamera(800.0, 800.0, 400.0, 300.0, 1, 1, R_wc, C, false);

    // 测试点
    const placamera::GroundCoordinate world{cam.groundFrame(), {10.0, 5.0, 20.0}};

    // 正向投影
    const auto projected = cam.groundToImage(world);
    ASSERT_TRUE(projected) << "Projection should succeed for visible point";
    ASSERT_TRUE(projected.value().positiveDepth);
    const double depth = *projected.value().positiveDepth;
    ASSERT_GT(depth, 0.0) << "Point should be in front of camera";

    // 反投影
    const auto reconstructed = cam.imageToGroundAtDepth(projected.value().image, depth);
    ASSERT_TRUE(reconstructed);

    EXPECT_NEAR(reconstructed.value().position[0], world.position[0], 0.05);
    EXPECT_NEAR(reconstructed.value().position[1], world.position[1], 0.05);
    EXPECT_NEAR(reconstructed.value().position[2], world.position[2], 0.05);
}

// 点在相机后方 → project 返回 false
TEST(CameraPositiveDepthTest, ProjectBehindCamera)
{
    double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double C[3] = {0, 0, 0};

    const auto cam = makeCamera(1000.0, 1000.0, 512.0, 384.0, 1, 1, R_wc, C, false);
    // Z = -10 在相机后方
    EXPECT_FALSE(cam.groundToImage({cam.groundFrame(), {0.0, 0.0, -10.0}}))
        << "Point behind camera should fail projection";
}

// depthFlippedZ 参数测试
TEST(CameraPositiveDepthTest, DepthFlippedZ)
{
    double R_wc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double C[3] = {0, 0, 0};

    // 正常模式
    const auto camNormal = makeCamera(1000.0, 1000.0, 512.0, 384.0, 1, 1, R_wc, C, false);

    // Z 翻转模式
    const auto flipped = makeCamera(1000.0, 1000.0, 512.0, 384.0, 1, 1, R_wc, C, true);
    const auto camFlipped =
        flipped.normalizedForPositiveDepth(placamera::CameraDefinitionId("mvs-flipped-positive-definition"),
                                           placamera::CameraInstanceId("mvs-flipped-positive-instance"));

    // 正常模式下 Z>0 的点可以投影
    EXPECT_TRUE(camNormal.groundToImage({camNormal.groundFrame(), {1.0, 1.0, 10.0}}));

    // Z 翻转模式下同一点（世界坐标 Z>0）在翻转后 Zc < 0，应该不能投影
    // 但翻转后 Z<0 的世界点 （Zc > 0）应该可以投影
    EXPECT_TRUE(camFlipped.groundToImage({camFlipped.groundFrame(), {1.0, 1.0, -10.0}}))
        << "In flipped mode, negative-Z world point should project";
}

// 从真实 .tsai 构造正深度模型
TEST(CameraPositiveDepthTest, FromRealTsaiCamera)
{
    std::string path;
#ifdef TEST_DATA_DIR
    path = std::string(TEST_DATA_DIR) + "/tsai/1.tsai";
#else
    path = "../testData/tsai/1.tsai";
#endif

    const auto parsed = placamera::loadTsaiFramePinhole(
        path, placamera::CameraDefinitionId("mvs-tsai"), placamera::FrameId("mvs-test-world"));
    ASSERT_TRUE(parsed) << parsed.message();
    const auto camera = placamera::FramePinholeModel::create(placamera::CameraInstanceId("mvs-tsai"),
                                                             placamera::ImageId("mvs-tsai-image"),
                                                             parsed.value().definition,
                                                             placamera::ImageSize{4000, 3000},
                                                             parsed.value().pose);
    const auto normalized = camera.normalizedForPositiveDepth(placamera::CameraDefinitionId("mvs-tsai-positive-depth"),
                                                              placamera::CameraInstanceId("mvs-tsai-positive-depth"));

    EXPECT_GT(normalized.pinholeDefinition().intrinsics().focalX, 0.0);
    EXPECT_GT(normalized.pinholeDefinition().intrinsics().focalY, 0.0);
    EXPECT_FALSE(normalized.pinholeDefinition().depthAxisFlipped());

    // 相机中心应一致
    for (int coordinate = 0; coordinate < 3; ++coordinate)
    {
        EXPECT_NEAR(normalized.pose().center[coordinate], camera.pose().center[coordinate], 0.1);
    }
}

// ─── DepthGenConfig 复合配置 ────────────────────────────────────

TEST(DepthGenConfigTest, DefaultConfig)
{
    DepthGenConfig cfg;

    // 内嵌的 PatchMatch 和 Fusion 配置应与独立创建一致
    PatchMatchConfig pm;
    EXPECT_EQ(cfg.patchMatch.patchHalf, pm.patchHalf);
    EXPECT_FLOAT_EQ(cfg.patchMatch.confidenceThresh, pm.confidenceThresh);

    FusionConfig fu;
    EXPECT_FLOAT_EQ(cfg.fusion.relDepthThresh, fu.relDepthThresh);
    EXPECT_FLOAT_EQ(cfg.fusion.pixelThresh, fu.pixelThresh);
    EXPECT_FALSE(cfg.depthPoseRefinement.enabled);
    EXPECT_TRUE(cfg.depthPoseRefinement.emitDerivedCameraCandidates);
    EXPECT_EQ(cfg.gpuFrameWorkerCount, 2);
}

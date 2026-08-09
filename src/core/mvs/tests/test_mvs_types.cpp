// ============================================================
// test_mvs_types.cpp — MVS 类型与参数配置单元测试
//
// 测试内容：
//   1. PatchMatchConfig 默认值验证（优化后的参数）
//   2. FusionConfig 默认值验证
//   3. Camera 正深度归一化与投影
//   4. Camera 投影-反投影一致性
//   5. 深度图后处理参数验证
// ============================================================

#include <gtest/gtest.h>
#include "MvsQualityReport.h"
#include "MvsTypes.h"
#include "MvsViewSelection.h"
#include "Camera.h"

#include <QJsonObject>

#include <cmath>
#include <array>

using namespace xjw::mvs;

namespace
{

xjw::Camera makeCamera(double fu,
                       double fv,
                       double cu,
                       double cv,
                       int uDir,
                       int vDir,
                       const double r_wc[9],
                       const double center[3],
                       bool depthAxisFlipped)
{
    xjw::Camera camera;
    std::array<double, 9> rotation{{
        r_wc[0], r_wc[1], r_wc[2],
        r_wc[3], r_wc[4], r_wc[5],
        r_wc[6], r_wc[7], r_wc[8]
    }};
    std::array<double, 3> cameraCenter{{center[0], center[1], center[2]}};
    camera.setIntrinsics(fu, fv, cu, cv);
    camera.setPose(rotation, cameraCenter);
    camera.setAxisDirections(uDir, vDir);
    camera.setDepthAxisFlipped(depthAxisFlipped);
    return camera;
}

} // namespace

// ─── PatchMatchConfig 参数验证 ─────────────────────────────────

TEST(PatchMatchConfigTest, DefaultParametersOptimized)
{
    PatchMatchConfig cfg;

    // 生产点云默认应偏质量，避免把整幅低置信深度图直接用于融合。
    EXPECT_EQ(cfg.patchHalf, 7)
        << "patchHalf should be 7 (15x15 window) for robust matching";
    EXPECT_EQ(cfg.numIterations, 16);
    EXPECT_GE(cfg.numSourceViews, 5)
        << "Aerial production MVS should use enough source views for consensus";
    EXPECT_FLOAT_EQ(cfg.confidenceThresh, 0.60f)
        << "Production PatchMatch confidence threshold should reject low-confidence full-frame depths";
    EXPECT_TRUE(cfg.useCuda);
    EXPECT_FALSE(cfg.cudaFallbackToCpu)
        << "CUDA failures must remain visible after the run selects CUDA";
    EXPECT_FALSE(cfg.openClFallbackToCpu)
        << "OpenCL failures must remain visible instead of running a GPU-tagged CPU fallback";
    EXPECT_EQ(cfg.cudaDeviceIndex, -1);
    EXPECT_EQ(cfg.downsampleFactor, 2);
    EXPECT_FALSE(cfg.returnNativeResolution)
        << "Standalone PatchMatch callers must retain the historical full-size output contract";
}

TEST(PatchMatchConfigTest, PostProcessingEnabled)
{
    PatchMatchConfig cfg;

    // 中值滤波应启用
    EXPECT_TRUE(cfg.doMedianBlur);
    EXPECT_EQ(cfg.medianKernelSize, 5)
        << "Median kernel should be 5 for effective noise removal";

    // 双边滤波应启用
    EXPECT_TRUE(cfg.doBilateralFilter);
    EXPECT_EQ(cfg.bilateralD, 9);
    EXPECT_GT(cfg.bilateralSigmaColor, 0.f);
    EXPECT_GT(cfg.bilateralSigmaSpace, 0.f);
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

    EXPECT_TRUE(cfg.geomConsistency)
        << "Geometric consistency should be enabled by default";
    EXPECT_FLOAT_EQ(cfg.geomConsistencyMaxErr, 1.0f)
        << "Max geometric consistency error should be 1.0 pixel";
}

// ─── MVS 源视图 / 稀疏 hint 可见性测试 ───────────────────────────

TEST(MvsSourceViewSelectionTest, SelectsSparseOverlapInsteadOfNearestIndex)
{
    const double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    const double badLeft[3] = {-100, 0, 0};
    const double badRight[3] = {100, 0, 0};
    const double refCenter[3] = {0, 0, 0};
    const double goodCenter[3] = {0.1, 0, 0};

    std::vector<CameraView> views(6);
    views[0].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badLeft, false);
    views[1].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badLeft, false);
    views[2].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, refCenter, false);
    views[3].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badRight, false);
    views[4].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badRight, false);
    views[5].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, goodCenter, false);
    for (auto &view : views)
    {
        view.imageWidth = 100;
        view.imageHeight = 100;
    }

    SparseCloud sparse;
    sparse.points = {{
        {0.00f, 0.00f, 10.0f},
        {0.02f, 0.01f, 10.0f},
        {-0.01f, 0.03f, 10.0f}
    }};

    const std::vector<int> selected = selectMvsSourceViewIndices(views, sparse, 2, 1);

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front(), 5);
}

TEST(MvsSourceViewSelectionTest, DoesNotPadScoredSourcesWithZeroOverlapNeighbors)
{
    const double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    const double badLeft[3] = {-100, 0, 0};
    const double badRight[3] = {100, 0, 0};
    const double refCenter[3] = {0, 0, 0};
    const double goodCenter[3] = {0.1, 0, 0};

    std::vector<CameraView> views(4);
    views[0].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badLeft, false);
    views[1].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, refCenter, false);
    views[2].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, badRight, false);
    views[3].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, goodCenter, false);
    for (auto &view : views)
    {
        view.imageWidth = 100;
        view.imageHeight = 100;
    }

    SparseCloud sparse;
    sparse.points = {{
        {0.00f, 0.00f, 10.0f},
        {0.02f, 0.01f, 10.0f}
    }};

    const std::vector<int> selected = selectMvsSourceViewIndices(views, sparse, 1, 3);

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front(), 3);
}

TEST(MvsSourceViewSelectionTest, RejectsPointBehindEitherCameraForBaselineGeometry)
{
    const double rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double firstCenter[3] = {0, 0, 0};
    const double secondCenter[3] = {1, 0, 0};

    CameraView first;
    first.camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1,
                              rotation, firstCenter, false);
    CameraView second;
    second.camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1,
                               rotation, secondCenter, false);

    EXPECT_NEAR(mvsTriangulationAngleDeg(first, second, {{0.0f, 0.0f, 10.0f}}),
                5.710593f,
                1e-5f);
    EXPECT_FLOAT_EQ(mvsTriangulationAngleDeg(first, second, {{0.0f, 0.0f, -10.0f}}), 0.0f);
}

TEST(MvsSparseHintVisibilityTest, RequiresReferenceAndSelectedSourceVisibility)
{
    const double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    const double refCenter[3] = {0, 0, 0};
    const double sourceCenter[3] = {0.1, 0, 0};

    std::vector<CameraView> views(2);
    views[0].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, refCenter, false);
    views[1].camera = makeCamera(1000.0, 1000.0, 50.0, 50.0, 1, 1, R_wc, sourceCenter, false);
    for (auto &view : views)
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

    const std::vector<size_t> visible =
        collectMvsVisibleSparsePointIndices(views, sparse, 0, std::vector<int>{1}, 1);

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
    EXPECT_FLOAT_EQ(cfg.pixelThresh, 1.5f)
        << "Pixel threshold should be strict for production fusion";
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

    const DepthMapQualityMetrics metrics =
        analyzeDepthMapQuality(depth, confidence, 5);

    EXPECT_EQ(metrics.validPixelCount, 64);
    EXPECT_FLOAT_EQ(metrics.validCoverage, 1.0f);
    EXPECT_EQ(metrics.localDepthOutlierCount, 2);
    EXPECT_GT(metrics.localDepthOutlierRatio, 0.02f);
    EXPECT_TRUE(metrics.hasLocalDepthOutliers);

    const QJsonObject json = depthMapQualityMetricsToJson(metrics);
    EXPECT_EQ(json.value(QStringLiteral("local_depth_outlier_count")).toInt(), 2);
    EXPECT_TRUE(json.value(QStringLiteral("has_local_depth_outliers")).toBool(false));
}

// ─── Camera 正深度归一化测试 ────────────────────────────────

// 从显式 ASP/Tsai 语义参数构造正深度模型
TEST(CameraPositiveDepthTest, FromCameraBasic)
{
    // 简单单位相机：焦距 1000px，主点 (512,384)，无翻转
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1}; // identity
    double C[3] = {0, 0, 0};

    xjw::Camera camera = makeCamera(
        1000.0, 1000.0,
        512.0, 384.0,
        1, 1,
        R_wc, C,
        false);
    const xjw::Camera cam = camera.normalizedForPositiveDepth();

    EXPECT_TRUE(cam.isValid());
    EXPECT_DOUBLE_EQ(cam.focalX(), 1000.0);
    EXPECT_DOUBLE_EQ(cam.focalY(), 1000.0);
    EXPECT_DOUBLE_EQ(cam.principalX(), 512.0);
    EXPECT_DOUBLE_EQ(cam.principalY(), 384.0);
}

// 投影测试：光轴上的点应投影到主点
TEST(CameraPositiveDepthTest, ProjectOnAxis)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {0, 0, 0};

    auto cam = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, false).normalizedForPositiveDepth();

    const double world[3] = {0.0, 0.0, 10.0};
    double pixel[2] = {};
    bool ok = cam.projectWorldPoint(world, pixel);
    EXPECT_TRUE(ok) << "Point on optical axis should project successfully";
    EXPECT_NEAR(pixel[0], 512.0, 0.01) << "Should project to principal point u";
    EXPECT_NEAR(pixel[1], 384.0, 0.01) << "Should project to principal point v";
}

// 投影-反投影往返一致性
TEST(CameraPositiveDepthTest, ProjectUnprojectRoundtrip)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {5, 3, -2};

    auto cam = makeCamera(
        800.0, 800.0, 400.0, 300.0,
        1, 1, R_wc, C, false).normalizedForPositiveDepth();

    // 测试点
    const double world[3] = {10.0, 5.0, 20.0};

    // 正向投影
    double pixel[2] = {};
    double depth = 0.0;
    bool ok = cam.projectWorldPointWithDepth(world, pixel, depth);
    ASSERT_TRUE(ok) << "Projection should succeed for visible point";
    ASSERT_GT(depth, 0.0) << "Point should be in front of camera";

    // 反投影
    double reconstructed[3] = {};
    ASSERT_TRUE(cam.unprojectPixel(pixel, depth, reconstructed));

    EXPECT_NEAR(reconstructed[0], world[0], 0.05) << "Roundtrip X should match";
    EXPECT_NEAR(reconstructed[1], world[1], 0.05) << "Roundtrip Y should match";
    EXPECT_NEAR(reconstructed[2], world[2], 0.05) << "Roundtrip Z should match";
}

// 点在相机后方 → project 返回 false
TEST(CameraPositiveDepthTest, ProjectBehindCamera)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {0, 0, 0};

    auto cam = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, false).normalizedForPositiveDepth();

    const double world[3] = {0.0, 0.0, -10.0};
    double pixel[2] = {};
    // Z = -10 在相机后方
    bool ok = cam.projectWorldPoint(world, pixel);
    EXPECT_FALSE(ok) << "Point behind camera should fail projection";
}

// depthFlippedZ 参数测试
TEST(CameraPositiveDepthTest, DepthFlippedZ)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {0, 0, 0};

    // 正常模式
    auto camNormal = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, false).normalizedForPositiveDepth();

    // Z 翻转模式
    auto camFlipped = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, true).normalizedForPositiveDepth();

    // 正常模式下 Z>0 的点可以投影
    const double normalWorld[3] = {1.0, 1.0, 10.0};
    double normalPixel[2] = {};
    EXPECT_TRUE(camNormal.projectWorldPoint(normalWorld, normalPixel));

    // Z 翻转模式下同一点（世界坐标 Z>0）在翻转后 Zc < 0，应该不能投影
    // 但翻转后 Z<0 的世界点 （Zc > 0）应该可以投影
    const double flippedWorld[3] = {1.0, 1.0, -10.0};
    double flippedPixel[2] = {};
    bool okFlipped = camFlipped.projectWorldPoint(flippedWorld, flippedPixel);
    EXPECT_TRUE(okFlipped) << "In flipped mode, negative-Z world point should project";
}

// 从真实 .tsai 构造正深度模型
TEST(CameraPositiveDepthTest, FromRealTsaiCamera)
{
    xjw::Camera cam;
    std::string path;
#ifdef TEST_DATA_DIR
    path = std::string(TEST_DATA_DIR) + "/tsai/1.tsai";
#else
    path = "../testData/tsai/1.tsai";
#endif

    if (!cam.loadFromFile(path)) {
        GTEST_SKIP() << "Could not load test camera file";
    }

    auto C = cam.cameraCenter();

    const xjw::Camera ccam = cam.normalizedForPositiveDepth();

    EXPECT_TRUE(ccam.isValid());
    EXPECT_GT(ccam.focalX(), 0.0);
    EXPECT_GT(ccam.focalY(), 0.0);

    // 相机中心应一致
    const std::array<double, 3> normalizedCenter = ccam.cameraCenter();
    EXPECT_NEAR(normalizedCenter[0], C[0], 0.1);
    EXPECT_NEAR(normalizedCenter[1], C[1], 0.1);
    EXPECT_NEAR(normalizedCenter[2], C[2], 0.1);
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

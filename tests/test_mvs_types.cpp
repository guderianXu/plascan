// ============================================================
// test_mvs_types.cpp — MVS 类型与参数配置单元测试
//
// 测试内容：
//   1. PatchMatchConfig 默认值验证（优化后的参数）
//   2. FusionConfig 默认值验证
//   3. Camera::PositiveDepthModel 构造与投影
//   4. Camera::PositiveDepthModel 投影-反投影一致性
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
    EXPECT_EQ(cfg.downsampleFactor, 2);
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

// ─── Camera::PositiveDepthModel 测试 ───────────────────────────

// 从显式 ASP/Tsai 语义参数构造正深度模型
TEST(PositiveDepthCameraModelTest, FromCameraBasic)
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
    auto cam = camera.toPositiveDepthModel();

    EXPECT_TRUE(cam.valid());
    EXPECT_FLOAT_EQ(cam.fx, 1000.f);
    EXPECT_FLOAT_EQ(cam.fy, 1000.f);
    EXPECT_FLOAT_EQ(cam.cx, 512.f);
    EXPECT_FLOAT_EQ(cam.cy, 384.f);
}

// 投影测试：光轴上的点应投影到主点
TEST(PositiveDepthCameraModelTest, ProjectOnAxis)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {0, 0, 0};

    auto cam = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, false).toPositiveDepthModel();

    float u, v;
    bool ok = cam.project(0.f, 0.f, 10.f, u, v);
    EXPECT_TRUE(ok) << "Point on optical axis should project successfully";
    EXPECT_NEAR(u, 512.f, 0.01f) << "Should project to principal point u";
    EXPECT_NEAR(v, 384.f, 0.01f) << "Should project to principal point v";
}

// 投影-反投影往返一致性
TEST(PositiveDepthCameraModelTest, ProjectUnprojectRoundtrip)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {5, 3, -2};

    auto cam = makeCamera(
        800.0, 800.0, 400.0, 300.0,
        1, 1, R_wc, C, false).toPositiveDepthModel();

    // 测试点
    float Xw = 10.f, Yw = 5.f, Zw = 20.f;

    // 正向投影
    float u, v;
    bool ok = cam.project(Xw, Yw, Zw, u, v);
    ASSERT_TRUE(ok) << "Projection should succeed for visible point";

    // 计算深度 (Zc)
    float Xc = cam.R_cw[0]*Xw + cam.R_cw[1]*Yw + cam.R_cw[2]*Zw + cam.T[0];
    float Yc = cam.R_cw[3]*Xw + cam.R_cw[4]*Yw + cam.R_cw[5]*Zw + cam.T[1];
    float Zc = cam.R_cw[6]*Xw + cam.R_cw[7]*Yw + cam.R_cw[8]*Zw + cam.T[2];
    ASSERT_GT(Zc, 0) << "Point should be in front of camera";

    // 反投影
    float Xw2, Yw2, Zw2;
    cam.unproject(u, v, Zc, Xw2, Yw2, Zw2);

    EXPECT_NEAR(Xw2, Xw, 0.05f) << "Roundtrip X should match";
    EXPECT_NEAR(Yw2, Yw, 0.05f) << "Roundtrip Y should match";
    EXPECT_NEAR(Zw2, Zw, 0.05f) << "Roundtrip Z should match";
}

// 点在相机后方 → project 返回 false
TEST(PositiveDepthCameraModelTest, ProjectBehindCamera)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {0, 0, 0};

    auto cam = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, false).toPositiveDepthModel();

    float u, v;
    // Z = -10 在相机后方
    bool ok = cam.project(0.f, 0.f, -10.f, u, v);
    EXPECT_FALSE(ok) << "Point behind camera should fail projection";
}

// depthFlippedZ 参数测试
TEST(PositiveDepthCameraModelTest, DepthFlippedZ)
{
    double R_wc[9] = {1,0,0, 0,1,0, 0,0,1};
    double C[3] = {0, 0, 0};

    // 正常模式
    auto camNormal = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, false).toPositiveDepthModel();

    // Z 翻转模式
    auto camFlipped = makeCamera(
        1000.0, 1000.0, 512.0, 384.0,
        1, 1, R_wc, C, true).toPositiveDepthModel();

    // 正常模式下 Z>0 的点可以投影
    float u1, v1;
    EXPECT_TRUE(camNormal.project(1.f, 1.f, 10.f, u1, v1));

    // Z 翻转模式下同一点（世界坐标 Z>0）在翻转后 Zc < 0，应该不能投影
    // 但翻转后 Z<0 的世界点 （Zc > 0）应该可以投影
    float u2, v2;
    bool okFlipped = camFlipped.project(1.f, 1.f, -10.f, u2, v2);
    EXPECT_TRUE(okFlipped) << "In flipped mode, negative-Z world point should project";
}

// 从真实 .tsai 构造正深度模型
TEST(PositiveDepthCameraModelTest, FromRealTsaiCamera)
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
    auto R = cam.cameraToWorldRotation();

    auto ccam = cam.toPositiveDepthModel();

    EXPECT_TRUE(ccam.valid());
    EXPECT_GT(ccam.fx, 0.f);
    EXPECT_GT(ccam.fy, 0.f);

    // 相机中心应一致
    EXPECT_NEAR(ccam.C[0], static_cast<float>(C[0]), 0.1f);
    EXPECT_NEAR(ccam.C[1], static_cast<float>(C[1]), 0.1f);
    EXPECT_NEAR(ccam.C[2], static_cast<float>(C[2]), 0.1f);
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
}

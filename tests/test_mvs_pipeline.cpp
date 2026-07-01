// =============================================================================
// MVS 管道集成测试
// 覆盖：CPU PatchMatch 深度估计 → DepthMapFusion 融合 → DenseCloudBuilder 反投影
// =============================================================================

#include <gtest/gtest.h>

#include "PatchMatchCUDA.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "DenseCloudBuilder.h"
#include "MvsQualityReport.h"
#include "SparseCloudPreprocessor.h"
#include "Camera.h"

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/io/ply_io.h>

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

// 合成一张有纹理的灰度图
cv::Mat makeSyntheticGray(int w, int h)
{
    cv::Mat img(h, w, CV_8U);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            double v = 128.0
                + 40.0 * std::sin(0.21 * x)
                + 35.0 * std::cos(0.17 * y)
                + (((x / 5) + (y / 6)) % 2 == 0 ? 20.0 : -20.0);
            img.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
        }
    return img;
}

// 水平位移图（模拟视差）
cv::Mat makeShifted(const cv::Mat &src, int d)
{
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            if (x + d >= 0 && x + d < src.cols)
                dst.at<uint8_t>(y, x) = src.at<uint8_t>(y, x + d);
    return dst;
}

xjw::mvs::PositiveDepthCameraModel makePosCam(double fu, double fv,
                                               double cu, double cv,
                                               const double Rwc[9],
                                               const double C[3])
{
    xjw::Camera cam;
    std::array<double, 9> R{Rwc[0],Rwc[1],Rwc[2],Rwc[3],Rwc[4],Rwc[5],Rwc[6],Rwc[7],Rwc[8]};
    std::array<double, 3> Cv{C[0],C[1],C[2]};
    cam.setIntrinsics(fu, fv, cu, cv);
    cam.setPose(R, Cv);
    cam.setAxisDirections(1, 1);
    cam.setDepthAxisFlipped(false);
    return cam.toPositiveDepthModel();
}

} // namespace

TEST(MvsPipelineTest, SparseCloudPreprocessorReadsBinaryPly)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_mvs_binary_sparse_test";
    fs::create_directories(root);
    const fs::path plyPath = root / "sparse_binary.ply";

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points(0, 0) = 0.0f; points(0, 1) = 0.0f; points(0, 2) = 0.0f;
    points(1, 0) = 1.0f; points(1, 1) = 0.0f; points(1, 2) = 0.0f;
    points(2, 0) = 0.0f; points(2, 1) = 1.0f; points(2, 2) = 0.0f;
    points(3, 0) = 0.0f; points(3, 1) = 0.0f; points(3, 2) = 1.0f;
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    plapoint::io::writePly<float>(plyPath.string(), cloud, plapoint::io::PlyFormat::BinaryLE);

    xjw::mvs::SparseCloudPreprocessor preprocessor(plapoint::ProcessingDevice::CPU);
    xjw::mvs::PreprocessResult result;
    std::string error;
    ASSERT_TRUE(preprocessor.run(plyPath.string(), {}, result, &error)) << error;
    EXPECT_EQ(result.rawCount, 4);
    EXPECT_EQ(result.filteredCount, 4);
    EXPECT_EQ(result.cloud.points.size(), 4u);
}

// ---------------------------------------------------------------------------
// Test 1: CPU PatchMatch 输出格式与有效像素
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, PatchMatchCpuOutputValid)
{
    constexpr int W = 80, H = 60;
    constexpr double FOCAL = 64.0, BASELINE = 1.0, DISP = 6;

    cv::Mat ref = makeSyntheticGray(W, H);
    cv::Mat src = makeShifted(ref, DISP);
    cv::GaussianBlur(src, src, cv::Size(3,3), 0.0);

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0}, C1[3]={BASELINE,0,0};
    auto refCam = makePosCam(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);
    auto srcCam = makePosCam(FOCAL, FOCAL, W*0.5, H*0.5, I, C1);

    xjw::mvs::PatchMatchConfig cfg;
    cfg.useCuda           = false;
    cfg.downsampleFactor  = 2;
    cfg.patchHalf         = 2;
    cfg.confidenceThresh  = 0.10f;
    cfg.doMedianBlur      = true;
    cfg.medianKernelSize  = 3;
    cfg.doBilateralFilter = false;

    cv::Mat depth, conf;
    std::string err;
    bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
        ref, {src}, refCam, {srcCam}, 4.0f, 20.0f, cfg, depth, &conf, &err);

    ASSERT_TRUE(ok) << err;
    ASSERT_EQ(depth.type(), CV_32F);
    ASSERT_EQ(depth.rows, H);
    ASSERT_EQ(depth.cols, W);

    // 至少 30% 的中心 ROI 像素有效
    cv::Rect roi(W/4, H/4, W/2, H/2);
    int valid = 0;
    for (int y = roi.y; y < roi.y+roi.height; ++y)
        for (int x = roi.x; x < roi.x+roi.width; ++x)
            if (depth.at<float>(y,x) > 0.0f) ++valid;

    EXPECT_GT(valid, roi.area() * 0.30)
        << "CPU PatchMatch should yield > 30% valid pixels in ROI";
}

// ---------------------------------------------------------------------------
// Test 2: DepthMapFusion 双帧融合输出非空点云
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, DepthMapFusionTwoFrames)
{
    constexpr int W = 48, H = 36;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0, BASELINE = 1.0;

    // 合成两帧一致的恒定深度图（模拟平面场景）
    cv::Mat d0(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    cv::Mat d1(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    // 边缘置 0（无效），避免边界效应
    d0(cv::Rect(0,0,W,4)) = 0;  d0(cv::Rect(0,H-4,W,4)) = 0;
    d1(cv::Rect(0,0,W,4)) = 0;  d1(cv::Rect(0,H-4,W,4)) = 0;

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0}, C1[3]={BASELINE,0,0};

    xjw::mvs::FusionFrameInput fr0, fr1;
    fr0.depthMap    = d0;
    fr0.cameraModel = makePosCam(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);
    fr0.imgW = W; fr0.imgH = H;

    fr1.depthMap    = d1;
    fr1.cameraModel = makePosCam(FOCAL, FOCAL, W*0.5, H*0.5, I, C1);
    fr1.imgW = W; fr1.imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels   = 2;
    fcfg.maxReprojError = 3.0f;
    fcfg.maxDepthError  = 0.10f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    bool ok = fusion.fuse({fr0, fr1}, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(static_cast<int>(pts.size()), 0)
        << "DepthMapFusion should produce at least one fused point";
}

TEST(MvsPipelineTest, DepthMapFusionCancelBeforeWorkClearsStaleOutput)
{
    constexpr int W = 16;
    constexpr int H = 12;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C0[3] = {0,0,0};

    xjw::mvs::FusionFrameInput frame;
    frame.depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    frame.cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C0);
    frame.imgW = W;
    frame.imgH = H;

    auto cancelFlag = std::make_shared<std::atomic_bool>(true);
    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.cancelFlag = cancelFlag;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts(1);
    pts.front().x = 123.0f;
    std::string err;

    const bool ok = fusion.fuse({frame}, pts, nullptr, &err);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(err.find("取消") != std::string::npos);
    EXPECT_TRUE(pts.empty()) << "Cancelled fusion must not leave stale points from a previous run";
}

TEST(MvsPipelineTest, DepthMapFusionTwoViewSingleObservationUsesFastParallelPath)
{
    constexpr int W = 32, H = 24;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0, BASELINE = 1.0;

    cv::Mat d0(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    cv::Mat d1(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    d0(cv::Rect(0, 0, W, 2)) = 0;
    d1(cv::Rect(0, 0, W, 2)) = 0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C0[3] = {0,0,0};
    const double C1[3] = {BASELINE,0,0};

    xjw::mvs::FusionFrameInput fr0, fr1;
    fr0.depthMap = d0;
    fr0.cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C0);
    fr0.imgW = W;
    fr0.imgH = H;

    fr1.depthMap = d1;
    fr1.cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C1);
    fr1.imgW = W;
    fr1.imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels = 1;
    fcfg.maxReprojError = 3.0f;
    fcfg.maxDepthError = 0.10f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::vector<std::string> stages;
    std::string err;
    const bool ok = fusion.fuse({fr0, fr1},
                                pts,
                                [&](const std::string &stage, float) {
                                    stages.push_back(stage);
                                },
                                &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(static_cast<int>(pts.size()), 0);
    EXPECT_TRUE(std::any_of(stages.begin(), stages.end(), [](const std::string &stage) {
        return stage.find("快速并行融合") != std::string::npos;
    }));
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionRejectsDepthsWithoutNeighborAgreement)
{
    constexpr int W = 16;
    constexpr int H = 12;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(2);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[0].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[0].imgW = W;
    frames[0].imgH = H;

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(12.0f));
    frames[1].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[1].imgW = W;
    frames[1].imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 2;
    fcfg.checkNumImages = 1;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_TRUE(pts.empty())
        << "Streaming fusion must not directly back-project first-frame depths when neighbors disagree.";
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionKeepsProductionStrictWhenFallbackIsNotEnabled)
{
    constexpr int W = 20;
    constexpr int H = 14;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[0].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[0].imgW = W;
    frames[0].imgH = H;
    frames[0].sourceImageIndices = {1, 2};

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[1].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[1].imgW = W;
    frames[1].imgH = H;

    frames[2].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(12.0f));
    frames[2].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[2].imgW = W;
    frames[2].imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_TRUE(pts.empty())
        << "Production fusion must keep strict 3-view agreement unless low-yield fallback is explicitly enabled.";
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionFallsBackToTwoViewAgreementWhenStrictYieldCollapses)
{
    constexpr int W = 20;
    constexpr int H = 14;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[0].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[0].imgW = W;
    frames[0].imgH = H;
    frames[0].sourceImageIndices = {1, 2};

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(8.0f));
    frames[1].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[1].imgW = W;
    frames[1].imgH = H;

    frames[2].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(12.0f));
    frames[2].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
    frames[2].imgW = W;
    frames[2].imgH = H;

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;
    fcfg.enableLowYieldFallback = true;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(pts.size(), 0u)
        << "When strict 3-view streaming fusion collapses, the pipeline should preserve supported "
           "2-view dense observations instead of producing a sparse-like cloud.";
}

TEST(MvsPipelineTest, DepthMapFusionFilteredDepthsIncludeAllAcceptedObservations)
{
    constexpr int W = 24;
    constexpr int H = 18;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0;

    cv::Mat depth(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    depth(cv::Rect(0, 0, W, 3)) = 0;
    depth(cv::Rect(0, H - 3, W, 3)) = 0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    for (auto &frame : frames)
    {
        frame.depthMap = depth.clone();
        frame.cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
        frame.imgW = W;
        frame.imgH = H;
    }

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 1.0f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    ASSERT_GT(static_cast<int>(pts.size()), 0);

    const auto &filteredDepths = fusion.filteredDepths();
    ASSERT_EQ(filteredDepths.size(), frames.size());

    const int validPixels = cv::countNonZero(depth > 0);
    for (size_t frameIndex = 0; frameIndex < filteredDepths.size(); ++frameIndex)
    {
        EXPECT_EQ(cv::countNonZero(filteredDepths[frameIndex] > 0), validPixels)
            << "Accepted source observations should be visible in filtered depth frame " << frameIndex;
    }
}

TEST(MvsPipelineTest, DepthMapFusionUsesPlannedSourceImagesBeforeNearestCenters)
{
    constexpr int W = 24;
    constexpr int H = 18;
    constexpr float DEPTH_VAL = 8.0f;
    constexpr double FOCAL = 40.0;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C0[3] = {0,0,0};
    const double C1[3] = {0.1,0,0};
    const double C2[3] = {0.2,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    frames[0].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    frames[0].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C0);
    frames[0].imgW = W;
    frames[0].imgH = H;
    frames[0].sourceImageIndices = {2};

    frames[1].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(0.0f));
    frames[1].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C1);
    frames[1].imgW = W;
    frames[1].imgH = H;

    frames[2].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
    frames[2].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C2);
    frames[2].imgW = W;
    frames[2].imgH = H;
    frames[2].sourceImageIndices = {0};

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.minNumPixels = 2;
    fcfg.checkNumImages = 1;
    fcfg.maxReprojError = 100.0f;
    fcfg.maxDepthError = 1.0f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(static_cast<int>(pts.size()), 0)
        << "Fusion should prefer planned source image 2 over nearest invalid image 1.";
}

TEST(MvsPipelineTest, MvsQualityReportFlagsLowConfidenceFullCoverageDepthMap)
{
    cv::Mat depth(20, 30, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(20, 30, CV_32F, cv::Scalar(0.56f));
    confidence.at<float>(10, 15) = 0.82f;

    const xjw::mvs::DepthMapQualityMetrics metrics =
        xjw::mvs::analyzeDepthMapQuality(depth, confidence, 3);

    EXPECT_GT(metrics.validCoverage, 0.95f);
    EXPECT_LT(metrics.meanConfidence, 0.65f);
    EXPECT_TRUE(metrics.lowConfidenceFullCoverage);
    EXPECT_GE(metrics.recommendedFusionConfidence, 0.65f);

    const QJsonObject json = xjw::mvs::depthMapQualityMetricsToJson(metrics);
    EXPECT_TRUE(json.value(QStringLiteral("low_confidence_full_coverage")).toBool());
    EXPECT_GE(json.value(QStringLiteral("recommended_fusion_confidence")).toDouble(), 0.65);
}

TEST(MvsPipelineTest, SparseSupportMaskTracksProjectedSparseStructure)
{
    constexpr int W = 120;
    constexpr int H = 90;
    constexpr double FOCAL = 80.0;
    constexpr float Z = 10.0f;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    xjw::mvs::CameraView view;
    view.imageWidth = W;
    view.imageHeight = H;
    view.camera.setIntrinsics(FOCAL, FOCAL, W * 0.5, H * 0.5);
    view.camera.setPose(
        std::array<double, 9>{I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8]},
        std::array<double, 3>{C[0], C[1], C[2]});
    view.camera.setAxisDirections(1, 1);
    view.camera.setDepthAxisFlipped(false);

    xjw::mvs::SparseCloud sparse;
    for (int py = 36; py <= 54; py += 6)
    {
        for (int px = 45; px <= 75; px += 6)
        {
            const float x = static_cast<float>((px - W * 0.5) * Z / FOCAL);
            const float y = static_cast<float>((py - H * 0.5) * Z / FOCAL);
            sparse.points.push_back({x, y, Z});
        }
    }

    const cv::Mat mask = xjw::mvs::DepthMapGenerator::buildSparseSupportMask({view}, sparse, 0, W, H);

    ASSERT_FALSE(mask.empty());
    EXPECT_GT(cv::countNonZero(mask(cv::Rect(35, 25, 50, 40))), 0);
    EXPECT_EQ(mask.at<uint8_t>(5, 5), 0);

    const float coverage = static_cast<float>(cv::countNonZero(mask)) / static_cast<float>(W * H);
    EXPECT_GT(coverage, 0.05f);
    EXPECT_LT(coverage, 0.90f);
}

TEST(MvsPipelineTest, ProjectedSparseSamplesFeedHintAndSupportReuse)
{
    constexpr int W = 120;
    constexpr int H = 90;
    constexpr double FOCAL = 80.0;
    constexpr float Z = 10.0f;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    xjw::mvs::CameraView view;
    view.imageWidth = W;
    view.imageHeight = H;
    view.camera.setIntrinsics(FOCAL, FOCAL, W * 0.5, H * 0.5);
    view.camera.setPose(
        std::array<double, 9>{I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8]},
        std::array<double, 3>{C[0], C[1], C[2]});
    view.camera.setAxisDirections(1, 1);
    view.camera.setDepthAxisFlipped(false);

    xjw::mvs::SparseCloud sparse;
    std::vector<size_t> indices;
    for (int py = 36; py <= 54; py += 6)
    {
        for (int px = 45; px <= 75; px += 6)
        {
            const float x = static_cast<float>((px - W * 0.5) * Z / FOCAL);
            const float y = static_cast<float>((py - H * 0.5) * Z / FOCAL);
            indices.push_back(sparse.points.size());
            sparse.points.push_back({x, y, Z});
        }
    }
    indices.push_back(sparse.points.size());
    sparse.points.push_back({0.0f, 0.0f, Z * 8.0f});

    const std::vector<xjw::mvs::ProjectedSparseDepthSample> samples =
        xjw::mvs::DepthMapGenerator::collectProjectedSparseDepthSamples(
            sparse, view.positiveDepthModel(), W, H, indices);

    EXPECT_EQ(samples.size(), indices.size() - 1)
        << "The depth outlier should be excluded once before building hint/support rasters.";

    const cv::Mat hint = xjw::mvs::DepthMapGenerator::buildHintDepthFromProjectedSamples(
        0, W / 2, H / 2, samples);
    const cv::Mat support = xjw::mvs::DepthMapGenerator::buildSparseSupportMaskFromProjectedSamples(
        0, W, H, samples);

    ASSERT_FALSE(hint.empty());
    ASSERT_FALSE(support.empty());
    EXPECT_GT(cv::countNonZero(hint > 0), 0);
    EXPECT_GT(cv::countNonZero(support(cv::Rect(35, 25, 50, 40))), 0);
    EXPECT_EQ(support.at<uint8_t>(5, 5), 0);
}

TEST(MvsPipelineTest, SparseSeedDepthOverlayDoesNotPropagateAcrossFineHint)
{
    std::vector<xjw::mvs::ProjectedSparseDepthSample> samples;
    xjw::mvs::ProjectedSparseDepthSample sample;
    sample.uNorm = 0.5f;
    sample.vNorm = 0.5f;
    sample.depth = 10.0f;
    samples.push_back(sample);

    const cv::Mat propagated = xjw::mvs::DepthMapGenerator::buildHintDepthFromProjectedSamples(
        0, 64, 64, samples);
    const cv::Mat seedOnly = xjw::mvs::DepthMapGenerator::buildSparseSeedDepthFromProjectedSamples(
        0, 64, 64, samples);

    ASSERT_FALSE(propagated.empty());
    ASSERT_FALSE(seedOnly.empty());
    EXPECT_LT(cv::countNonZero(seedOnly > 0), cv::countNonZero(propagated > 0))
        << "Fine sparse overlay should stamp local seeds only; full propagation is reserved for coarse hints.";
    EXPECT_FLOAT_EQ(seedOnly.at<float>(32, 32), 10.0f);
    EXPECT_FLOAT_EQ(seedOnly.at<float>(0, 0), 0.0f);
}

TEST(MvsPipelineTest, SparseSupportPriorKeepsDepthAndSoftensConfidence)
{
    cv::Mat depth(3, 3, CV_32F, cv::Scalar(12.0f));
    cv::Mat confidence(3, 3, CV_32F, cv::Scalar(0.8f));
    cv::Mat support(3, 3, CV_8U, cv::Scalar(0));
    support.at<uint8_t>(1, 1) = 255;

    const int beforeValid = cv::countNonZero(depth > 0);

    xjw::mvs::DepthMapGenerator::applySparseSupportPrior(depth, confidence, support, 0);

    EXPECT_EQ(cv::countNonZero(depth > 0), beforeValid)
        << "Sparse support must not hard-clip PatchMatch depth pixels.";
    EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 12.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(1, 1), 12.0f);

    EXPECT_FLOAT_EQ(confidence.at<float>(1, 1), 0.8f);
    EXPECT_GT(confidence.at<float>(0, 0), 0.0f);
    EXPECT_LT(confidence.at<float>(0, 0), 0.8f);
}

TEST(MvsPipelineTest, LocalDepthOutlierFilterRemovesIsolatedDepthSpike)
{
    cv::Mat depth(7, 7, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    depth.at<float>(3, 3) = 30.0f;

    const int removed = xjw::mvs::DepthMapGenerator::removeLocalDepthOutliers(
        depth, confidence, 3, 0.25f, 0.50f, 0);

    EXPECT_EQ(removed, 1);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 3), 0.0f);
    EXPECT_FLOAT_EQ(confidence.at<float>(3, 3), 0.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 2), 10.0f);
}

TEST(MvsPipelineTest, LocalDepthOutlierFilterPreservesSmoothSlope)
{
    cv::Mat depth(7, 7, CV_32F);
    cv::Mat confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    for (int y = 0; y < depth.rows; ++y)
    {
        for (int x = 0; x < depth.cols; ++x)
        {
            depth.at<float>(y, x) = 10.0f + 0.10f * static_cast<float>(x) + 0.05f * static_cast<float>(y);
        }
    }

    const int beforeValid = cv::countNonZero(depth > 0);
    const int removed = xjw::mvs::DepthMapGenerator::removeLocalDepthOutliers(
        depth, confidence, 3, 0.25f, 0.50f, 0);

    EXPECT_EQ(removed, 0);
    EXPECT_EQ(cv::countNonZero(depth > 0), beforeValid);
}

TEST(MvsPipelineTest, FusionDepthPostprocessReportsConfidenceAndLocalOutliers)
{
    cv::Mat depth(7, 7, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(7, 7, CV_32F, cv::Scalar(0.9f));
    depth.at<float>(3, 3) = 30.0f;
    confidence.at<float>(0, 0) = 0.10f;

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.25f;
    config.enableLocalDepthOutlierFilter = true;
    config.localDepthOutlierKernelSize = 3;
    config.localDepthOutlierRelThresh = 0.25f;
    config.maxLocalDepthOutlierRemovalRatio = 0.50f;

    const xjw::mvs::DepthPostProcessStats stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(depth, confidence, config, 0, 4);

    EXPECT_EQ(stats.validBeforePostprocess, 49);
    EXPECT_EQ(stats.validAfterConfidenceFilter, 48);
    EXPECT_EQ(stats.confidenceRemoved, 1);
    EXPECT_EQ(stats.localDepthOutlierRemoved, 1);
    EXPECT_EQ(stats.speckleRemoved, 0);
    EXPECT_EQ(stats.edgeConfidenceRemoved, 0);
    EXPECT_EQ(stats.geomConsistencyRemoved, 0);
    EXPECT_EQ(stats.validAfterPostprocess, 47);
    EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(3, 3), 0.0f);
    EXPECT_FLOAT_EQ(confidence.at<float>(3, 3), 0.0f);
}

TEST(MvsPipelineTest, FusionDepthPostprocessRaisesThresholdForLowConfidenceFullCoverage)
{
    cv::Mat depth(10, 10, CV_32F, cv::Scalar(10.0f));
    cv::Mat confidence(10, 10, CV_32F, cv::Scalar(0.56f));
    confidence.at<float>(4, 4) = 0.80f;
    confidence.at<float>(5, 5) = 0.78f;

    xjw::mvs::FusionConfig config;
    config.confidenceThresh = 0.25f;
    config.enableAdaptiveConfidenceFilter = true;
    config.adaptiveFullCoverageThreshold = 0.95f;
    config.adaptiveLowMeanConfidenceThreshold = 0.65f;
    config.adaptiveStrictConfidenceThreshold = 0.65f;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = false;

    const xjw::mvs::DepthPostProcessStats stats =
        xjw::mvs::DepthMapGenerator::postprocessFusionDepthMap(depth, confidence, config, 0, 4);

    EXPECT_FLOAT_EQ(stats.effectiveConfidenceThreshold, 0.65f);
    EXPECT_EQ(stats.validAfterConfidenceFilter, 2);
    EXPECT_EQ(stats.confidenceRemoved, 98);
    EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(depth.at<float>(4, 4), 10.0f);
}

TEST(MvsPipelineTest, StreamingFirstFrameFusionEstimatesNormalsWhenNormalMapMissing)
{
    constexpr int W = 16;
    constexpr int H = 12;
    constexpr double FOCAL = 40.0;
    constexpr float DEPTH_VAL = 8.0f;

    const double I[9] = {1,0,0,0,1,0,0,0,1};
    const double C[3] = {0,0,0};

    std::vector<xjw::mvs::FusionFrameInput> frames(3);
    for (int i = 0; i < 3; ++i)
    {
        frames[static_cast<size_t>(i)].depthMap = cv::Mat(H, W, CV_32F, cv::Scalar(DEPTH_VAL));
        frames[static_cast<size_t>(i)].cameraModel = makePosCam(FOCAL, FOCAL, W * 0.5, H * 0.5, I, C);
        frames[static_cast<size_t>(i)].imgW = W;
        frames[static_cast<size_t>(i)].imgH = H;
    }
    frames[0].sourceImageIndices = {1, 2};

    xjw::mvs::StereoFusionConfig fcfg;
    fcfg.fuseOnlyFirstFrame = true;
    fcfg.minNumPixels = 3;
    fcfg.checkNumImages = 2;
    fcfg.maxReprojError = 0.5f;
    fcfg.maxDepthError = 0.01f;

    xjw::mvs::DepthMapFusion fusion(fcfg);
    std::vector<xjw::mvs::FusedPoint> pts;
    std::string err;
    const bool ok = fusion.fuse(frames, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    ASSERT_GT(pts.size(), 0u);
    const xjw::mvs::FusedPoint &p = pts[pts.size() / 2];
    const float normalLength = std::sqrt(p.nx * p.nx + p.ny * p.ny + p.nz * p.nz);
    EXPECT_GT(normalLength, 0.90f)
        << "Production dense clouds must not write all-zero normals into PLY output";
}

TEST(MvsPipelineTest, ContentMaskSkipsNearlyFullAerialFrame)
{
    cv::Mat gray(120, 200, CV_8U, cv::Scalar(122));
    cv::Mat mask = xjw::mvs::DepthMapGenerator::buildContentMask(gray);

    EXPECT_TRUE(mask.empty())
        << "Nearly full-content aerial frames should skip the content mask instead of filtering no pixels";
}

TEST(MvsPipelineTest, ContentMaskKeepsRealBlackBorderMask)
{
    cv::Mat gray(120, 200, CV_8U, cv::Scalar(0));
    gray(cv::Rect(35, 25, 130, 70)) = cv::Scalar(125);

    cv::Mat mask = xjw::mvs::DepthMapGenerator::buildContentMask(gray);

    ASSERT_FALSE(mask.empty());
    EXPECT_EQ(mask.at<uint8_t>(5, 5), 0);
    EXPECT_GT(mask.at<uint8_t>(60, 100), 0);

    const float coverage = static_cast<float>(cv::countNonZero(mask)) / static_cast<float>(mask.rows * mask.cols);
    EXPECT_GT(coverage, 0.20f);
    EXPECT_LT(coverage, 0.80f);
}

TEST(MvsPipelineTest, CudaRetryDownsampleIncreasesUntilGpuFriendlyScale)
{
    xjw::mvs::PatchMatchConfig cfg;
    cfg.downsampleFactor = 2;

    cfg = xjw::mvs::DepthMapGenerator::nextCudaRetryPatchMatchConfig(cfg, 6000, 4000);
    EXPECT_EQ(cfg.downsampleFactor, 3);

    cfg = xjw::mvs::DepthMapGenerator::nextCudaRetryPatchMatchConfig(cfg, 6000, 4000);
    EXPECT_EQ(cfg.downsampleFactor, 4);

    cfg = xjw::mvs::DepthMapGenerator::nextCudaRetryPatchMatchConfig(cfg, 6000, 4000);
    EXPECT_EQ(cfg.downsampleFactor, 6);
}

// ---------------------------------------------------------------------------
// Test 3: DenseCloudBuilder CPU 反投影基本正确性
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, DenseCloudBuilderUnprojectBasic)
{
    constexpr int W = 32, H = 24;
    constexpr float DEPTH_VAL = 5.0f;
    constexpr double FOCAL = 30.0;

    cv::Mat depth(H, W, CV_32F, cv::Scalar(DEPTH_VAL));

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0};
    auto cam = makePosCam(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);

    xjw::mvs::DenseCloudOptions opt;
    opt.minDepth = 0.1f;
    opt.maxDepth = 100.0f;

    auto pts = xjw::mvs::DenseCloudBuilder::unproject(depth, cv::Mat(), cam, cv::Mat(), opt);

    ASSERT_EQ(static_cast<int>(pts.size()), W * H);

    // 中心像素应精确落在 Z=DEPTH_VAL
    // 中心像素 (W/2, H/2) 对应 X_cam = 0, Y_cam = 0, Z_cam = DEPTH_VAL
    double cx_pix = W / 2.0 - 0.5; // 像素中心
    double cy_pix = H / 2.0 - 0.5;

    // 找最近中心点
    float minDist = 1e9f;
    xjw::mvs::DensePoint closest{};
    for (auto &p : pts)
    {
        float dx = p.x, dy = p.y;
        float d = std::sqrt(dx*dx + dy*dy);
        if (d < minDist) { minDist = d; closest = p; }
    }
    EXPECT_NEAR(closest.z, DEPTH_VAL, 0.5f)
        << "Center pixel depth should unproject to ~DEPTH_VAL";
    EXPECT_NEAR(closest.x, 0.0f, 0.5f)
        << "Center pixel X should be ~0 in camera space";
}

// ---------------------------------------------------------------------------
// Test 4: PatchMatch → DenseCloudBuilder 端到端，点云非空
// ---------------------------------------------------------------------------
TEST(MvsPipelineTest, PatchMatchToDenseCloudEndToEnd)
{
    constexpr int W = 80, H = 60;
    constexpr double FOCAL = 64.0, BASELINE = 1.0, DISP = 6;

    cv::Mat ref = makeSyntheticGray(W, H);
    cv::Mat src = makeShifted(ref, DISP);
    cv::GaussianBlur(src, src, cv::Size(3,3), 0.0);

    const double I[9]={1,0,0,0,1,0,0,0,1};
    const double C0[3]={0,0,0}, C1[3]={BASELINE,0,0};
    auto refCam = makePosCam(FOCAL, FOCAL, W*0.5, H*0.5, I, C0);
    auto srcCam = makePosCam(FOCAL, FOCAL, W*0.5, H*0.5, I, C1);

    xjw::mvs::PatchMatchConfig cfg;
    cfg.useCuda           = false;
    cfg.downsampleFactor  = 2;
    cfg.patchHalf         = 2;
    cfg.confidenceThresh  = 0.10f;
    cfg.doMedianBlur      = true;
    cfg.medianKernelSize  = 3;
    cfg.doBilateralFilter = false;

    cv::Mat depth;
    bool ok = xjw::mvs::PatchMatchDepthEstimator::estimate(
        ref, {src}, refCam, {srcCam}, 4.0f, 20.0f, cfg, depth);
    ASSERT_TRUE(ok);

    xjw::mvs::DenseCloudOptions opt;
    opt.minDepth = 1.0f;
    opt.maxDepth = 50.0f;
    auto pts = xjw::mvs::DenseCloudBuilder::unproject(depth, cv::Mat(), refCam, cv::Mat(), opt);

    EXPECT_GT(static_cast<int>(pts.size()), 0)
        << "End-to-end CPU MVS pipeline should produce at least one 3D point";

    // 深度应在合理范围内
    for (auto &p : pts)
    {
        EXPECT_GT(p.z, 0.0f);
        EXPECT_LT(p.z, 50.0f);
    }
}

// =============================================================================
// MVS 管道集成测试
// 覆盖：CPU PatchMatch 深度估计 → DepthMapFusion 融合 → DenseCloudBuilder 反投影
// =============================================================================

#include <gtest/gtest.h>

#include "PatchMatchCUDA.h"
#include "DepthMapFusion.h"
#include "DenseCloudBuilder.h"
#include "Camera.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>

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
    std::vector<xjw::mvs::DensePoint> pts;
    std::string err;
    bool ok = fusion.fuse({fr0, fr1}, pts, nullptr, &err);

    ASSERT_TRUE(ok) << err;
    EXPECT_GT(static_cast<int>(pts.size()), 0)
        << "DepthMapFusion should produce at least one fused point";
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

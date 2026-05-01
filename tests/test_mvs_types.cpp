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
#include "MvsTypes.h"
#include "Camera.h"

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

    // 核心参数已收紧
    EXPECT_EQ(cfg.patchHalf, 7)
        << "patchHalf should be 7 (15x15 window) for robust matching";
    EXPECT_EQ(cfg.numIterations, 16);
    EXPECT_EQ(cfg.numSourceViews, 4);
    EXPECT_FLOAT_EQ(cfg.confidenceThresh, 0.30f)
        << "confidence threshold should be 0.30";
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

// ─── FusionConfig 参数验证 ──────────────────────────────────────

TEST(FusionConfigTest, DefaultParametersOptimized)
{
    FusionConfig cfg;

    // 收紧后的融合参数
    EXPECT_EQ(cfg.minConsistentViews, 2);
    EXPECT_FLOAT_EQ(cfg.relDepthThresh, 0.05f)
        << "Relative depth threshold should be 0.05 (tightened from 0.08)";
    EXPECT_FLOAT_EQ(cfg.pixelThresh, 2.0f)
        << "Pixel threshold should be 2.0 (tightened from 4.0)";
    EXPECT_FLOAT_EQ(cfg.confidenceThresh, 0.25f)
        << "Fusion confidence threshold should be 0.25";
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

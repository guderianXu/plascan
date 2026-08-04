// ============================================================
// test_sfm_params.cpp — SfM 参数收紧验证
//
// 验证优化后的 SfM 管线参数确实比原始值更严格，
// 以确保从源头减少外点。
// ============================================================

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "triangulation/Triangulator.h"
#include "pipeline/IncrementalSfm.h"
#include "pose/PnpSolver.h"
#include "BundleAdjust.h"
#include "filtering/SparsePointCloudProcessor.h"
#include "Intersection.h"

using namespace xjw;

namespace
{

struct LowRatioPnpCase
{
    std::vector<std::array<double, 3>> worldPoints;
    std::vector<std::array<double, 2>> imagePoints;
    double fu = 768.0;
    double fv = 768.0;
    double cu = 320.0;
    double cv = 240.0;
};

LowRatioPnpCase makeLowRatioPnpCase()
{
    LowRatioPnpCase data;
    data.worldPoints.reserve(85);
    data.imagePoints.reserve(85);

    for (int i = 0; i < 20; ++i)
    {
        const double x = (static_cast<double>(i % 5) - 2.0) * 0.35;
        const double y = (static_cast<double>(i / 5) - 1.5) * 0.28;
        const double z = 5.0 + static_cast<double>(i % 4) * 0.2;
        data.worldPoints.push_back({{x, y, z}});
        data.imagePoints.push_back({{
            data.fu * x / z + data.cu + (i % 2 == 0 ? 0.2 : -0.2),
            data.fv * y / z + data.cv + (i % 3 == 0 ? -0.15 : 0.15),
        }});
    }

    for (int i = 0; i < 65; ++i)
    {
        const double x = -2.0 + static_cast<double>(i % 13) * 0.31;
        const double y = -1.8 + static_cast<double>(i % 11) * 0.29;
        const double z = 4.5 + static_cast<double>(i % 7) * 0.35;
        data.worldPoints.push_back({{x, y, z}});
        data.imagePoints.push_back({{
            40.0 + static_cast<double>((i * 37) % 560),
            30.0 + static_cast<double>((i * 53) % 420),
        }});
    }

    return data;
}

std::array<double, 2> projectWithCameraCenter(const std::array<double, 3> &point,
                                              const std::array<double, 3> &cameraCenter,
                                              double fu,
                                              double fv,
                                              double cu,
                                              double cv)
{
    const double x = point[0] - cameraCenter[0];
    const double y = point[1] - cameraCenter[1];
    const double z = point[2] - cameraCenter[2];
    return {{fu * x / z + cu, fv * y / z + cv}};
}

LowRatioPnpCase makeSmallSamplePnpCase(int inlierCount, int outlierCount)
{
    LowRatioPnpCase data;
    data.worldPoints.reserve(static_cast<std::size_t>(inlierCount + outlierCount));
    data.imagePoints.reserve(static_cast<std::size_t>(inlierCount + outlierCount));

    for (int i = 0; i < inlierCount + outlierCount; ++i)
    {
        const double x = (static_cast<double>(i % 4) - 1.5) * 0.45;
        const double y = (static_cast<double>(i / 4) - 1.5) * 0.35;
        const double z = 4.8 + static_cast<double>(i % 3) * 0.25;
        const std::array<double, 3> point{{x, y, z}};
        data.worldPoints.push_back(point);
        if (i < inlierCount)
        {
            data.imagePoints.push_back(projectWithCameraCenter(
                point, {{0.0, 0.0, 0.0}}, data.fu, data.fv, data.cu, data.cv));
        }
        else
        {
            data.imagePoints.push_back({{
                45.0 + static_cast<double>((i * 137) % 510),
                35.0 + static_cast<double>((i * 173) % 390),
            }});
        }
    }
    return data;
}

} // namespace

// ─── Triangulator 参数验证 ──────────────────────────────────────

TEST(TriangulatorParamsTest, DefaultsAreTightened)
{
    TriangulatorOptions opts;

    // 最小三角化角度已从 1.5° 收紧到 2.0°
    EXPECT_GE(opts.minTriAngle, 2.0)
        << "minTriAngle should be >= 2.0 degrees to reject poorly-conditioned points";

    // 最大重投影误差已从 4.0 收紧到 2.5
    EXPECT_LE(opts.maxReprojError, 2.5)
        << "maxReprojError should be <= 2.5 pixels";

    EXPECT_LE(opts.continueMaxReprojError, 2.5);
    EXPECT_LE(opts.completeMaxReprojError, 2.5);
}

// ─── IncrementalSfm 参数验证 ────────────────────────────────────

TEST(IncrementalSfmParamsTest, FilterParamsTightened)
{
    IncrementalSfmOptions opts;

    // 过滤重投影误差
    EXPECT_LE(opts.filterMaxReprojError, 2.0)
        << "filterMaxReprojError should be <= 2.0";

    // 过滤三角化角度
    EXPECT_GE(opts.filterMinTriAngle, 2.0)
        << "filterMinTriAngle should be >= 2.0";

    // 短轨迹过滤
    EXPECT_GE(opts.filterMinTrackLen, 2)
        << "filterMinTrackLen should be >= 2";
}

TEST(IncrementalSfmParamsTest, BAIntervals)
{
    IncrementalSfmOptions opts;

    // BA 间隔应合理
    EXPECT_GT(opts.localBAInterval, 0);
    EXPECT_GT(opts.globalBAInterval, 0);
    EXPECT_GT(opts.localBANumImages, 0);
}

TEST(IncrementalSfmParamsTest, BracketedSequencePnpRequiresAbsoluteGeometricSupport)
{
    IncrementalSfmOptions opts;

    EXPECT_TRUE(opts.allowBracketedSequencePnpRelaxation);
    EXPECT_TRUE(opts.allowOneSidedSequencePoseRecovery);
    EXPECT_GE(opts.oneSidedSequencePosePrefilterMaxReprojError, 96.0);
    EXPECT_GE(opts.oneSidedSequencePnpMinInliers, 12);
    EXPECT_LT(opts.oneSidedSequencePnpMinInliers, opts.bracketedSequencePnpMinInliers);
    EXPECT_GE(opts.oneSidedSequencePnpMinInlierRatio, 0.025);
    EXPECT_LE(opts.oneSidedSequencePnpMinInlierRatio, 0.10);
    EXPECT_GE(opts.bracketedSequencePnpMinInliers, 28);
    EXPECT_GE(opts.bracketedSequencePnpMinInlierRatio, 0.02);
    EXPECT_LE(opts.bracketedSequencePnpMinInlierRatio, 0.05);
}

TEST(IncrementalSfmParamsTest, FinalGlobalBaRetriesUnregisteredImages)
{
    const xjw::IncrementalSfmOptions opts;

    EXPECT_TRUE(opts.retryUnregisteredAfterFinalBA);
    EXPECT_GE(opts.maxFinalRegistrationRetryPasses, 1);
    EXPECT_LE(opts.maxFinalRegistrationRetryPasses, 3);
}

// ─── PnP 参数验证 ──────────────────────────────────────────────

TEST(PnpParamsTest, MaxReprojErrorTightened)
{
    PnpOptions opts;

    // 从 8.0 收紧到 4.0
    EXPECT_LE(opts.maxReprojError, 4.0)
        << "PnP maxReprojError should be <= 4.0 to reduce registration outliers";
    EXPECT_LE(opts.initialPosePrefilterMaxReprojError, 48.0);
}

TEST(PnpParamsTest, SmallSampleAcceptsTwelveFullyConsistentCorrespondences)
{
    const LowRatioPnpCase data = makeSmallSamplePnpCase(12, 0);
    PnpOptions opts;
    opts.minNumInliers = 12;
    opts.smallSampleThreshold = 20;
    opts.smallSampleMinInlierRatio = 0.80;

    const PnpResult result = PnpSolver::solve(data.worldPoints,
                                              data.imagePoints,
                                              data.fu,
                                              data.fv,
                                              data.cu,
                                              data.cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.numInliers, 12);
}

TEST(PnpParamsTest, SmallSampleRejectsTwelveOfSixteenInliers)
{
    const LowRatioPnpCase data = makeSmallSamplePnpCase(12, 4);
    PnpOptions opts;
    opts.maxIterations = 5000;
    opts.minNumInliers = 12;
    opts.minInlierRatio = 0.25;
    opts.allowRelaxedInlierRatio = true;
    opts.relaxedMinNumInliers = 12;
    opts.relaxedMinInlierRatio = 0.05;
    opts.smallSampleThreshold = 20;
    opts.smallSampleMinInlierRatio = 0.80;

    const PnpResult result = PnpSolver::solve(data.worldPoints,
                                              data.imagePoints,
                                              data.fu,
                                              data.fv,
                                              data.cu,
                                              data.cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.numInliers, 12);
    EXPECT_NEAR(result.inlierRatio, 0.75, 1e-6);
}

TEST(PnpParamsTest, InitialPoseGuessUsesCameraCenterConvention)
{
    std::vector<std::array<double, 3>> worldPoints;
    std::vector<std::array<double, 2>> imagePoints;
    const std::array<double, 3> trueCenter{{0.25, -0.15, 0.80}};
    const double fu = 900.0;
    const double fv = 910.0;
    const double cu = 320.0;
    const double cv = 240.0;

    for (int i = 0; i < 30; ++i)
    {
        const double x = -1.2 + static_cast<double>(i % 6) * 0.45;
        const double y = -0.8 + static_cast<double>(i / 6) * 0.35;
        const double z = 4.5 + static_cast<double>(i % 5) * 0.25;
        const std::array<double, 3> point{{x, y, z}};
        worldPoints.push_back(point);
        imagePoints.push_back(projectWithCameraCenter(point, trueCenter, fu, fv, cu, cv));
    }

    PnpOptions opts;
    opts.minNumInliers = 12;
    opts.maxIterations = 1000;
    opts.useInitialPose = true;
    opts.initialCameraCenter = trueCenter;
    opts.initialCameraToWorldRotation = {{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0}};

    const PnpResult result = PnpSolver::solve(worldPoints,
                                              imagePoints,
                                              fu,
                                              fv,
                                              cu,
                                              cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    ASSERT_TRUE(result.success);
    EXPECT_GE(result.numInliers, 25);
    EXPECT_NEAR(result.C[0], trueCenter[0], 1e-4);
    EXPECT_NEAR(result.C[1], trueCenter[1], 1e-4);
    EXPECT_NEAR(result.C[2], trueCenter[2], 1e-4);
}

TEST(PnpParamsTest, InitialPosePrefilterRecoversLowRatioInliers)
{
    const LowRatioPnpCase data = makeLowRatioPnpCase();
    PnpOptions opts;
    opts.maxIterations = 100;
    opts.minNumInliers = 10;
    opts.allowRelaxedInlierRatio = true;
    opts.relaxedMinInlierRatio = 0.20;
    opts.useInitialPose = true;
    opts.useInitialPosePrefilter = true;
    opts.initialPosePrefilterMaxReprojError = 12.0;
    opts.initialPosePrefilterMinCandidates = 10;
    opts.initialCameraCenter = {{0.0, 0.0, 0.0}};
    opts.initialCameraToWorldRotation = {{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0}};

    const PnpResult result = PnpSolver::solve(data.worldPoints,
                                              data.imagePoints,
                                              data.fu,
                                              data.fv,
                                              data.cu,
                                              data.cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numInliers, 18);
    EXPECT_EQ(result.inlierMask.size(), data.worldPoints.size());
    EXPECT_TRUE(result.usedInitialPosePrefilter);
    EXPECT_GE(result.prefilterCandidateCount, result.numInliers);
    EXPECT_LT(result.prefilterCandidateCount, static_cast<int>(data.worldPoints.size()));
}

TEST(PnpParamsTest, InitialPosePrefilterReportsCandidatesBelowActivationThreshold)
{
    const LowRatioPnpCase data = makeLowRatioPnpCase();
    PnpOptions opts;
    opts.useInitialPose = true;
    opts.useInitialPosePrefilter = true;
    opts.initialPosePrefilterMaxReprojError = 12.0;
    opts.initialPosePrefilterMinCandidates = 1000;
    opts.initialCameraCenter = {{0.0, 0.0, 0.0}};
    opts.initialCameraToWorldRotation = {{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0}};

    const PnpResult result = PnpSolver::solve(data.worldPoints,
                                              data.imagePoints,
                                              data.fu,
                                              data.fv,
                                              data.cu,
                                              data.cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    EXPECT_FALSE(result.usedInitialPosePrefilter);
    EXPECT_GT(result.prefilterCandidateCount, 0);
    EXPECT_LT(result.prefilterCandidateCount, opts.initialPosePrefilterMinCandidates);
}

TEST(PnpParamsTest, AcceptsLowRatioWhenAbsoluteInlierSupportIsEnough)
{
    const LowRatioPnpCase data = makeLowRatioPnpCase();

    PnpOptions opts;
    opts.maxIterations = 50000;
    opts.minNumInliers = 10;
    opts.allowRelaxedInlierRatio = true;

    const PnpResult result = PnpSolver::solve(data.worldPoints,
                                              data.imagePoints,
                                              data.fu,
                                              data.fv,
                                              data.cu,
                                              data.cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numInliers, 18);
    EXPECT_LT(static_cast<double>(20) / static_cast<double>(data.worldPoints.size()), 0.25);
}

TEST(PnpParamsTest, RelaxedInlierRatioThresholdIsConfigurable)
{
    const LowRatioPnpCase data = makeLowRatioPnpCase();

    PnpOptions strictRelaxedOpts;
    strictRelaxedOpts.maxIterations = 50000;
    strictRelaxedOpts.minNumInliers = 10;
    strictRelaxedOpts.allowRelaxedInlierRatio = true;
    strictRelaxedOpts.relaxedMinInlierRatio = 0.24;

    const PnpResult strictRelaxed = PnpSolver::solve(data.worldPoints,
                                                     data.imagePoints,
                                                     data.fu,
                                                     data.fv,
                                                     data.cu,
                                                     data.cv,
                                                     1,
                                                     1,
                                                     false,
                                                     strictRelaxedOpts);

    EXPECT_FALSE(strictRelaxed.success);
    EXPECT_GE(strictRelaxed.numInliers, 18);
    EXPECT_LT(strictRelaxed.inlierRatio, strictRelaxedOpts.relaxedMinInlierRatio);

    PnpOptions looseRelaxedOpts = strictRelaxedOpts;
    looseRelaxedOpts.relaxedMinInlierRatio = 0.20;

    const PnpResult looseRelaxed = PnpSolver::solve(data.worldPoints,
                                                    data.imagePoints,
                                                    data.fu,
                                                    data.fv,
                                                    data.cu,
                                                    data.cv,
                                                    1,
                                                    1,
                                                    false,
                                                    looseRelaxedOpts);

    EXPECT_TRUE(looseRelaxed.success);
    EXPECT_GE(looseRelaxed.inlierRatio, looseRelaxedOpts.relaxedMinInlierRatio);
    EXPECT_LT(looseRelaxed.inlierRatio, looseRelaxedOpts.minInlierRatio);
}

TEST(PnpParamsTest, RejectsLowRatioByDefault)
{
    const LowRatioPnpCase data = makeLowRatioPnpCase();

    PnpOptions opts;
    opts.maxIterations = 50000;
    opts.minNumInliers = 10;

    const PnpResult result = PnpSolver::solve(data.worldPoints,
                                              data.imagePoints,
                                              data.fu,
                                              data.fv,
                                              data.cu,
                                              data.cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    EXPECT_FALSE(result.success);
    EXPECT_GE(result.numInliers, 18);
    EXPECT_LT(result.inlierRatio, opts.minInlierRatio);
}

TEST(PnpParamsTest, DeterministicSeedIsIndependentOfExternalOpenCvRngState)
{
    const LowRatioPnpCase data = makeLowRatioPnpCase();

    PnpOptions opts;
    opts.maxIterations = 50000;
    opts.minNumInliers = 10;
    opts.allowRelaxedInlierRatio = true;
    opts.ransacSeed = 20260711;

    cv::setRNGSeed(17);
    const PnpResult first = PnpSolver::solve(data.worldPoints,
                                             data.imagePoints,
                                             data.fu,
                                             data.fv,
                                             data.cu,
                                             data.cv,
                                             1,
                                             1,
                                             false,
                                             opts);

    cv::setRNGSeed(918273);
    for (int i = 0; i < 1000; ++i)
    {
        static_cast<void>(cv::theRNG().next());
    }
    const PnpResult second = PnpSolver::solve(data.worldPoints,
                                              data.imagePoints,
                                              data.fu,
                                              data.fv,
                                              data.cu,
                                              data.cv,
                                              1,
                                              1,
                                              false,
                                              opts);

    ASSERT_EQ(first.success, second.success);
    ASSERT_EQ(first.numInliers, second.numInliers);
    EXPECT_EQ(first.inlierMask, second.inlierMask);
    for (std::size_t i = 0; i < first.C.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(first.C[i], second.C[i]);
    }
    for (std::size_t i = 0; i < first.R.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(first.R[i], second.R[i]);
    }
}

TEST(PnpParamsTest, SolveWithCameraHonorsBrownConradyDistortion)
{
    const std::array<double, 9> identity{{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    }};
    const std::array<double, 3> trueCenter{{0.25, -0.12, 0.35}};

    Camera camera;
    camera.setIntrinsics(820.0, 790.0, 640.0, 480.0);
    camera.setDistortion(-0.32, 0.11, -0.014, 0.004, -0.003);
    camera.setPose(identity, trueCenter);

    std::vector<std::array<double, 3>> worldPoints;
    std::vector<std::array<double, 2>> imagePoints;
    worldPoints.reserve(63);
    imagePoints.reserve(63);
    for (int row = 0; row < 7; ++row)
    {
        for (int column = 0; column < 9; ++column)
        {
            const double normalizedX = -0.62 + 0.155 * static_cast<double>(column);
            const double normalizedY = -0.45 + 0.15 * static_cast<double>(row);
            const double depth = 3.2 + 0.17 * static_cast<double>((row * 3 + column) % 5);
            const std::array<double, 3> world{{
                trueCenter[0] + normalizedX * depth,
                trueCenter[1] + normalizedY * depth,
                trueCenter[2] + depth,
            }};
            double pixel[2] = {0.0, 0.0};
            ASSERT_TRUE(camera.projectWorldPoint(world.data(), pixel));
            worldPoints.push_back(world);
            imagePoints.push_back({{pixel[0], pixel[1]}});
        }
    }

    PnpOptions options;
    options.maxReprojError = 0.05;
    options.maxIterations = 2000;
    options.minNumInliers = 50;
    options.minInlierRatio = 0.95;
    options.smallSampleThreshold = 0;
    options.useInitialPose = true;
    options.initialCameraToWorldRotation = identity;
    options.initialCameraCenter = trueCenter;
    options.ransacSeed = 20260730;

    const PnpResult pinholeResult =
        PnpSolver::solve(worldPoints, imagePoints,
                         camera.focalX(), camera.focalY(),
                         camera.principalX(), camera.principalY(),
                         camera.uAxisSign(), camera.vAxisSign(),
                         camera.depthAxisFlipped(), options);
    const PnpResult result =
        PnpSolver::solveWithCamera(worldPoints, imagePoints, camera, options);

    EXPECT_FALSE(pinholeResult.success)
        << "Synthetic case must distinguish the calibrated distortion path from the pinhole overload";
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numInliers, static_cast<int>(worldPoints.size()));
    for (std::size_t index = 0; index < trueCenter.size(); ++index)
    {
        EXPECT_NEAR(result.C[index], trueCenter[index], 1.0e-6);
    }
    for (std::size_t index = 0; index < identity.size(); ++index)
    {
        EXPECT_NEAR(result.R[index], identity[index], 1.0e-6);
    }

    Camera recovered = camera;
    recovered.setPose(result.R, result.C);
    for (std::size_t index = 0; index < worldPoints.size(); ++index)
    {
        double pixel[2] = {0.0, 0.0};
        ASSERT_TRUE(recovered.projectWorldPoint(worldPoints[index].data(), pixel));
        EXPECT_NEAR(pixel[0], imagePoints[index][0], 1.0e-4);
        EXPECT_NEAR(pixel[1], imagePoints[index][1], 1.0e-4);
    }
}

TEST(IntersectionDistortionTest, RecoversWorldPointFromDistortedPixels)
{
    const std::array<double, 9> identity{{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    }};
    Camera camera1;
    camera1.setIntrinsics(900.0, 875.0, 640.0, 480.0);
    camera1.setDistortion(-0.28, 0.09, -0.012, 0.003, -0.004);
    camera1.setPose(identity, {{-1.0, 0.10, 0.0}});

    Camera camera2 = camera1;
    camera2.setPose(identity, {{1.0, -0.05, 0.10}});

    const std::array<double, 3> expectedPoint{{0.75, -0.55, 3.70}};
    double pixel1[2] = {0.0, 0.0};
    double pixel2[2] = {0.0, 0.0};
    ASSERT_TRUE(camera1.projectWorldPoint(expectedPoint.data(), pixel1));
    ASSERT_TRUE(camera2.projectWorldPoint(expectedPoint.data(), pixel2));

    const Intersection::Result result =
        Intersection::intersectPair(camera1, pixel1[0], pixel1[1],
                                    camera2, pixel2[0], pixel2[1]);

    ASSERT_TRUE(result.valid);
    for (std::size_t index = 0; index < expectedPoint.size(); ++index)
    {
        EXPECT_NEAR(result.point[index], expectedPoint[index], 1.0e-7);
    }
    EXPECT_LT(result.ray_miss_distance, 1.0e-8);
    EXPECT_LT(result.reproj_error_rms, 1.0e-5);
}

// ─── BundleAdjust 参数验证 ──────────────────────────────────────

TEST(BAParamsTest, FilterReprojErrorTightened)
{
    BAOptions opts;

    // 从 4.0 收紧到 2.5
    EXPECT_LE(opts.filterMaxReprojError, 2.5)
        << "BA filterMaxReprojError should be <= 2.5";
}

// ─── SparsePointCloudProcessor 选项默认值验证 ───────────────────

TEST(SparsePointCloudFilterOptionsTest, DefaultsReasonable)
{
    SparsePointCloudFilterOptions opts;

    // 所有过滤器默认启用
    EXPECT_TRUE(opts.filterByReprojError);
    EXPECT_TRUE(opts.filterByTrackLen);
    EXPECT_TRUE(opts.filterByTriAngle);
    EXPECT_TRUE(opts.filterByStatistical);

    // 阈值合理
    EXPECT_LE(opts.maxReprojError, 2.5);
    EXPECT_GE(opts.minTrackLen, 3);
    EXPECT_GE(opts.minTriAngleDeg, 2.0);
    EXPECT_GE(opts.statK, 10);
    EXPECT_GT(opts.statStdDevMul, 0.0);
}

// ─── 参数一致性验证 ────────────────────────────────────────────

TEST(ParameterConsistencyTest, TriangulatorAndSparsePointFilterAligned)
{
    TriangulatorOptions triOpts;
    SparsePointCloudFilterOptions filterOpts;

    // SfM 过滤器的阈值应不大于三角化器的阈值
    // 否则过滤器无法清理三角化器遗漏的点
    EXPECT_LE(filterOpts.maxReprojError, triOpts.maxReprojError);
    EXPECT_GE(filterOpts.minTriAngleDeg, triOpts.minTriAngle);
}

// ─── 新增选项参数验证 ────────────────────────────────────────────

TEST(IncrementalSfmParamsTest, ChiralityThresholdSofterThanInlierThreshold)
{
    IncrementalSfmOptions opts;

    // chirality 阈值应独立于 initMinNumInliers，且更宽松
    EXPECT_GT(opts.initMinChiralityInliers, 0)
        << "chirality threshold should be positive";
    EXPECT_LE(opts.initMinChiralityInliers, opts.initMinNumInliers)
        << "chirality threshold should be softer (<=) than main inlier threshold";
}

TEST(IncrementalSfmParamsTest, MaxInitPairCandidatesReasonable)
{
    IncrementalSfmOptions opts;

    // 多候选重试策略需要至少 1 个候选
    EXPECT_GE(opts.maxInitPairCandidates, 1)
        << "Should try at least 1 candidate pair";
    EXPECT_LE(opts.maxInitPairCandidates, 50)
        << "Candidate limit should be bounded";
    EXPECT_GE(opts.maxInitPairCandidates, 5)
        << "Should try multiple candidates for robustness";
}

TEST(IncrementalSfmParamsTest, IterativeBARoundsReasonable)
{
    IncrementalSfmOptions opts;

    EXPECT_GE(opts.iterativeBARounds, 1)
        << "At least 1 round of iterative BA needed";
    EXPECT_LE(opts.iterativeBARounds, 10)
        << "iterativeBARounds should be bounded";
}

TEST(IncrementalSfmParamsTest, NegativeDepthFilterEnabled)
{
    IncrementalSfmOptions opts;

    EXPECT_TRUE(opts.filterNegativeDepth)
        << "Negative depth filtering should be enabled by default";
}

TEST(IncrementalSfmParamsTest, KnownPoseBaUsesSoftPriorsAndKeepsIntrinsicsFixed)
{
    IncrementalSfmOptions opts;

    EXPECT_TRUE(opts.refineKnownCameraPoseWithSoftPrior)
        << "Known external orientations should be soft priors, not hard-fixed poses";
    EXPECT_TRUE(opts.keepIntrinsicsFixedInKnownPoseBa)
        << "Aerial weak-geometry BA should not release intrinsics in the first pass";
    EXPECT_GT(opts.knownPosePriorPositionSigmaScale, 0.0)
        << "Soft pose prior needs a positive adaptive sigma scale";
    EXPECT_GT(opts.knownPosePriorRotationSigmaDegrees, 0.0)
        << "Soft pose prior needs a positive rotation sigma";
}

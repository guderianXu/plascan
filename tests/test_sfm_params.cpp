// ============================================================
// test_sfm_params.cpp — SfM 参数收紧验证
//
// 验证优化后的 SfM 管线参数确实比原始值更严格，
// 以确保从源头减少外点。
// ============================================================

#include <gtest/gtest.h>
#include "triangulation/Triangulator.h"
#include "pipeline/IncrementalSfm.h"
#include "pose/PnpSolver.h"
#include "BundleAdjust.h"
#include "filtering/SfmPointCloudFilter.h"

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

// ─── PnP 参数验证 ──────────────────────────────────────────────

TEST(PnpParamsTest, MaxReprojErrorTightened)
{
    PnpOptions opts;

    // 从 8.0 收紧到 4.0
    EXPECT_LE(opts.maxReprojError, 4.0)
        << "PnP maxReprojError should be <= 4.0 to reduce registration outliers";
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

// ─── BundleAdjust 参数验证 ──────────────────────────────────────

TEST(BAParamsTest, FilterReprojErrorTightened)
{
    BAOptions opts;

    // 从 4.0 收紧到 2.5
    EXPECT_LE(opts.filterMaxReprojError, 2.5)
        << "BA filterMaxReprojError should be <= 2.5";
}

// ─── SfmPointCloudFilter 选项默认值验证 ──────────────────────────

TEST(SfmFilterOptionsTest, DefaultsReasonable)
{
    SfmPointCloudFilterOptions opts;

    // 所有过滤器默认启用
    EXPECT_TRUE(opts.filterByReprojError);
    EXPECT_TRUE(opts.filterByTrackLen);
    EXPECT_TRUE(opts.filterByTriAngle);
    EXPECT_TRUE(opts.filterByStatistical);

    // 阈值合理
    EXPECT_LE(opts.maxReprojError, 2.0);
    EXPECT_GE(opts.minTrackLen, 3);
    EXPECT_GE(opts.minTriAngleDeg, 2.0);
    EXPECT_GE(opts.statK, 10);
    EXPECT_GT(opts.statStdDevMul, 0.0);
}

// ─── 参数一致性验证 ────────────────────────────────────────────

TEST(ParameterConsistencyTest, TriangulatorAndSfmFilterAligned)
{
    TriangulatorOptions triOpts;
    SfmPointCloudFilterOptions filterOpts;

    // SfM 过滤器的阈值应不大于三角化器的阈值
    // 否则过滤器无法清理三角化器遗漏的点
    EXPECT_LE(filterOpts.maxReprojError, triOpts.maxReprojError);
    EXPECT_GE(filterOpts.minTriAngleDeg, triOpts.minTriAngle);
}

TEST(ParameterConsistencyTest, SfmOptionsAndFilterAligned)
{
    IncrementalSfmOptions sfmOpts;
    SfmPointCloudFilterOptions filterOpts;

    // SfM 管线内部过滤与外部点云过滤器参数应一致
    EXPECT_LE(filterOpts.maxReprojError, sfmOpts.filterMaxReprojError);
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

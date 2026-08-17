#include "DepthTsdfRecoveryTransaction.h"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace
{

xjw::mesh::MeshTopologyQualityStatistics baselineStatistics()
{
    xjw::mesh::MeshTopologyQualityStatistics statistics;
    statistics.validFaceCount = 100;
    statistics.boundaryEdgeCount = 20;
    statistics.nonManifoldEdgeCount = 2;
    statistics.nonManifoldVertexCount = 3;
    statistics.componentCount = 3;
    statistics.topologicalComplexity = 12;
    statistics.largestComponentFaceRatio = 0.70;
    statistics.extremeAspectFaceRatio = 0.020;
    return statistics;
}

xjw::mesh::MeshTopologyQualityStatistics improvingCandidate()
{
    auto statistics = baselineStatistics();
    statistics.validFaceCount = 108;
    statistics.boundaryEdgeCount = 19;
    statistics.nonManifoldVertexCount = 2;
    statistics.componentCount = 2;
    statistics.topologicalComplexity = 11;
    statistics.largestComponentFaceRatio = 0.72;
    statistics.extremeAspectFaceRatio = 0.019;
    return statistics;
}

} // namespace

TEST(DepthTsdfRecoveryTransactionTest, AcceptsStrictBoundaryImprovementWithoutRegression)
{
    const auto evaluation = xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baselineStatistics(), improvingCandidate());

    EXPECT_TRUE(evaluation.accepted);
    EXPECT_EQ(
        evaluation.rejectionFlags,
        xjw::mesh::DepthTsdfRecoveryTransactionAccepted);
    EXPECT_TRUE(evaluation.reason.isEmpty());
    EXPECT_TRUE(evaluation.diagnostics.value(QStringLiteral(
        "depth_tsdf_recovery_transaction_accepted")).toBool());
}

TEST(DepthTsdfRecoveryTransactionTest, RejectsWhenBoundaryDoesNotStrictlyDecrease)
{
    auto candidate = improvingCandidate();
    candidate.boundaryEdgeCount = baselineStatistics().boundaryEdgeCount;

    const auto evaluation = xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baselineStatistics(), candidate);

    EXPECT_FALSE(evaluation.accepted);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionBoundaryNotReduced,
        0U);
}

TEST(DepthTsdfRecoveryTransactionTest, ReportsEveryProtectedMetricRegression)
{
    const auto baseline = baselineStatistics();
    auto candidate = improvingCandidate();
    candidate.componentCount = baseline.componentCount + 1;
    candidate.nonManifoldEdgeCount = baseline.nonManifoldEdgeCount + 1;
    candidate.nonManifoldVertexCount = baseline.nonManifoldVertexCount + 1;
    candidate.topologicalComplexity = baseline.topologicalComplexity + 1;
    candidate.largestComponentFaceRatio =
        baseline.largestComponentFaceRatio - 0.01;
    candidate.extremeAspectFaceRatio =
        baseline.extremeAspectFaceRatio + 0.001;

    const auto evaluation = xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baseline, candidate);

    EXPECT_FALSE(evaluation.accepted);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionComponentGrowth,
        0U);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionNonManifoldEdgeGrowth,
        0U);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionNonManifoldVertexGrowth,
        0U);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionTopologicalComplexityGrowth,
        0U);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionLargestComponentRatioRegression,
        0U);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionExtremeAspectRatioRegression,
        0U);
}

TEST(DepthTsdfRecoveryTransactionTest, RatioToleranceOnlyAppliesToFloatingMetrics)
{
    const auto baseline = baselineStatistics();
    auto candidate = improvingCandidate();
    candidate.largestComponentFaceRatio =
        baseline.largestComponentFaceRatio - 5.0e-7;
    candidate.extremeAspectFaceRatio =
        baseline.extremeAspectFaceRatio + 5.0e-7;
    xjw::mesh::DepthTsdfRecoveryTransactionOptions options;
    options.ratioTolerance = 1.0e-6;

    EXPECT_TRUE(xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baseline, candidate, options).accepted);

    candidate.boundaryEdgeCount = baseline.boundaryEdgeCount;
    EXPECT_FALSE(xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baseline, candidate, options).accepted);
}

TEST(DepthTsdfRecoveryTransactionTest, InvalidStatisticsFailClosed)
{
    auto baseline = baselineStatistics();
    auto candidate = improvingCandidate();
    baseline.validFaceCount = 0;
    candidate.extremeAspectFaceRatio =
        std::numeric_limits<double>::quiet_NaN();

    const auto evaluation = xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baseline, candidate);

    EXPECT_FALSE(evaluation.accepted);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionInvalidBaseline,
        0U);
    EXPECT_NE(
        evaluation.rejectionFlags &
            xjw::mesh::DepthTsdfRecoveryTransactionInvalidCandidate,
        0U);
}

TEST(DepthTsdfRecoveryTransactionTest, RejectedCandidateRestoresBaselineSupport)
{
    auto candidate = improvingCandidate();
    candidate.boundaryEdgeCount = baselineStatistics().boundaryEdgeCount;
    const auto evaluation = xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baselineStatistics(), candidate);
    const std::vector<std::uint8_t> baseline_support{1, 0, 1, 0};
    std::vector<std::uint8_t> candidate_support{1, 1, 1, 1};

    EXPECT_FALSE(xjw::mesh::commitDepthTsdfRecoveryTransaction(
        evaluation, baseline_support, &candidate_support));
    EXPECT_EQ(candidate_support, baseline_support);
}

TEST(DepthTsdfRecoveryTransactionTest, AcceptedCandidateKeepsCandidateSupport)
{
    const auto evaluation = xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baselineStatistics(), improvingCandidate());
    const std::vector<std::uint8_t> baseline_support{1, 0, 1, 0};
    const std::vector<std::uint8_t> expected_candidate{1, 1, 1, 1};
    std::vector<std::uint8_t> candidate_support = expected_candidate;

    EXPECT_TRUE(xjw::mesh::commitDepthTsdfRecoveryTransaction(
        evaluation, baseline_support, &candidate_support));
    EXPECT_EQ(candidate_support, expected_candidate);
}

TEST(DepthTsdfRecoveryTransactionTest, SupportSizeMismatchFailsClosed)
{
    const auto evaluation = xjw::mesh::evaluateDepthTsdfRecoveryTransaction(
        baselineStatistics(), improvingCandidate());
    const std::vector<std::uint8_t> baseline_support{1, 0, 1, 0};
    std::vector<std::uint8_t> candidate_support{1, 1};

    EXPECT_FALSE(xjw::mesh::commitDepthTsdfRecoveryTransaction(
        evaluation, baseline_support, &candidate_support));
    EXPECT_EQ(candidate_support, baseline_support);
}

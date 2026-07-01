#include "GuidedRematchService.h"
#include "common/SfmTypes.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace
{

cv::Mat descriptors(std::initializer_list<std::initializer_list<float>> rows)
{
    const int row_count = static_cast<int>(rows.size());
    const int col_count = row_count > 0 ? static_cast<int>(rows.begin()->size()) : 0;
    cv::Mat mat(row_count, col_count, CV_32F);

    int row_index = 0;
    for (const auto &row : rows)
    {
        int col_index = 0;
        for (const float value : row)
        {
            mat.at<float>(row_index, col_index) = value;
            ++col_index;
        }
        ++row_index;
    }

    return mat;
}

xjw::gui::GuidedRematchPair makeEligiblePair()
{
    xjw::gui::GuidedRematchPair pair;
    pair.hasRegisteredCameraA = true;
    pair.hasRegisteredCameraB = true;
    pair.overlapScore = 0.62;
    pair.geometricInlierCount = 24;
    pair.permanentlyRejected = false;
    return pair;
}

} // namespace

TEST(GuidedRematchServiceTest, RejectsPairsThatCannotUseGuidedEpipolarSearch)
{
    xjw::gui::GuidedRematchOptions options;
    options.minOverlapScore = 0.2;
    options.targetInlierCount = 80;

    xjw::gui::GuidedRematchPair pair = makeEligiblePair();
    EXPECT_TRUE(xjw::gui::isEligibleForGuidedRematch(pair, options));

    pair = makeEligiblePair();
    pair.hasRegisteredCameraA = false;
    EXPECT_FALSE(xjw::gui::isEligibleForGuidedRematch(pair, options));

    pair = makeEligiblePair();
    pair.overlapScore = 0.19;
    EXPECT_FALSE(xjw::gui::isEligibleForGuidedRematch(pair, options));

    pair = makeEligiblePair();
    pair.geometricInlierCount = 80;
    EXPECT_FALSE(xjw::gui::isEligibleForGuidedRematch(pair, options));

    pair = makeEligiblePair();
    pair.permanentlyRejected = true;
    EXPECT_FALSE(xjw::gui::isEligibleForGuidedRematch(pair, options));
}

TEST(GuidedRematchServiceTest, GeneratesOnlyNewMatchesInsideEpipolarBand)
{
    xjw::gui::GuidedRematchOptions options;
    options.minOverlapScore = 0.2;
    options.targetInlierCount = 80;
    options.epipolarBandPx = 1.0;

    xjw::gui::GuidedRematchInput input;
    input.pair = makeEligiblePair();
    input.options = options;
    input.fundamentalMatrix = (cv::Mat_<double>(3, 3) << 0.0, 0.0, 0.0,
                                                       0.0, 0.0, -1.0,
                                                       0.0, 1.0, 0.0);
    input.keypointsA = {
        cv::Point2f(10.0f, 10.0f),
        cv::Point2f(20.0f, 30.0f),
        cv::Point2f(30.0f, 50.0f),
    };
    input.keypointsB = {
        cv::Point2f(100.0f, 10.2f),
        cv::Point2f(120.0f, 22.0f),
        cv::Point2f(130.0f, 30.4f),
        cv::Point2f(140.0f, 50.8f),
    };
    input.descriptorsA = descriptors({
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.5f, 0.5f},
    });
    input.descriptorsB = descriptors({
        {1.0f, 0.0f},
        {0.2f, 0.8f},
        {0.0f, 1.0f},
        {0.5f, 0.5f},
    });
    input.existingMatches.push_back({0, 0});
    input.existingMatches.push_back({2, 3});

    const xjw::gui::GuidedRematchResult result =
        xjw::gui::generateGuidedRematchCandidates(input);

    ASSERT_TRUE(result.executed);
    ASSERT_EQ(result.matches.size(), 1);
    EXPECT_EQ(result.matches.front().queryIndex, 1);
    EXPECT_EQ(result.matches.front().trainIndex, 2);
    EXPECT_EQ(result.matches.front().source, xjw::gui::GuidedRematchSource::GuidedRematch);
    EXPECT_FALSE(result.matches.front().replacesExistingMatch);
}

TEST(GuidedRematchServiceTest, MergesGuidedCandidatesAsAppendOnlySfmMatches)
{
    std::vector<xjw::FeatureMatch> existingMatches;
    existingMatches.push_back({0, 0, 0.91f});

    xjw::gui::GuidedRematchResult guidedResult;
    guidedResult.executed = true;

    xjw::gui::GuidedRematchMatch duplicateQuery;
    duplicateQuery.queryIndex = 0;
    duplicateQuery.trainIndex = 2;
    duplicateQuery.score = 0.72f;
    guidedResult.matches.push_back(duplicateQuery);

    xjw::gui::GuidedRematchMatch newMatch;
    newMatch.queryIndex = 1;
    newMatch.trainIndex = 3;
    newMatch.score = 0.84f;
    guidedResult.matches.push_back(newMatch);

    const xjw::gui::GuidedRematchMergeResult merged =
        xjw::gui::mergeGuidedRematchMatches(existingMatches, guidedResult);

    ASSERT_EQ(merged.matches.size(), 2);
    EXPECT_EQ(merged.addedMatchCount, 1);
    EXPECT_EQ(merged.skippedExistingMatchCount, 1);
    EXPECT_EQ(merged.skippedInvalidMatchCount, 0);

    EXPECT_EQ(merged.matches.front().idx1, 0u);
    EXPECT_EQ(merged.matches.front().idx2, 0u);
    EXPECT_FLOAT_EQ(merged.matches.front().score, 0.91f);

    EXPECT_EQ(merged.matches.back().idx1, 1u);
    EXPECT_EQ(merged.matches.back().idx2, 3u);
    EXPECT_FLOAT_EQ(merged.matches.back().score, 0.84f);
}

#include "search/AdaptiveFocalSearch.h"

#include <gtest/gtest.h>

#include <vector>

TEST(AdaptiveFocalSearchTest, PrioritizesRegistrationCoverageBeforePointCountAndRms)
{
    using xjw::aerial_triangulation::AdaptiveFocalCandidate;
    std::vector<AdaptiveFocalCandidate> candidates;
    candidates.push_back({0.9, true, 16, 100, 1.0});
    candidates.push_back({1.0, true, 15, 1000, 0.3});
    candidates.push_back({1.1, true, 16, 500, 0.4});

    const int best = xjw::aerial_triangulation::AdaptiveFocalSearch::selectBestCandidate(candidates, 16);

    ASSERT_EQ(best, 2);
    EXPECT_DOUBLE_EQ(candidates[static_cast<std::size_t>(best)].focalScale, 1.1);
}

TEST(AdaptiveFocalSearchTest, RejectsFailedCandidateEvenWhenItsCountsAreHigher)
{
    using xjw::aerial_triangulation::AdaptiveFocalCandidate;
    std::vector<AdaptiveFocalCandidate> candidates;
    candidates.push_back({0.8, false, 16, 2000, 0.1});
    candidates.push_back({1.0, true, 12, 300, 0.8});

    EXPECT_EQ(xjw::aerial_triangulation::AdaptiveFocalSearch::selectBestCandidate(candidates, 16), 1);
}

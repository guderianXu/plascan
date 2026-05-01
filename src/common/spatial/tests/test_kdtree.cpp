#include <gtest/gtest.h>

#include "spatial/KDTree.h"

#include <array>
#include <vector>

namespace
{

using Tree2D = xjw::common::spatial::KDTree<2, double>;
using Tree3F = xjw::common::spatial::KDTree<3, float>;

TEST(CommonKDTreeTest, NearestInTwoDimensions)
{
    std::vector<Tree2D::Point> points{
        {{{0.0, 0.0}}, 10},
        {{{5.0, 5.0}}, 20},
        {{{1.0, 1.0}}, 30},
        {{{-2.0, 4.0}}, 40}
    };

    Tree2D tree(points);

    double distance = 0.0;
    const int nearestIndex = tree.nearest({0.8, 1.2}, &distance);

    EXPECT_EQ(nearestIndex, 30);
    EXPECT_NEAR(distance, std::sqrt(0.08), 1e-8);
}

TEST(CommonKDTreeTest, KNearestByPointIndexSkipsSelf)
{
    std::vector<Tree3F::CoordinateArray> points{
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 3.0f}
    };

    Tree3F tree(points);
    const std::vector<Tree3F::Neighbor> neighbors = tree.kNearestByPointIndex(0, 2);

    ASSERT_EQ(neighbors.size(), 2U);
    EXPECT_EQ(neighbors[0].index, 1);
    EXPECT_FLOAT_EQ(neighbors[0].distanceSquared, 1.0f);
    EXPECT_EQ(neighbors[1].index, 2);
    EXPECT_FLOAT_EQ(neighbors[1].distanceSquared, 4.0f);
}

TEST(CommonKDTreeTest, RadiusSearchReturnsMatchingIndices)
{
    std::vector<Tree2D::CoordinateArray> points{
        {0.0, 0.0},
        {1.0, 0.0},
        {2.0, 0.0},
        {4.0, 0.0}
    };

    Tree2D tree(points);
    const std::vector<int> result = tree.radiusSearch({0.0, 0.0}, 1.5);

    ASSERT_EQ(result.size(), 2U);
    EXPECT_TRUE((result[0] == 0 || result[1] == 0));
    EXPECT_TRUE((result[0] == 1 || result[1] == 1));
}

TEST(CommonKDTreeTest, RadiusCountByPointIndexSupportsEarlyStop)
{
    std::vector<Tree3F::CoordinateArray> points{
        {0.0f, 0.0f, 0.0f},
        {0.1f, 0.0f, 0.0f},
        {0.2f, 0.0f, 0.0f},
        {0.3f, 0.0f, 0.0f},
        {5.0f, 0.0f, 0.0f}
    };

    Tree3F tree(points);
    const int count = tree.radiusCountByPointIndex(0, 1.0f, 2);

    EXPECT_EQ(count, 2);
}

TEST(CommonKDTreeTest, EmptyTreeQueriesAreSafe)
{
    Tree2D tree;
    double distance = 123.0;

    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.nearest({0.0, 0.0}, &distance), -1);
    EXPECT_TRUE(tree.kNearest({0.0, 0.0}, 3).empty());
    EXPECT_TRUE(tree.radiusSearch({0.0, 0.0}, 1.0).empty());
    EXPECT_EQ(tree.radiusCount({0.0, 0.0}, 1.0), 0);
}

} // namespace
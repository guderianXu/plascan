#include <gtest/gtest.h>

#include "HierarchicalVocabularyTree.h"
#include "VocabularyOverlapRetriever.h"

#include <opencv2/core.hpp>

#include <limits>
#include <vector>

namespace {

cv::Mat makeDescriptors(const std::vector<std::vector<float>> &rows)
{
    if (rows.empty())
    {
        return cv::Mat();
    }

    cv::Mat descriptors(static_cast<int>(rows.size()),
                        static_cast<int>(rows.front().size()),
                        CV_32F);
    for (int r = 0; r < descriptors.rows; ++r)
    {
        for (int c = 0; c < descriptors.cols; ++c)
        {
            descriptors.at<float>(r, c) = rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
        }
    }
    return descriptors;
}

cv::Mat makeGeneratedDescriptors(int rows, int columns, float base)
{
    cv::Mat descriptors(rows, columns, CV_32F);
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            descriptors.at<float>(row, column) =
                base + static_cast<float>(row % 7) * 0.01f +
                static_cast<float>(column % 11) * 0.001f;
        }
    }
    return descriptors;
}

xjw::VocabularyImageFeatures makeImage(const std::string &path, const cv::Mat &descriptors)
{
    xjw::VocabularyImageFeatures image;
    image.imagePath = path;
    image.descriptors = descriptors;
    image.keypoints.reserve(static_cast<std::size_t>(descriptors.rows));
    for (int row = 0; row < descriptors.rows; ++row)
    {
        image.keypoints.emplace_back(static_cast<float>(row), static_cast<float>(row), 1.0f);
    }
    return image;
}

} // namespace

TEST(HierarchicalVocabularyTreeTest, BuildsAndTraversesMultipleKMeansLevels)
{
    cv::Mat training(64, 4, CV_32F);
    for (int row = 0; row < training.rows; ++row)
    {
        for (int column = 0; column < training.cols; ++column)
        {
            training.at<float>(row, column) = static_cast<float>(row) + column * 0.01f;
        }
    }

    xjw::HierarchicalVocabularyTreeConfig config;
    config.branchFactor = 2;
    config.maximumDepth = 3;
    config.maximumLeaves = 8;

    xjw::HierarchicalVocabularyTree tree;
    std::string error;
    ASSERT_TRUE(tree.train(training, config, &error)) << error;

    EXPECT_EQ(tree.actualDepth(), 3);
    EXPECT_EQ(tree.wordCount(), 8);
    EXPECT_EQ(tree.nodeCount(), 15);
    const int first_word = tree.quantize(training.ptr<float>(0), training.cols);
    const int last_word = tree.quantize(training.ptr<float>(training.rows - 1), training.cols);
    EXPECT_GE(first_word, 0);
    EXPECT_LT(first_word, tree.wordCount());
    EXPECT_GE(last_word, 0);
    EXPECT_LT(last_word, tree.wordCount());
    EXPECT_NE(first_word, last_word);
}

TEST(HierarchicalVocabularyTreeTest, HonorsLeafLimitWithPartialFinalLevel)
{
    const cv::Mat training = makeGeneratedDescriptors(60, 8, 0.0f);
    xjw::HierarchicalVocabularyTreeConfig config;
    config.branchFactor = 3;
    config.maximumDepth = 3;
    config.maximumLeaves = 5;

    xjw::HierarchicalVocabularyTree tree;
    std::string error;
    ASSERT_TRUE(tree.train(training, config, &error)) << error;

    EXPECT_EQ(tree.wordCount(), 5);
    EXPECT_LE(tree.actualDepth(), 3);
    EXPECT_GT(tree.nodeCount(), tree.wordCount());
}

TEST(HierarchicalVocabularyTreeTest, CanCancelBetweenKMeansNodes)
{
    const cv::Mat training = makeGeneratedDescriptors(60, 8, 0.0f);
    xjw::HierarchicalVocabularyTreeConfig config;
    config.branchFactor = 3;
    config.maximumDepth = 3;
    config.maximumLeaves = 27;
    config.progressCallback = [](int, int, int)
    {
        return false;
    };

    xjw::HierarchicalVocabularyTree tree;
    std::string error;
    EXPECT_FALSE(tree.train(training, config, &error));
    EXPECT_NE(error.find("取消"), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, RetrievesExpectedPairFromSharedDescriptors)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeDescriptors({
        {0.00f, 0.00f},
        {0.02f, 0.01f},
        {1.00f, 1.00f},
        {1.02f, 1.01f},
    })));
    images.push_back(makeImage("b.tif", makeDescriptors({
        {0.01f, 0.00f},
        {0.03f, 0.02f},
        {1.01f, 1.00f},
        {1.03f, 1.02f},
    })));
    images.push_back(makeImage("c.tif", makeDescriptors({
        {8.00f, 8.00f},
        {8.10f, 8.00f},
        {9.00f, 9.00f},
        {9.10f, 9.00f},
    })));

    xjw::VocabularyOverlapConfig config;
    config.branchFactor = 2;
    config.treeDepth = 2;
    config.samplePerImage = 100;
    config.maxTrainingDescriptors = 1000;
    config.topK = 1;
    config.minSimilarity = 0.05;
    config.useTfidf = true;
    config.mutualTopK = true;
    config.keepOneWayTopK = false;
    config.cycleClosureMaxPairsPerImage = 2;
    config.connectComponents = false;
    config.closeSequenceLoop = false;
    config.geometryCheck = false;

    xjw::VocabularyOverlapResult result;
    std::string error;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error)) << error;

    ASSERT_EQ(result.acceptedPairs.size(), 1u);
    EXPECT_EQ(result.acceptedPairs.front().indexA, 0);
    EXPECT_EQ(result.acceptedPairs.front().indexB, 1);
    EXPECT_GT(result.acceptedPairs.front().bowScore, 0.5);
    EXPECT_TRUE(result.acceptedPairs.front().accepted);
    EXPECT_NE(result.detail.find("pair_top_k=1"), std::string::npos);
    EXPECT_NE(result.detail.find("cycle_closure_budget_per_image=2"), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, RejectsDescriptorDimensionMismatch)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeDescriptors({{0.0f, 0.0f}, {1.0f, 1.0f}})));
    images.push_back(makeImage("b.tif", makeDescriptors({{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}})));

    xjw::VocabularyOverlapConfig config;
    xjw::VocabularyOverlapResult result;
    std::string error;

    EXPECT_FALSE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error));
    EXPECT_NE(error.find("描述子维度"), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, RejectsNonFiniteDescriptorsBeforeKMeans)
{
    cv::Mat invalid = makeGeneratedDescriptors(8, 4, 0.0f);
    invalid.at<float>(3, 2) = std::numeric_limits<float>::quiet_NaN();
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("valid.tif", makeGeneratedDescriptors(8, 4, 0.1f)));
    images.push_back(makeImage("invalid.tif", invalid));

    xjw::VocabularyOverlapConfig config;
    xjw::VocabularyOverlapResult result;
    std::string error;

    EXPECT_FALSE(xjw::VocabularyOverlapRetriever::retrieve(
        images, config, &result, &error));
    EXPECT_NE(error.find("NaN 或无穷值"), std::string::npos);
    EXPECT_NE(error.find("invalid.tif"), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, CapsOnlyDescriptorsUsedByVocabularyAssignment)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeGeneratedDescriptors(20, 8, 0.0f)));
    images.push_back(makeImage("b.tif", makeGeneratedDescriptors(20, 8, 0.1f)));
    images.push_back(makeImage("c.tif", makeGeneratedDescriptors(20, 8, 4.0f)));

    xjw::VocabularyOverlapConfig config;
    config.branchFactor = 2;
    config.treeDepth = 1;
    config.samplePerImage = 5;
    config.maxTrainingDescriptors = 15;
    config.maxDescriptorsPerImage = 6;
    config.connectComponents = false;

    xjw::VocabularyOverlapResult result;
    std::string error;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error)) << error;

    EXPECT_EQ(result.totalDescriptorCount, 60u);
    EXPECT_EQ(result.assignedDescriptorCount, 18u);
    EXPECT_NE(result.detail.find("descriptors_total=60"), std::string::npos);
    EXPECT_NE(result.detail.find("descriptors_assigned=18"), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, SupportsLoMaRDescriptorDimensionWithHierarchicalTree)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeGeneratedDescriptors(12, 256, 0.0f)));
    images.push_back(makeImage("b.tif", makeGeneratedDescriptors(12, 256, 0.02f)));
    images.push_back(makeImage("c.tif", makeGeneratedDescriptors(12, 256, 5.0f)));

    xjw::VocabularyOverlapConfig config;
    config.branchFactor = 2;
    config.treeDepth = 2;
    config.samplePerImage = 8;
    config.maxTrainingDescriptors = 24;
    config.maxDescriptorsPerImage = 8;
    config.connectComponents = false;

    xjw::VocabularyOverlapResult result;
    std::string error;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error)) << error;

    EXPECT_EQ(result.totalDescriptorCount, 36u);
    EXPECT_EQ(result.assignedDescriptorCount, 24u);
    EXPECT_NE(result.detail.find("descriptor_dims=256"), std::string::npos);
    EXPECT_NE(result.detail.find("assignment=hierarchical_kmeans_tree"), std::string::npos);
    EXPECT_EQ(result.vocabularyTreeDepth, 2);
    EXPECT_GT(result.vocabularyTreeNodeCount, result.vocabularySize);
    EXPECT_GE(xjw::VocabularyOverlapConfig().maxDescriptorsPerImage, 3840);
}

TEST(VocabularyOverlapRetrieverTest, InvertedIndexMatchesDensePairScoring)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeDescriptors({
        {0.00f, 0.00f}, {0.02f, 0.01f}, {1.00f, 1.00f}, {1.02f, 1.01f},
    })));
    images.push_back(makeImage("b.tif", makeDescriptors({
        {0.01f, 0.00f}, {0.03f, 0.02f}, {1.01f, 1.00f}, {1.03f, 1.02f},
    })));
    images.push_back(makeImage("c.tif", makeDescriptors({
        {8.00f, 8.00f}, {8.10f, 8.00f}, {9.00f, 9.00f}, {9.10f, 9.00f},
    })));

    xjw::VocabularyOverlapConfig config;
    config.branchFactor = 2;
    config.treeDepth = 2;
    config.samplePerImage = 100;
    config.maxTrainingDescriptors = 1000;
    config.topK = 1;
    config.minSimilarity = 0.05;
    config.useTfidf = true;
    config.mutualTopK = true;
    config.geometryCheck = false;
    config.numThreads = 2;

    xjw::VocabularyOverlapResult denseResult;
    std::string denseError;
    config.useInvertedIndex = false;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &denseResult, &denseError)) << denseError;

    xjw::VocabularyOverlapResult invertedResult;
    std::string invertedError;
    config.useInvertedIndex = true;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &invertedResult, &invertedError)) << invertedError;

    ASSERT_EQ(invertedResult.acceptedPairs.size(), denseResult.acceptedPairs.size());
    ASSERT_FALSE(invertedResult.acceptedPairs.empty());
    EXPECT_EQ(invertedResult.acceptedPairs.front().indexA, denseResult.acceptedPairs.front().indexA);
    EXPECT_EQ(invertedResult.acceptedPairs.front().indexB, denseResult.acceptedPairs.front().indexB);
}

TEST(VocabularyOverlapRetrieverTest, PlannerKeepsOneWayTopKWhenMutualTopKWouldDisconnect)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeDescriptors({
        {0.00f, 0.00f}, {0.01f, 0.00f}, {0.02f, 0.01f}, {0.03f, 0.01f},
        {8.00f, 8.00f}, {8.10f, 8.00f},
    })));
    images.push_back(makeImage("b.tif", makeDescriptors({
        {0.00f, 0.01f}, {0.01f, 0.02f}, {0.02f, 0.03f}, {0.03f, 0.04f},
    })));
    images.push_back(makeImage("c.tif", makeDescriptors({
        {8.00f, 8.10f}, {8.10f, 8.10f},
    })));

    xjw::VocabularyOverlapConfig config;
    config.branchFactor = 2;
    config.treeDepth = 1;
    config.samplePerImage = 100;
    config.maxTrainingDescriptors = 1000;
    config.topK = 1;
    config.minPairsPerImage = 1;
    config.minSimilarity = 0.01;
    config.useTfidf = false;
    config.mutualTopK = true;
    config.keepOneWayTopK = true;
    config.connectComponents = false;
    config.closeSequenceLoop = false;
    config.geometryCheck = false;
    config.useInvertedIndex = false;

    xjw::VocabularyOverlapResult result;
    std::string error;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error)) << error;

    const auto hasPair = [](const std::vector<xjw::VocabularyOverlapPairResult> &pairs, int indexA, int indexB)
    {
        return std::any_of(pairs.begin(), pairs.end(), [=](const xjw::VocabularyOverlapPairResult &pair)
        {
            return pair.indexA == indexA && pair.indexB == indexB;
        });
    };

    EXPECT_TRUE(hasPair(result.acceptedPairs, 0, 1));
    EXPECT_TRUE(hasPair(result.acceptedPairs, 0, 2));
}

TEST(VocabularyOverlapRetrieverTest, PlannerReportsConnectivityRepairInDetail)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeDescriptors({{0.00f, 0.00f}, {0.01f, 0.00f}})));
    images.push_back(makeImage("b.tif", makeDescriptors({{0.00f, 0.01f}, {0.01f, 0.02f}})));
    images.push_back(makeImage("c.tif", makeDescriptors({{8.00f, 8.00f}, {8.10f, 8.00f}})));
    images.push_back(makeImage("d.tif", makeDescriptors({{8.00f, 8.10f}, {8.10f, 8.10f}})));

    xjw::VocabularyOverlapConfig config;
    config.branchFactor = 2;
    config.treeDepth = 1;
    config.samplePerImage = 100;
    config.maxTrainingDescriptors = 1000;
    config.topK = 1;
    config.minPairsPerImage = 0;
    config.minSimilarity = 0.01;
    config.useTfidf = false;
    config.mutualTopK = true;
    config.keepOneWayTopK = false;
    config.connectComponents = true;
    config.useSequenceFallback = true;
    config.sequenceWindow = 1;
    config.closeSequenceLoop = false;
    config.geometryCheck = false;
    config.useInvertedIndex = false;

    xjw::VocabularyOverlapResult result;
    std::string error;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error)) << error;

    EXPECT_NE(result.detail.find("components_before="), std::string::npos);
    EXPECT_NE(result.detail.find("components_after="), std::string::npos);
    EXPECT_NE(result.detail.find("sequence_bridges="), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, ProgressCallbackCanCancel)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeDescriptors({{0.0f, 0.0f}, {1.0f, 1.0f}})));
    images.push_back(makeImage("b.tif", makeDescriptors({{0.0f, 0.0f}, {1.0f, 1.0f}})));

    xjw::VocabularyOverlapConfig config;
    config.progressCallback = [](const std::string &, int)
    {
        return false;
    };

    xjw::VocabularyOverlapResult result;
    std::string error;
    EXPECT_FALSE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error));
    EXPECT_NE(error.find("取消"), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, ProgressCallbackCanCancelDuringChunkedAssignment)
{
    std::vector<xjw::VocabularyImageFeatures> images;
    images.push_back(makeImage("a.tif", makeGeneratedDescriptors(40, 8, 0.0f)));
    images.push_back(makeImage("b.tif", makeGeneratedDescriptors(40, 8, 0.1f)));

    bool reached_assignment = false;
    xjw::VocabularyOverlapConfig config;
    config.branchFactor = 2;
    config.treeDepth = 2;
    config.samplePerImage = 20;
    config.maxTrainingDescriptors = 40;
    config.maxDescriptorsPerImage = 40;
    config.progressCallback = [&reached_assignment](const std::string &stage, int percent)
    {
        if (stage.rfind("分配层次词汇树 ", 0) == 0)
        {
            reached_assignment = true;
            return percent <= 30;
        }
        return true;
    };

    xjw::VocabularyOverlapResult result;
    std::string error;
    EXPECT_FALSE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error));
    EXPECT_TRUE(reached_assignment);
    EXPECT_NE(error.find("取消"), std::string::npos);
}

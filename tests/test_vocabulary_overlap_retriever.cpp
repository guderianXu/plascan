#include <gtest/gtest.h>

#include "VocabularyOverlapRetriever.h"

#include <opencv2/core.hpp>

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
    config.useFlannAssignment = false;
    config.connectComponents = false;

    xjw::VocabularyOverlapResult result;
    std::string error;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error)) << error;

    EXPECT_EQ(result.totalDescriptorCount, 60u);
    EXPECT_EQ(result.assignedDescriptorCount, 18u);
    EXPECT_NE(result.detail.find("descriptors_total=60"), std::string::npos);
    EXPECT_NE(result.detail.find("descriptors_assigned=18"), std::string::npos);
}

TEST(VocabularyOverlapRetrieverTest, SupportsLoMaRDescriptorDimensionWithFlann)
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
    config.assignmentBatchRows = 3;
    config.connectComponents = false;

    xjw::VocabularyOverlapResult result;
    std::string error;
    ASSERT_TRUE(xjw::VocabularyOverlapRetriever::retrieve(images, config, &result, &error)) << error;

    EXPECT_EQ(result.totalDescriptorCount, 36u);
    EXPECT_EQ(result.assignedDescriptorCount, 24u);
    EXPECT_NE(result.detail.find("descriptor_dims=256"), std::string::npos);
    EXPECT_NE(result.detail.find("assignment=flann"), std::string::npos);
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
    config.useFlannAssignment = false;
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
    config.useFlannAssignment = false;
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
    config.useFlannAssignment = false;
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
    config.assignmentBatchRows = 4;
    config.progressCallback = [&reached_assignment](const std::string &stage, int percent)
    {
        if (stage.rfind("分配视觉词汇 ", 0) == 0)
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

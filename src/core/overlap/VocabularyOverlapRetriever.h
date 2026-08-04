#pragma once

#include "OverlapPairGraphPlanner.h"

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw {

struct VocabularyImageFeatures
{
    std::string imagePath;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct VocabularyOverlapConfig
{
    int branchFactor = 10;
    int treeDepth = 3;
    int samplePerImage = 500;
    int maxTrainingDescriptors = 50000;
    int maxDescriptorsPerImage = 4096;
    int maxVocabularyWords = 4096;
    int kmeansMaxIterations = 20;
    int kmeansAttempts = 1;
    double kmeansEpsilon = 1e-3;
    int topK = 8;
    int minPairsPerImage = 4;
    double minSimilarity = 0.05;
    bool useTfidf = true;
    bool mutualTopK = true;
    bool keepOneWayTopK = true;
    bool connectComponents = true;
    bool useSequenceFallback = true;
    int sequenceWindow = 1;
    bool closeSequenceLoop = true;
    int componentBridgeMaxPairs = 0;
    bool geometryCheck = false;
    int minInliers = 30;
    double ransacThreshold = 3.0;
    int numThreads = 0;
    bool useInvertedIndex = true;
    bool useCuda = false;
    int geometryMaxDescriptors = 2048;
    int geometryMaxCandidatePairs = 2000;
    std::function<bool(const std::string &stage, int percent)> progressCallback;
};

struct VocabularyOverlapPairResult
{
    int indexA = -1;
    int indexB = -1;
    std::string imagePathA;
    std::string imagePathB;
    double bowScore = 0.0;
    int sharedWordCount = 0;
    int geometricInliers = 0;
    bool accepted = false;
    std::string rejectReason;
    std::vector<std::string> sourceTypes;
};

struct VocabularyOverlapResult
{
    int vocabularySize = 0;
    int vocabularyTreeNodeCount = 0;
    int vocabularyTreeDepth = 0;
    std::uint64_t totalDescriptorCount = 0;
    std::uint64_t assignedDescriptorCount = 0;
    std::vector<VocabularyOverlapPairResult> candidates;
    std::vector<VocabularyOverlapPairResult> acceptedPairs;
    std::string detail;
};

class VocabularyOverlapRetriever
{
public:
    static bool retrieve(const std::vector<VocabularyImageFeatures> &images,
                         const VocabularyOverlapConfig &config,
                         VocabularyOverlapResult *result,
                         std::string *errorMsg = nullptr);
};

} // namespace xjw

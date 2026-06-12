#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

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
    int maxVocabularyWords = 4096;
    int topK = 8;
    double minSimilarity = 0.05;
    bool useTfidf = true;
    bool mutualTopK = true;
    bool geometryCheck = false;
    int minInliers = 30;
    double ransacThreshold = 3.0;
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
};

struct VocabularyOverlapResult
{
    int vocabularySize = 0;
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

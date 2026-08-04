#pragma once

#include <opencv2/core.hpp>

#include <functional>
#include <string>
#include <vector>

namespace xjw
{

struct HierarchicalVocabularyTreeConfig
{
    int branchFactor = 10;
    int maximumDepth = 3;
    int maximumLeaves = 4096;
    int kmeansMaxIterations = 20;
    int kmeansAttempts = 1;
    double kmeansEpsilon = 1e-3;
    std::function<bool(int completedNodes, int estimatedNodes, int depth)> progressCallback;
};

class HierarchicalVocabularyTree
{
public:
    bool train(const cv::Mat &training,
               const HierarchicalVocabularyTreeConfig &config,
               std::string *errorMessage = nullptr);

    int quantize(const float *descriptor, int dimensions) const;
    int wordCount() const;
    int nodeCount() const;
    int actualDepth() const;

private:
    struct Node
    {
        cv::Mat center;
        std::vector<int> children;
        int wordIndex = -1;
    };

    void makeLeaf(int nodeIndex, int depth);
    std::vector<int> childLeafBudgets(const std::vector<std::vector<int>> &childRows,
                                      int leafBudget,
                                      int remainingDepth) const;
    bool reportTrainingProgress(const HierarchicalVocabularyTreeConfig &config,
                                int depth,
                                std::string *errorMessage);
    bool buildNode(int nodeIndex,
                   const cv::Mat &training,
                   const std::vector<int> &rows,
                   int depth,
                   int leafBudget,
                   const HierarchicalVocabularyTreeConfig &config,
                   std::string *errorMessage);

    std::vector<Node> _nodes;
    int _wordCount = 0;
    int _actualDepth = 0;
    int _branchFactor = 2;
    int _maximumDepth = 1;
    int _trainedInternalNodes = 0;
    int _estimatedInternalNodes = 1;
};

} // namespace xjw

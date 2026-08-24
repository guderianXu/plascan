#include "HierarchicalVocabularyTree.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <utility>

namespace
{

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

int maximumLeaves(const xjw::HierarchicalVocabularyTreeConfig &config, int descriptorCount)
{
    const int branch_factor = std::max(2, config.branchFactor);
    const int maximum_depth = std::max(1, config.maximumDepth);
    const int leaf_limit = std::max(1, config.maximumLeaves);
    std::int64_t leaves = 1;
    for (int depth = 0; depth < maximum_depth && leaves < leaf_limit; ++depth)
    {
        leaves = std::min<std::int64_t>(leaf_limit, leaves * branch_factor);
    }
    return std::max(1, static_cast<int>(std::min<std::int64_t>(leaves, descriptorCount)));
}

int maximumSubtreeLeaves(int branchFactor, int remainingDepth, int descriptorCount)
{
    std::int64_t leaves = 1;
    for (int depth = 0; depth < remainingDepth && leaves < descriptorCount; ++depth)
    {
        leaves = std::min<std::int64_t>(descriptorCount, leaves * branchFactor);
    }
    return std::max(1, static_cast<int>(leaves));
}

double descriptorDistanceSquared(const float *left, const float *right, int dimensions)
{
    double distance = 0.0;
    for (int dimension = 0; dimension < dimensions; ++dimension)
    {
        const double delta = static_cast<double>(left[dimension]) - right[dimension];
        distance += delta * delta;
    }
    return distance;
}

} // namespace

namespace xjw
{

bool HierarchicalVocabularyTree::train(const cv::Mat &training,
                                       const HierarchicalVocabularyTreeConfig &config,
                                       std::string *errorMessage)
{
    if (training.empty() || training.type() != CV_32F)
    {
        setError(errorMessage, "层次词汇树缺少有效的浮点训练描述子");
        return false;
    }

    _nodes.clear();
    _wordCount = 0;
    _actualDepth = 0;
    _trainedInternalNodes = 0;
    _branchFactor = std::max(2, config.branchFactor);
    _maximumDepth = std::max(1, config.maximumDepth);
    const int leaf_count = maximumLeaves(config, training.rows);
    _estimatedInternalNodes = std::max(1, (leaf_count - 1) / (_branchFactor - 1));
    _nodes.reserve(static_cast<std::size_t>(leaf_count * 2));
    _nodes.emplace_back();

    std::vector<int> rows(static_cast<std::size_t>(training.rows));
    std::iota(rows.begin(), rows.end(), 0);
    cv::setRNGSeed(0);
    try
    {
        return buildNode(0, training, rows, 0, leaf_count, config, errorMessage);
    }
    catch (const cv::Exception &ex)
    {
        setError(errorMessage, std::string("训练层次词汇树失败: ") + ex.what());
        return false;
    }
    catch (const std::exception &ex)
    {
        setError(errorMessage, std::string("训练层次词汇树失败: ") + ex.what());
        return false;
    }
}

int HierarchicalVocabularyTree::quantize(const float *descriptor, int dimensions) const
{
    if (!descriptor || dimensions <= 0 || _nodes.empty())
    {
        return -1;
    }

    int node_index = 0;
    while (!_nodes[static_cast<std::size_t>(node_index)].children.empty())
    {
        const Node &node = _nodes[static_cast<std::size_t>(node_index)];
        double best_distance = std::numeric_limits<double>::max();
        int best_child = node.children.front();
        for (int child_index : node.children)
        {
            const Node &child = _nodes[static_cast<std::size_t>(child_index)];
            const double distance = descriptorDistanceSquared(
                descriptor,
                child.center.ptr<float>(0),
                dimensions);
            if (distance < best_distance)
            {
                best_distance = distance;
                best_child = child_index;
            }
        }
        node_index = best_child;
    }
    return _nodes[static_cast<std::size_t>(node_index)].wordIndex;
}

int HierarchicalVocabularyTree::wordCount() const
{
    return _wordCount;
}

int HierarchicalVocabularyTree::nodeCount() const
{
    return static_cast<int>(_nodes.size());
}

int HierarchicalVocabularyTree::actualDepth() const
{
    return _actualDepth;
}

void HierarchicalVocabularyTree::makeLeaf(int nodeIndex, int depth)
{
    Node &node = _nodes[static_cast<std::size_t>(nodeIndex)];
    node.children.clear();
    node.wordIndex = _wordCount++;
    _actualDepth = std::max(_actualDepth, depth);
}

std::vector<int> HierarchicalVocabularyTree::childLeafBudgets(
    const std::vector<std::vector<int>> &childRows,
    int leafBudget,
    int remainingDepth) const
{
    const int child_count = static_cast<int>(childRows.size());
    std::vector<int> budgets(static_cast<std::size_t>(child_count), 1);
    std::vector<int> capacities(static_cast<std::size_t>(child_count), 1);
    int allocated = child_count;
    for (int child = 0; child < child_count; ++child)
    {
        capacities[static_cast<std::size_t>(child)] = maximumSubtreeLeaves(
            _branchFactor,
            remainingDepth,
            static_cast<int>(childRows[static_cast<std::size_t>(child)].size()));
    }

    while (allocated < leafBudget)
    {
        bool allocated_any = false;
        for (int child = 0; child < child_count; ++child)
        {
            if (budgets[static_cast<std::size_t>(child)] >=
                capacities[static_cast<std::size_t>(child)])
            {
                continue;
            }
            ++budgets[static_cast<std::size_t>(child)];
            ++allocated;
            allocated_any = true;
            if (allocated >= leafBudget)
            {
                break;
            }
        }
        if (!allocated_any)
        {
            break;
        }
    }
    return budgets;
}

bool HierarchicalVocabularyTree::reportTrainingProgress(
    const HierarchicalVocabularyTreeConfig &config,
    int depth,
    std::string *errorMessage)
{
    ++_trainedInternalNodes;
    if (!config.progressCallback ||
        config.progressCallback(_trainedInternalNodes, _estimatedInternalNodes, depth))
    {
        return true;
    }
    setError(errorMessage, "用户取消训练层次词汇树");
    return false;
}

bool HierarchicalVocabularyTree::buildNode(
    int nodeIndex,
    const cv::Mat &training,
    const std::vector<int> &rows,
    int depth,
    int leafBudget,
    const HierarchicalVocabularyTreeConfig &config,
    std::string *errorMessage)
{
    if (depth >= _maximumDepth || leafBudget <= 1 || rows.size() < 2)
    {
        makeLeaf(nodeIndex, depth);
        return true;
    }

    const int child_count = std::min({
        _branchFactor,
        leafBudget,
        static_cast<int>(rows.size())});
    cv::Mat node_descriptors(
        static_cast<int>(rows.size()), training.cols, training.type());
    for (std::size_t node_row = 0; node_row < rows.size(); ++node_row)
    {
        training.row(rows[node_row]).copyTo(
            node_descriptors.row(static_cast<int>(node_row)));
    }

    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(node_descriptors,
               child_count,
               labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                std::max(1, config.kmeansMaxIterations),
                                std::max(0.0, config.kmeansEpsilon)),
               std::max(1, config.kmeansAttempts),
               cv::KMEANS_PP_CENTERS,
               centers);

    std::vector<std::vector<int>> child_rows(static_cast<std::size_t>(child_count));
    for (int row = 0; row < labels.rows; ++row)
    {
        const int child = labels.at<int>(row, 0);
        child_rows[static_cast<std::size_t>(child)].push_back(rows[static_cast<std::size_t>(row)]);
    }

    std::vector<int> child_indices;
    child_indices.reserve(static_cast<std::size_t>(child_count));
    for (int child = 0; child < child_count; ++child)
    {
        Node child_node;
        child_node.center = centers.row(child).clone();
        child_indices.push_back(static_cast<int>(_nodes.size()));
        _nodes.push_back(std::move(child_node));
    }
    _nodes[static_cast<std::size_t>(nodeIndex)].children = child_indices;
    if (!reportTrainingProgress(config, depth + 1, errorMessage))
    {
        return false;
    }

    const std::vector<int> budgets = childLeafBudgets(
        child_rows,
        leafBudget,
        _maximumDepth - depth - 1);
    for (int child = 0; child < child_count; ++child)
    {
        if (!buildNode(child_indices[static_cast<std::size_t>(child)],
                       training,
                       child_rows[static_cast<std::size_t>(child)],
                       depth + 1,
                       budgets[static_cast<std::size_t>(child)],
                       config,
                       errorMessage))
        {
            return false;
        }
    }
    return true;
}

} // namespace xjw

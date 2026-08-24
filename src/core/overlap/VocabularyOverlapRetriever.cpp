#include "VocabularyOverlapRetriever.h"

#include "HierarchicalVocabularyTree.h"
#include <opencv2/geometry.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void setError(std::string *errorMsg, const std::string &message)
{
    if (errorMsg)
    {
        *errorMsg = message;
    }
}

cv::Mat toFloatDescriptors(const cv::Mat &descriptors)
{
    if (descriptors.empty())
    {
        return cv::Mat();
    }
    if (descriptors.type() == CV_32F)
    {
        return descriptors;
    }

    cv::Mat converted;
    descriptors.convertTo(converted, CV_32F);
    return converted;
}

cv::Mat sampleTrainingDescriptors(const std::vector<cv::Mat> &descriptors,
                                  const xjw::VocabularyOverlapConfig &config)
{
    const int per_image = std::max(1, config.samplePerImage);
    const int max_training = std::max(1, config.maxTrainingDescriptors);

    std::vector<int> candidate_counts;
    candidate_counts.reserve(descriptors.size());
    std::int64_t total_candidates = 0;
    for (const cv::Mat &image_descriptors : descriptors)
    {
        const int candidate_count = std::min(per_image, image_descriptors.rows);
        candidate_counts.push_back(candidate_count);
        total_candidates += candidate_count;
    }

    const int target_count = static_cast<int>(std::min<std::int64_t>(max_training, total_candidates));
    if (target_count <= 0)
    {
        return cv::Mat();
    }

    const auto first_non_empty = std::find_if(
        descriptors.begin(), descriptors.end(), [](const cv::Mat &value) { return !value.empty(); });
    cv::Mat training(target_count, first_non_empty->cols, first_non_empty->type());
    std::int64_t candidates_seen = 0;
    int samples_emitted = 0;
    int output_row = 0;
    for (std::size_t image_index = 0; image_index < descriptors.size(); ++image_index)
    {
        const cv::Mat &image_descriptors = descriptors[image_index];
        candidates_seen += candidate_counts[image_index];
        const int target_through_image = static_cast<int>(
            candidates_seen * target_count / total_candidates);
        const int take_count = target_through_image - samples_emitted;
        samples_emitted = target_through_image;
        if (take_count <= 0)
        {
            continue;
        }

        const double step = static_cast<double>(image_descriptors.rows) / take_count;
        for (int sample = 0; sample < take_count; ++sample)
        {
            const int row = std::min(image_descriptors.rows - 1,
                                     static_cast<int>(std::floor(sample * step)));
            image_descriptors.row(row).copyTo(training.row(output_row));
            ++output_row;
        }
    }
    return training.rowRange(0, output_row);
}

bool reportProgress(const xjw::VocabularyOverlapConfig &config,
                    const std::string &stage,
                    int percent,
                    std::string *errorMsg)
{
    if (config.progressCallback && !config.progressCallback(stage, std::clamp(percent, 0, 100)))
    {
        setError(errorMsg, "用户取消获取重叠对");
        return false;
    }
    return true;
}

int effectiveThreadCount(const xjw::VocabularyOverlapConfig &config, int workItems)
{
    if (workItems <= 1)
    {
        return 1;
    }

    const int requested = config.numThreads;
    const unsigned int hardware = std::max(1u, std::thread::hardware_concurrency());
    const int target = requested > 0 ? requested : static_cast<int>(hardware);
    return std::max(1, std::min(target, workItems));
}

std::uint64_t pairKey(int indexA, int indexB)
{
    if (indexA > indexB)
    {
        std::swap(indexA, indexB);
    }
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(indexA)) << 32) |
           static_cast<std::uint32_t>(indexB);
}

bool validPair(int imageCount, int indexA, int indexB)
{
    return indexA >= 0 &&
           indexB >= 0 &&
           indexA < imageCount &&
           indexB < imageCount &&
           indexA != indexB;
}

std::pair<int, int> decodePairKey(std::uint64_t key)
{
    return {
        static_cast<int>(key >> 32),
        static_cast<int>(key & 0xffffffffu)
    };
}

void normalizeRows(cv::Mat *histograms, int threadCount)
{
    if (!histograms)
    {
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(threadCount) schedule(static) if(threadCount > 1)
#endif
    for (int row = 0; row < histograms->rows; ++row)
    {
        cv::Mat current = histograms->row(row);
        const double norm = cv::norm(current, cv::NORM_L2);
        if (norm > 0.0)
        {
            current /= norm;
        }
    }
}

bool containsIndex(const std::vector<int> &indices, int value)
{
    return std::find(indices.begin(), indices.end(), value) != indices.end();
}

double descriptorDistanceSquared(const float *a, const float *b, int cols)
{
    double distance = 0.0;
    for (int col = 0; col < cols; ++col)
    {
        const double delta = static_cast<double>(a[col]) - static_cast<double>(b[col]);
        distance += delta * delta;
    }
    return distance;
}

std::vector<int> nearestRows(const cv::Mat &source, const cv::Mat &target)
{
    std::vector<int> nearest(static_cast<std::size_t>(source.rows), -1);
    for (int row = 0; row < source.rows; ++row)
    {
        const float *source_ptr = source.ptr<float>(row);
        double best_distance = std::numeric_limits<double>::max();
        int best_index = -1;
        for (int candidate = 0; candidate < target.rows; ++candidate)
        {
            const double distance = descriptorDistanceSquared(source_ptr, target.ptr<float>(candidate), source.cols);
            if (distance < best_distance)
            {
                best_distance = distance;
                best_index = candidate;
            }
        }
        nearest[static_cast<std::size_t>(row)] = best_index;
    }
    return nearest;
}

cv::Mat sampleDescriptorRows(const cv::Mat &descriptors, int maxRows, std::vector<int> *sourceRows)
{
    if (sourceRows)
    {
        sourceRows->clear();
    }
    if (descriptors.empty())
    {
        return cv::Mat();
    }
    if (maxRows <= 0 || descriptors.rows <= maxRows)
    {
        if (sourceRows)
        {
            sourceRows->reserve(static_cast<std::size_t>(descriptors.rows));
            for (int row = 0; row < descriptors.rows; ++row)
            {
                sourceRows->push_back(row);
            }
        }
        return descriptors;
    }

    cv::Mat sampled(maxRows, descriptors.cols, descriptors.type());
    if (sourceRows)
    {
        sourceRows->reserve(static_cast<std::size_t>(maxRows));
    }

    const double step = static_cast<double>(descriptors.rows) / static_cast<double>(maxRows);
    for (int i = 0; i < maxRows; ++i)
    {
        const int row = std::min(descriptors.rows - 1, static_cast<int>(std::floor(i * step)));
        descriptors.row(row).copyTo(sampled.row(i));
        if (sourceRows)
        {
            sourceRows->push_back(row);
        }
    }
    return sampled;
}

int geometryInlierCount(const xjw::VocabularyImageFeatures &image_a,
                        const xjw::VocabularyImageFeatures &image_b,
                        const cv::Mat &descriptors_a,
                        const cv::Mat &descriptors_b,
                        double ransacThreshold,
                        int maxDescriptors)
{
    if (descriptors_a.rows < 8 || descriptors_b.rows < 8 ||
        image_a.keypoints.size() < static_cast<std::size_t>(descriptors_a.rows) ||
        image_b.keypoints.size() < static_cast<std::size_t>(descriptors_b.rows))
    {
        return 0;
    }

    std::vector<int> sampled_rows_a;
    std::vector<int> sampled_rows_b;
    const cv::Mat sampled_a = sampleDescriptorRows(descriptors_a, maxDescriptors, &sampled_rows_a);
    const cv::Mat sampled_b = sampleDescriptorRows(descriptors_b, maxDescriptors, &sampled_rows_b);
    if (sampled_a.rows < 8 || sampled_b.rows < 8)
    {
        return 0;
    }

    const std::vector<int> a_to_b = nearestRows(sampled_a, sampled_b);
    const std::vector<int> b_to_a = nearestRows(sampled_b, sampled_a);

    std::vector<cv::Point2f> points_a;
    std::vector<cv::Point2f> points_b;
    for (int row = 0; row < sampled_a.rows; ++row)
    {
        const int match_b = a_to_b[static_cast<std::size_t>(row)];
        if (match_b < 0 || match_b >= sampled_b.rows)
        {
            continue;
        }
        if (b_to_a[static_cast<std::size_t>(match_b)] != row)
        {
            continue;
        }
        const int source_row_a = sampled_rows_a[static_cast<std::size_t>(row)];
        const int source_row_b = sampled_rows_b[static_cast<std::size_t>(match_b)];
        points_a.push_back(image_a.keypoints[static_cast<std::size_t>(source_row_a)].pt);
        points_b.push_back(image_b.keypoints[static_cast<std::size_t>(source_row_b)].pt);
    }

    if (points_a.size() < 8)
    {
        return 0;
    }

    cv::Mat inlier_mask;
    cv::findFundamentalMat(points_a, points_b, cv::FM_RANSAC, ransacThreshold, 0.99, inlier_mask);
    if (inlier_mask.empty())
    {
        return 0;
    }

    int count = 0;
    for (int row = 0; row < inlier_mask.rows; ++row)
    {
        if (inlier_mask.at<uchar>(row, 0) != 0)
        {
            ++count;
        }
    }
    return count;
}


bool assignWordsWithTree(const std::vector<cv::Mat> &descriptors,
                         const xjw::HierarchicalVocabularyTree &tree,
                         const xjw::VocabularyOverlapConfig &config,
                         int threadCount,
                         std::vector<std::vector<int>> *wordsPerImage,
                         std::string *errorMsg)
{
    if (!wordsPerImage)
    {
        setError(errorMsg, "层次词汇树分配输出为空");
        return false;
    }

    const std::uint64_t total_rows = std::accumulate(
        descriptors.cbegin(), descriptors.cend(), std::uint64_t{0},
        [](std::uint64_t total, const cv::Mat &value)
        {
            return total + static_cast<std::uint64_t>(value.rows);
        });
    wordsPerImage->clear();
    wordsPerImage->resize(descriptors.size());
    std::uint64_t processed_rows = 0;
    const int image_batch = std::max(1, threadCount);
    for (int begin = 0; begin < static_cast<int>(descriptors.size()); begin += image_batch)
    {
        const int end = std::min(static_cast<int>(descriptors.size()), begin + image_batch);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threadCount) schedule(static) if(threadCount > 1)
#endif
        for (int image_index = begin; image_index < end; ++image_index)
        {
            const cv::Mat &image_descriptors = descriptors[static_cast<std::size_t>(image_index)];
            std::vector<int> &words = (*wordsPerImage)[static_cast<std::size_t>(image_index)];
            words.resize(static_cast<std::size_t>(image_descriptors.rows));
            for (int row = 0; row < image_descriptors.rows; ++row)
            {
                words[static_cast<std::size_t>(row)] = tree.quantize(
                    image_descriptors.ptr<float>(row),
                    image_descriptors.cols);
            }
        }

        for (int image_index = begin; image_index < end; ++image_index)
        {
            processed_rows += static_cast<std::uint64_t>(
                descriptors[static_cast<std::size_t>(image_index)].rows);
        }
        const int percent = total_rows == 0
            ? 49
            : 30 + static_cast<int>(std::min<std::uint64_t>(
                  19, processed_rows * 19 / total_rows));
        std::ostringstream stage;
        stage << "分配层次词汇树 " << processed_rows << '/' << total_rows;
        if (!reportProgress(config, stage.str(), percent, errorMsg))
        {
            return false;
        }
    }
    return true;
}

struct PairScoreAccumulator
{
    double score = 0.0;
    int sharedWordCount = 0;
};

using PairScoreMap = std::unordered_map<std::uint64_t, PairScoreAccumulator>;

void mergePairScoreMaps(PairScoreMap *target, const PairScoreMap &source)
{
    if (!target)
    {
        return;
    }
    for (const auto &entry : source)
    {
        PairScoreAccumulator &dst = (*target)[entry.first];
        dst.score += entry.second.score;
        dst.sharedWordCount += entry.second.sharedWordCount;
    }
}

PairScoreMap buildPairScoresInverted(const cv::Mat &histograms,
                                     const std::vector<std::vector<unsigned char>> &wordPresence,
                                     int threadCount)
{
    const int image_count = histograms.rows;
    const int vocabulary_size = histograms.cols;
    std::vector<std::vector<std::pair<int, float>>> word_images(static_cast<std::size_t>(vocabulary_size));

    for (int image_index = 0; image_index < image_count; ++image_index)
    {
        for (int word = 0; word < vocabulary_size; ++word)
        {
            if (wordPresence[static_cast<std::size_t>(image_index)][static_cast<std::size_t>(word)])
            {
                const float weight = histograms.at<float>(image_index, word);
                if (weight != 0.0f)
                {
                    word_images[static_cast<std::size_t>(word)].emplace_back(image_index, weight);
                }
            }
        }
    }

    PairScoreMap merged;
#ifdef _OPENMP
    const int local_count = std::max(1, threadCount);
    std::vector<PairScoreMap> local_maps(static_cast<std::size_t>(local_count));
#pragma omp parallel for num_threads(threadCount) schedule(dynamic) if(threadCount > 1)
    for (int word = 0; word < vocabulary_size; ++word)
    {
        const int tid =
#ifdef _OPENMP
            omp_get_thread_num();
#else
            0;
#endif
        PairScoreMap &local = local_maps[static_cast<std::size_t>(tid)];
        const auto &images_for_word = word_images[static_cast<std::size_t>(word)];
        for (std::size_t a = 0; a < images_for_word.size(); ++a)
        {
            for (std::size_t b = a + 1; b < images_for_word.size(); ++b)
            {
                const std::uint64_t key = pairKey(images_for_word[a].first, images_for_word[b].first);
                PairScoreAccumulator &acc = local[key];
                acc.score += static_cast<double>(images_for_word[a].second) *
                             static_cast<double>(images_for_word[b].second);
                ++acc.sharedWordCount;
            }
        }
    }
    for (const PairScoreMap &local : local_maps)
    {
        mergePairScoreMaps(&merged, local);
    }
#else
    (void)threadCount;
    for (int word = 0; word < vocabulary_size; ++word)
    {
        const auto &images_for_word = word_images[static_cast<std::size_t>(word)];
        for (std::size_t a = 0; a < images_for_word.size(); ++a)
        {
            for (std::size_t b = a + 1; b < images_for_word.size(); ++b)
            {
                const std::uint64_t key = pairKey(images_for_word[a].first, images_for_word[b].first);
                PairScoreAccumulator &acc = merged[key];
                acc.score += static_cast<double>(images_for_word[a].second) *
                             static_cast<double>(images_for_word[b].second);
                ++acc.sharedWordCount;
            }
        }
    }
#endif
    return merged;
}

int sharedWordCount(const std::vector<unsigned char> &left, const std::vector<unsigned char> &right)
{
    int count = 0;
    const std::size_t size = std::min(left.size(), right.size());
    for (std::size_t i = 0; i < size; ++i)
    {
        if (left[i] && right[i])
        {
            ++count;
        }
    }
    return count;
}

PairScoreMap buildPairScoresDense(const cv::Mat &histograms,
                                  const std::vector<std::vector<unsigned char>> &wordPresence,
                                  double minSimilarity,
                                  int threadCount)
{
    const int image_count = histograms.rows;
    PairScoreMap merged;

#ifdef _OPENMP
    const int local_count = std::max(1, threadCount);
    std::vector<PairScoreMap> local_maps(static_cast<std::size_t>(local_count));
#pragma omp parallel for num_threads(threadCount) schedule(dynamic) if(threadCount > 1)
    for (int i = 0; i < image_count; ++i)
    {
        const int tid = omp_get_thread_num();
        PairScoreMap &local = local_maps[static_cast<std::size_t>(tid)];
        for (int j = i + 1; j < image_count; ++j)
        {
            const double score = histograms.row(i).dot(histograms.row(j));
            if (score < minSimilarity)
            {
                continue;
            }
            PairScoreAccumulator acc;
            acc.score = score;
            acc.sharedWordCount = sharedWordCount(wordPresence[static_cast<std::size_t>(i)],
                                                  wordPresence[static_cast<std::size_t>(j)]);
            local[pairKey(i, j)] = acc;
        }
    }
    for (const PairScoreMap &local : local_maps)
    {
        mergePairScoreMaps(&merged, local);
    }
#else
    (void)threadCount;
    for (int i = 0; i < image_count; ++i)
    {
        for (int j = i + 1; j < image_count; ++j)
        {
            const double score = histograms.row(i).dot(histograms.row(j));
            if (score < minSimilarity)
            {
                continue;
            }
            PairScoreAccumulator acc;
            acc.score = score;
            acc.sharedWordCount = sharedWordCount(wordPresence[static_cast<std::size_t>(i)],
                                                  wordPresence[static_cast<std::size_t>(j)]);
            merged[pairKey(i, j)] = acc;
        }
    }
#endif
    return merged;
}

std::vector<std::uint64_t> selectTopCandidatePairs(const PairScoreMap &pairScores,
                                                   int imageCount,
                                                   int topK,
                                                   double minSimilarity,
                                                   bool mutualTopK)
{
    std::vector<std::vector<std::pair<double, int>>> per_image(static_cast<std::size_t>(imageCount));
    for (const auto &entry : pairScores)
    {
        if (entry.second.score < minSimilarity)
        {
            continue;
        }
        const auto [a, b] = decodePairKey(entry.first);
        per_image[static_cast<std::size_t>(a)].emplace_back(entry.second.score, b);
        per_image[static_cast<std::size_t>(b)].emplace_back(entry.second.score, a);
    }

    const int effective_top_k = std::max(1, topK);
    std::vector<std::vector<int>> top_indices(static_cast<std::size_t>(imageCount));
    for (int image_index = 0; image_index < imageCount; ++image_index)
    {
        auto &scored = per_image[static_cast<std::size_t>(image_index)];
        std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.first == rhs.first)
            {
                return lhs.second < rhs.second;
            }
            return lhs.first > rhs.first;
        });

        const int keep = std::min(effective_top_k, static_cast<int>(scored.size()));
        top_indices[static_cast<std::size_t>(image_index)].reserve(static_cast<std::size_t>(keep));
        for (int i = 0; i < keep; ++i)
        {
            top_indices[static_cast<std::size_t>(image_index)].push_back(scored[static_cast<std::size_t>(i)].second);
        }
    }

    std::vector<std::uint64_t> selected;
    selected.reserve(pairScores.size());
    for (const auto &entry : pairScores)
    {
        if (entry.second.score < minSimilarity)
        {
            continue;
        }
        const auto [a, b] = decodePairKey(entry.first);
        const bool a_has_b = containsIndex(top_indices[static_cast<std::size_t>(a)], b);
        const bool b_has_a = containsIndex(top_indices[static_cast<std::size_t>(b)], a);
        if (mutualTopK ? (a_has_b && b_has_a) : (a_has_b || b_has_a))
        {
            selected.push_back(entry.first);
        }
    }
    return selected;
}

std::vector<std::string> sourceIds(const std::vector<xjw::OverlapPairGraphSource> &sources)
{
    std::vector<std::string> ids;
    ids.reserve(sources.size());
    for (xjw::OverlapPairGraphSource source : sources)
    {
        ids.emplace_back(xjw::overlapPairGraphSourceId(source));
    }
    return ids;
}

} // namespace

namespace xjw {

bool VocabularyOverlapRetriever::retrieve(const std::vector<VocabularyImageFeatures> &images,
                                          const VocabularyOverlapConfig &config,
                                          VocabularyOverlapResult *result,
                                          std::string *errorMsg)
{
    if (!result)
    {
        setError(errorMsg, "输出结果指针为空");
        return false;
    }

    *result = VocabularyOverlapResult();

    if (images.size() < 2)
    {
        setError(errorMsg, "至少需要两张影像才能获取重叠对");
        return false;
    }

    if (!reportProgress(config, "检查输入", 2, errorMsg))
    {
        return false;
    }

    std::vector<cv::Mat> descriptors;
    descriptors.reserve(images.size());
    int descriptor_cols = -1;
    std::uint64_t total_descriptor_count = 0;

    for (std::size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat current = toFloatDescriptors(images[i].descriptors);
        if (current.empty() || current.rows <= 0 || current.cols <= 0)
        {
            std::ostringstream oss;
            oss << "影像缺少有效描述子: " << images[i].imagePath;
            setError(errorMsg, oss.str());
            return false;
        }
        if (!cv::checkRange(current))
        {
            std::ostringstream oss;
            oss << "影像描述子包含 NaN 或无穷值: " << images[i].imagePath;
            setError(errorMsg, oss.str());
            return false;
        }

        if (descriptor_cols < 0)
        {
            descriptor_cols = current.cols;
        }
        else if (current.cols != descriptor_cols)
        {
            setError(errorMsg, "描述子维度不一致，无法构建同一个词汇树");
            return false;
        }

        descriptors.push_back(current);
        total_descriptor_count += static_cast<std::uint64_t>(current.rows);
    }

    std::vector<cv::Mat> assignment_descriptors;
    assignment_descriptors.reserve(descriptors.size());
    std::uint64_t assigned_descriptor_count = 0;
    for (const cv::Mat &image_descriptors : descriptors)
    {
        cv::Mat sampled = sampleDescriptorRows(image_descriptors,
                                               config.maxDescriptorsPerImage,
                                               nullptr);
        assigned_descriptor_count += static_cast<std::uint64_t>(sampled.rows);
        assignment_descriptors.push_back(std::move(sampled));
    }
    result->totalDescriptorCount = total_descriptor_count;
    result->assignedDescriptorCount = assigned_descriptor_count;

    const int image_count = static_cast<int>(images.size());
    const int thread_count = effectiveThreadCount(config, image_count);
    if (!reportProgress(config, "采样训练描述子", 8, errorMsg))
    {
        return false;
    }

    const cv::Mat training = sampleTrainingDescriptors(assignment_descriptors, config);
    if (training.empty())
    {
        setError(errorMsg, "没有足够的训练描述子用于构建词汇树");
        return false;
    }

    if (!reportProgress(config, "训练层次词汇树", 12, errorMsg))
    {
        return false;
    }
    HierarchicalVocabularyTreeConfig tree_config;
    tree_config.branchFactor = config.branchFactor;
    tree_config.maximumDepth = config.treeDepth;
    tree_config.maximumLeaves = config.maxVocabularyWords;
    tree_config.kmeansMaxIterations = config.kmeansMaxIterations;
    tree_config.kmeansAttempts = config.kmeansAttempts;
    tree_config.kmeansEpsilon = config.kmeansEpsilon;
    tree_config.progressCallback = [&config, errorMsg](int completedNodes,
                                                       int estimatedNodes,
                                                       int depth)
    {
        const int percent = 12 + std::min(17, completedNodes * 17 / std::max(1, estimatedNodes));
        std::ostringstream stage;
        stage << "训练层次词汇树 节点=" << completedNodes << '/' << estimatedNodes
              << " 深度=" << depth << '/' << std::max(1, config.treeDepth);
        return reportProgress(config, stage.str(), percent, errorMsg);
    };
    HierarchicalVocabularyTree vocabulary_tree;
    if (!vocabulary_tree.train(training, tree_config, errorMsg))
    {
        return false;
    }
    const int vocabulary_size = vocabulary_tree.wordCount();
    result->vocabularySize = vocabulary_size;
    result->vocabularyTreeNodeCount = vocabulary_tree.nodeCount();
    result->vocabularyTreeDepth = vocabulary_tree.actualDepth();

    if (!reportProgress(config, "分配层次词汇树", 30, errorMsg))
    {
        return false;
    }

    std::vector<std::vector<int>> words_per_image;
    if (!assignWordsWithTree(assignment_descriptors,
                             vocabulary_tree,
                             config,
                             thread_count,
                             &words_per_image,
                             errorMsg))
    {
        return false;
    }

    if (!reportProgress(config, "构建 TF-IDF 直方图", 50, errorMsg))
    {
        return false;
    }

    cv::Mat histograms = cv::Mat::zeros(image_count, vocabulary_size, CV_32F);
    std::vector<std::vector<unsigned char>> word_presence(
        images.size(), std::vector<unsigned char>(static_cast<std::size_t>(vocabulary_size), 0));
    std::vector<int> document_frequency(static_cast<std::size_t>(vocabulary_size), 0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_count) schedule(dynamic) if(thread_count > 1)
#endif
    for (int image_index = 0; image_index < image_count; ++image_index)
    {
        for (int word : words_per_image[static_cast<std::size_t>(image_index)])
        {
            if (word < 0 || word >= vocabulary_size)
            {
                continue;
            }
            histograms.at<float>(image_index, word) += 1.0f;
            word_presence[static_cast<std::size_t>(image_index)][static_cast<std::size_t>(word)] = 1;
        }
    }

    for (int image_index = 0; image_index < image_count; ++image_index)
    {
        for (int word = 0; word < vocabulary_size; ++word)
        {
            if (word_presence[static_cast<std::size_t>(image_index)][static_cast<std::size_t>(word)])
            {
                ++document_frequency[static_cast<std::size_t>(word)];
            }
        }
    }

    if (config.useTfidf)
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_count) schedule(static) if(thread_count > 1)
#endif
        for (int row = 0; row < histograms.rows; ++row)
        {
            for (int word = 0; word < histograms.cols; ++word)
            {
                const auto word_index = static_cast<std::size_t>(word);
                const double idf = std::log((static_cast<double>(image_count) + 1.0) /
                                            (static_cast<double>(document_frequency[word_index]) + 1.0)) +
                                   1.0;
                histograms.at<float>(row, word) *= static_cast<float>(idf);
            }
        }
    }
    normalizeRows(&histograms, thread_count);

    if (!reportProgress(config, "筛选候选影像对", 65, errorMsg))
    {
        return false;
    }

    const bool use_inverted_index = config.useInvertedIndex;
    PairScoreMap pair_scores = use_inverted_index
        ? buildPairScoresInverted(histograms, word_presence, thread_count)
        : buildPairScoresDense(histograms, word_presence, 0.0, thread_count);

    std::vector<VocabularyOverlapPairResult> candidates;
    candidates.reserve(pair_scores.size());
    std::unordered_map<std::uint64_t, std::size_t> candidate_index_by_key;
    std::vector<OverlapPairGraphInputEdge> planner_input;
    planner_input.reserve(pair_scores.size());
    for (const auto &score_entry : pair_scores)
    {
        if (score_entry.second.score <= 0.0)
        {
            continue;
        }
        const auto [i, j] = decodePairKey(score_entry.first);

        VocabularyOverlapPairResult pair;
        pair.indexA = i;
        pair.indexB = j;
        pair.imagePathA = images[static_cast<std::size_t>(i)].imagePath;
        pair.imagePathB = images[static_cast<std::size_t>(j)].imagePath;
        pair.bowScore = score_entry.second.score;
        pair.sharedWordCount = score_entry.second.sharedWordCount;
        pair.accepted = false;
        candidate_index_by_key.insert({score_entry.first, candidates.size()});
        candidates.push_back(std::move(pair));

        OverlapPairGraphInputEdge graph_edge;
        graph_edge.indexA = i;
        graph_edge.indexB = j;
        graph_edge.bowScore = score_entry.second.score;
        graph_edge.sharedWordCount = score_entry.second.sharedWordCount;
        planner_input.push_back(graph_edge);
    }

    OverlapPairGraphPlannerOptions planner_options;
    planner_options.imageCount = image_count;
    planner_options.topK = config.topK;
    planner_options.minPairsPerImage = config.minPairsPerImage;
    planner_options.minSimilarity = config.minSimilarity;
    planner_options.mutualTopK = config.mutualTopK;
    planner_options.keepOneWayTopK = config.keepOneWayTopK;
    planner_options.cycleClosureMaxPairsPerImage = config.cycleClosureMaxPairsPerImage;
    planner_options.connectComponents = config.connectComponents;
    planner_options.useSequenceFallback = config.useSequenceFallback;
    planner_options.sequenceWindow = config.sequenceWindow;
    planner_options.closeSequenceLoop = config.closeSequenceLoop;
    planner_options.componentBridgeMaxPairs = config.componentBridgeMaxPairs;
    const OverlapPairGraphPlan graph_plan = OverlapPairGraphPlanner::plan(planner_input, planner_options);

    for (const OverlapPairGraphEdge &edge : graph_plan.edges)
    {
        if (!validPair(image_count, edge.indexA, edge.indexB))
        {
            continue;
        }

        const std::uint64_t key = pairKey(edge.indexA, edge.indexB);
        auto candidate_it = candidate_index_by_key.find(key);
        if (candidate_it == candidate_index_by_key.end())
        {
            VocabularyOverlapPairResult pair;
            pair.indexA = std::min(edge.indexA, edge.indexB);
            pair.indexB = std::max(edge.indexA, edge.indexB);
            pair.imagePathA = images[static_cast<std::size_t>(pair.indexA)].imagePath;
            pair.imagePathB = images[static_cast<std::size_t>(pair.indexB)].imagePath;
            pair.bowScore = edge.bowScore;
            pair.sharedWordCount = edge.sharedWordCount;
            pair.geometricInliers = edge.geometricInliers;
            pair.accepted = true;
            pair.sourceTypes = sourceIds(edge.sources);
            candidate_index_by_key.insert({key, candidates.size()});
            candidates.push_back(std::move(pair));
        }
        else
        {
            VocabularyOverlapPairResult &pair = candidates[candidate_it->second];
            pair.accepted = true;
            pair.rejectReason.clear();
            pair.sourceTypes = sourceIds(edge.sources);
        }
    }

    for (VocabularyOverlapPairResult &pair : candidates)
    {
        if (!pair.accepted && pair.rejectReason.empty())
        {
            pair.rejectReason = "not_selected_by_pair_graph_planner";
        }
    }

    auto pair_sort = [](const VocabularyOverlapPairResult &lhs, const VocabularyOverlapPairResult &rhs) {
        if (lhs.bowScore == rhs.bowScore)
        {
            if (lhs.geometricInliers == rhs.geometricInliers)
            {
                return std::tie(lhs.indexA, lhs.indexB) < std::tie(rhs.indexA, rhs.indexB);
            }
            return lhs.geometricInliers > rhs.geometricInliers;
        }
        return lhs.bowScore > rhs.bowScore;
    };
    std::sort(candidates.begin(), candidates.end(), pair_sort);

    if (config.geometryCheck)
    {
        if (!reportProgress(config, "几何验证候选对", 80, errorMsg))
        {
            return false;
        }

        const int accepted_candidate_count =
            static_cast<int>(std::count_if(candidates.begin(), candidates.end(), [](const auto &pair) {
                return pair.accepted;
            }));
        const int geometry_limit = config.geometryMaxCandidatePairs <= 0
            ? accepted_candidate_count
            : std::min(config.geometryMaxCandidatePairs, accepted_candidate_count);

        int verified_candidate_count = 0;
        for (int idx = 0; idx < static_cast<int>(candidates.size()); ++idx)
        {
            VocabularyOverlapPairResult &pair = candidates[static_cast<std::size_t>(idx)];
            if (!pair.accepted)
            {
                continue;
            }

            if (verified_candidate_count >= geometry_limit)
            {
                pair.accepted = false;
                pair.rejectReason = "超出几何验证候选上限";
                continue;
            }
            ++verified_candidate_count;

            pair.geometricInliers = geometryInlierCount(images[static_cast<std::size_t>(pair.indexA)],
                                                        images[static_cast<std::size_t>(pair.indexB)],
                                                        descriptors[static_cast<std::size_t>(pair.indexA)],
                                                        descriptors[static_cast<std::size_t>(pair.indexB)],
                                                        config.ransacThreshold,
                                                        config.geometryMaxDescriptors);
            if (pair.geometricInliers < config.minInliers)
            {
                pair.accepted = false;
                pair.rejectReason = "几何内点数不足";
            }
        }
    }

    result->candidates = std::move(candidates);
    for (const VocabularyOverlapPairResult &pair : result->candidates)
    {
        if (pair.accepted)
        {
            result->acceptedPairs.push_back(pair);
        }
    }

    std::sort(result->candidates.begin(), result->candidates.end(), pair_sort);
    std::sort(result->acceptedPairs.begin(), result->acceptedPairs.end(), pair_sort);

    if (!reportProgress(config, "完成", 100, errorMsg))
    {
        return false;
    }

    std::ostringstream detail;
    detail << "images=" << image_count << " vocabulary=" << result->vocabularySize
           << " vocabulary_tree_nodes=" << result->vocabularyTreeNodeCount
           << " vocabulary_tree_depth=" << result->vocabularyTreeDepth
           << " vocabulary_tree_branch=" << std::max(2, config.branchFactor) << " descriptor_dims=" << descriptor_cols
           << " descriptors_total=" << result->totalDescriptorCount
           << " descriptors_assigned=" << result->assignedDescriptorCount
           << " assignment_limit_per_image=" << config.maxDescriptorsPerImage << " pair_top_k=" << config.topK
           << " min_pairs_per_image=" << config.minPairsPerImage
           << " cycle_closure_budget_per_image=" << config.cycleClosureMaxPairsPerImage
           << " candidates=" << result->candidates.size() << " accepted=" << result->acceptedPairs.size() << ' '
           << graph_plan.detail << " assignment=hierarchical_kmeans_tree"
           << " pair_scoring=" << (use_inverted_index ? "inverted" : "dense") << " threads=" << thread_count
           << " cuda_requested=" << (config.useCuda ? 1 : 0) << " cuda_used=0";
    result->detail = detail.str();

    return true;
}

} // namespace xjw

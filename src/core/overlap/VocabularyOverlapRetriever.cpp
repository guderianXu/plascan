#include "VocabularyOverlapRetriever.h"

#include "OpenCvCompat.h"
#include <opencv2/flann.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
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
        return descriptors.clone();
    }

    cv::Mat converted;
    descriptors.convertTo(converted, CV_32F);
    return converted;
}

int boundedVocabularySize(const xjw::VocabularyOverlapConfig &config, int descriptorCount)
{
    const int branch = std::max(2, config.branchFactor);
    const int depth = std::max(1, config.treeDepth);
    const int max_words = std::max(2, config.maxVocabularyWords);

    long long requested = 1;
    for (int i = 0; i < depth; ++i)
    {
        requested *= branch;
        if (requested > max_words)
        {
            requested = max_words;
            break;
        }
    }

    const int clamped = static_cast<int>(std::min<long long>(requested, descriptorCount));
    return std::max(1, clamped);
}

cv::Mat sampleTrainingDescriptors(const std::vector<cv::Mat> &descriptors,
                                  const xjw::VocabularyOverlapConfig &config)
{
    const int per_image = std::max(1, config.samplePerImage);
    const int max_training = std::max(1, config.maxTrainingDescriptors);

    cv::Mat training;
    for (const cv::Mat &image_descriptors : descriptors)
    {
        if (training.rows >= max_training)
        {
            break;
        }

        const int take_count = std::min(per_image, image_descriptors.rows);
        const int stride = std::max(1, image_descriptors.rows / take_count);
        int taken = 0;
        for (int row = 0; row < image_descriptors.rows && taken < take_count && training.rows < max_training;
             row += stride, ++taken)
        {
            training.push_back(image_descriptors.row(row));
        }
    }
    return training;
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

std::pair<int, int> decodePairKey(std::uint64_t key)
{
    return {
        static_cast<int>(key >> 32),
        static_cast<int>(key & 0xffffffffu)
    };
}

int nearestWordBrute(const cv::Mat &descriptor, const cv::Mat &centers)
{
    int best_index = 0;
    double best_distance = std::numeric_limits<double>::max();
    for (int center = 0; center < centers.rows; ++center)
    {
        const double distance = cv::norm(descriptor, centers.row(center), cv::NORM_L2SQR);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_index = center;
        }
    }
    return best_index;
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

    cv::Mat sampled;
    sampled.reserve(maxRows);
    if (sourceRows)
    {
        sourceRows->reserve(static_cast<std::size_t>(maxRows));
    }

    const double step = static_cast<double>(descriptors.rows) / static_cast<double>(maxRows);
    for (int i = 0; i < maxRows; ++i)
    {
        const int row = std::min(descriptors.rows - 1, static_cast<int>(std::floor(i * step)));
        sampled.push_back(descriptors.row(row));
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

bool assignWordsWithFlann(const std::vector<cv::Mat> &descriptors,
                          const cv::Mat &centers,
                          std::vector<std::vector<int>> *wordsPerImage,
                          std::string *errorMsg)
{
    if (!wordsPerImage)
    {
        return false;
    }

    try
    {
        cv::Mat centers_contiguous = centers.isContinuous() ? centers : centers.clone();
        cv::Mat all_descriptors;
        std::vector<int> offsets;
        offsets.reserve(descriptors.size() + 1);
        offsets.push_back(0);

        for (const cv::Mat &image_descriptors : descriptors)
        {
            all_descriptors.push_back(image_descriptors);
            offsets.push_back(all_descriptors.rows);
        }

        if (all_descriptors.empty())
        {
            setError(errorMsg, "描述子为空，无法分配词汇");
            return false;
        }

        cv::flann::Index flann(centers_contiguous, cv::flann::KDTreeIndexParams(4));
        cv::Mat indices(all_descriptors.rows, 1, CV_32S);
        cv::Mat distances(all_descriptors.rows, 1, CV_32F);
        flann.knnSearch(all_descriptors, indices, distances, 1, cv::flann::SearchParams(64));

        wordsPerImage->clear();
        wordsPerImage->resize(descriptors.size());
        for (std::size_t image_index = 0; image_index < descriptors.size(); ++image_index)
        {
            const int begin = offsets[image_index];
            const int end = offsets[image_index + 1];
            std::vector<int> &words = (*wordsPerImage)[image_index];
            words.reserve(static_cast<std::size_t>(end - begin));
            for (int row = begin; row < end; ++row)
            {
                words.push_back(indices.at<int>(row, 0));
            }
        }
        return true;
    }
    catch (const cv::Exception &ex)
    {
        setError(errorMsg, ex.what());
        return false;
    }
    catch (const std::exception &ex)
    {
        setError(errorMsg, ex.what());
        return false;
    }
}

void assignWordsBrute(const std::vector<cv::Mat> &descriptors,
                      const cv::Mat &centers,
                      int threadCount,
                      std::vector<std::vector<int>> *wordsPerImage)
{
    wordsPerImage->clear();
    wordsPerImage->resize(descriptors.size());

#ifdef _OPENMP
#pragma omp parallel for num_threads(threadCount) schedule(dynamic) if(threadCount > 1)
#endif
    for (int image_index = 0; image_index < static_cast<int>(descriptors.size()); ++image_index)
    {
        const cv::Mat &image_descriptors = descriptors[static_cast<std::size_t>(image_index)];
        std::vector<int> words;
        words.reserve(static_cast<std::size_t>(image_descriptors.rows));
        for (int row = 0; row < image_descriptors.rows; ++row)
        {
            words.push_back(nearestWordBrute(image_descriptors.row(row), centers));
        }
        (*wordsPerImage)[static_cast<std::size_t>(image_index)] = std::move(words);
    }
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
    }

    const int image_count = static_cast<int>(images.size());
    const int thread_count = effectiveThreadCount(config, image_count);
    if (!reportProgress(config, "采样训练描述子", 8, errorMsg))
    {
        return false;
    }

    const cv::Mat training = sampleTrainingDescriptors(descriptors, config);
    if (training.empty())
    {
        setError(errorMsg, "没有足够的训练描述子用于构建词汇树");
        return false;
    }

    const int vocabulary_size = boundedVocabularySize(config, training.rows);
    result->vocabularySize = vocabulary_size;

    cv::Mat centers;
    if (vocabulary_size == 1)
    {
        centers = training.row(0).clone();
    }
    else
    {
        if (!reportProgress(config, "训练视觉词汇", 12, errorMsg))
        {
            return false;
        }
        cv::Mat labels;
        cv::setRNGSeed(0);
        cv::kmeans(training,
                   vocabulary_size,
                   labels,
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 1e-4),
                   3,
                   cv::KMEANS_PP_CENTERS,
                   centers);
    }

    if (!reportProgress(config, "分配视觉词汇", 30, errorMsg))
    {
        return false;
    }

    std::vector<std::vector<int>> words_per_image;
    bool used_flann_assignment = false;
    std::string flann_error;
    if (config.useFlannAssignment && centers.rows >= 2 && centers.cols >= 2)
    {
        used_flann_assignment = assignWordsWithFlann(descriptors, centers, &words_per_image, &flann_error);
    }
    if (!used_flann_assignment)
    {
        assignWordsBrute(descriptors, centers, thread_count, &words_per_image);
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
                const double idf = std::log((static_cast<double>(image_count) + 1.0) /
                                            (static_cast<double>(document_frequency[static_cast<std::size_t>(word)]) + 1.0)) +
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
        : buildPairScoresDense(histograms, word_presence, config.minSimilarity, thread_count);

    std::vector<std::uint64_t> selected_pair_keys = selectTopCandidatePairs(pair_scores,
                                                                            image_count,
                                                                            config.topK,
                                                                            config.minSimilarity,
                                                                            config.mutualTopK);

    std::vector<VocabularyOverlapPairResult> candidates;
    candidates.reserve(selected_pair_keys.size());
    for (std::uint64_t key : selected_pair_keys)
    {
        const auto score_it = pair_scores.find(key);
        if (score_it == pair_scores.end())
        {
            continue;
        }
        const auto [i, j] = decodePairKey(key);

        VocabularyOverlapPairResult pair;
        pair.indexA = i;
        pair.indexB = j;
        pair.imagePathA = images[static_cast<std::size_t>(i)].imagePath;
        pair.imagePathB = images[static_cast<std::size_t>(j)].imagePath;
        pair.bowScore = score_it->second.score;
        pair.sharedWordCount = score_it->second.sharedWordCount;
        pair.accepted = true;
        candidates.push_back(std::move(pair));
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

        const int geometry_limit = config.geometryMaxCandidatePairs <= 0
            ? static_cast<int>(candidates.size())
            : std::min(config.geometryMaxCandidatePairs, static_cast<int>(candidates.size()));

        for (int idx = 0; idx < static_cast<int>(candidates.size()); ++idx)
        {
            VocabularyOverlapPairResult &pair = candidates[static_cast<std::size_t>(idx)];
            if (idx >= geometry_limit)
            {
                pair.accepted = false;
                pair.rejectReason = "超出几何验证候选上限";
                continue;
            }

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
    detail << "images=" << image_count
           << " vocabulary=" << result->vocabularySize
           << " candidates=" << result->candidates.size()
           << " accepted=" << result->acceptedPairs.size()
           << " assignment=" << (used_flann_assignment ? "flann" : "brute")
           << " pair_scoring=" << (use_inverted_index ? "inverted" : "dense")
           << " threads=" << thread_count
           << " cuda_requested=" << (config.useCuda ? 1 : 0)
           << " cuda_used=0";
    if (config.useFlannAssignment && !used_flann_assignment && !flann_error.empty())
    {
        detail << " flann_fallback=" << flann_error;
    }
    result->detail = detail.str();

    return true;
}

} // namespace xjw

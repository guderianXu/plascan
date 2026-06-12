#include "VocabularyOverlapRetriever.h"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <unordered_set>

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

int nearestWord(const cv::Mat &descriptor, const cv::Mat &centers)
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

void normalizeRows(cv::Mat *histograms)
{
    if (!histograms)
    {
        return;
    }

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

int geometryInlierCount(const xjw::VocabularyImageFeatures &image_a,
                        const xjw::VocabularyImageFeatures &image_b,
                        const cv::Mat &descriptors_a,
                        const cv::Mat &descriptors_b,
                        double ransacThreshold)
{
    if (descriptors_a.rows < 8 || descriptors_b.rows < 8 ||
        image_a.keypoints.size() < static_cast<std::size_t>(descriptors_a.rows) ||
        image_b.keypoints.size() < static_cast<std::size_t>(descriptors_b.rows))
    {
        return 0;
    }

    const std::vector<int> a_to_b = nearestRows(descriptors_a, descriptors_b);
    const std::vector<int> b_to_a = nearestRows(descriptors_b, descriptors_a);

    std::vector<cv::Point2f> points_a;
    std::vector<cv::Point2f> points_b;
    for (int row = 0; row < descriptors_a.rows; ++row)
    {
        const int match_b = a_to_b[static_cast<std::size_t>(row)];
        if (match_b < 0 || match_b >= descriptors_b.rows)
        {
            continue;
        }
        if (b_to_a[static_cast<std::size_t>(match_b)] != row)
        {
            continue;
        }
        points_a.push_back(image_a.keypoints[static_cast<std::size_t>(row)].pt);
        points_b.push_back(image_b.keypoints[static_cast<std::size_t>(match_b)].pt);
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

    const int image_count = static_cast<int>(images.size());
    cv::Mat histograms = cv::Mat::zeros(image_count, vocabulary_size, CV_32F);
    std::vector<std::vector<unsigned char>> word_presence(
        images.size(), std::vector<unsigned char>(static_cast<std::size_t>(vocabulary_size), 0));
    std::vector<int> document_frequency(static_cast<std::size_t>(vocabulary_size), 0);

    for (int image_index = 0; image_index < image_count; ++image_index)
    {
        std::unordered_set<int> seen_words;
        for (int row = 0; row < descriptors[static_cast<std::size_t>(image_index)].rows; ++row)
        {
            const int word = nearestWord(descriptors[static_cast<std::size_t>(image_index)].row(row), centers);
            histograms.at<float>(image_index, word) += 1.0f;
            word_presence[static_cast<std::size_t>(image_index)][static_cast<std::size_t>(word)] = 1;
            seen_words.insert(word);
        }

        for (int word : seen_words)
        {
            ++document_frequency[static_cast<std::size_t>(word)];
        }
    }

    if (config.useTfidf)
    {
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
    normalizeRows(&histograms);

    std::vector<std::vector<int>> top_indices(images.size());
    const int top_k = std::max(1, config.topK);
    for (int row = 0; row < image_count; ++row)
    {
        std::vector<std::pair<double, int>> scored;
        for (int col = 0; col < image_count; ++col)
        {
            if (row == col)
            {
                continue;
            }
            const double score = histograms.row(row).dot(histograms.row(col));
            if (score >= config.minSimilarity)
            {
                scored.emplace_back(score, col);
            }
        }

        std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.first == rhs.first)
            {
                return lhs.second < rhs.second;
            }
            return lhs.first > rhs.first;
        });

        for (int i = 0; i < std::min(top_k, static_cast<int>(scored.size())); ++i)
        {
            top_indices[static_cast<std::size_t>(row)].push_back(scored[static_cast<std::size_t>(i)].second);
        }
    }

    for (int i = 0; i < image_count; ++i)
    {
        for (int j = i + 1; j < image_count; ++j)
        {
            const bool i_has_j = containsIndex(top_indices[static_cast<std::size_t>(i)], j);
            const bool j_has_i = containsIndex(top_indices[static_cast<std::size_t>(j)], i);
            if (config.mutualTopK ? !(i_has_j && j_has_i) : !(i_has_j || j_has_i))
            {
                continue;
            }

            VocabularyOverlapPairResult pair;
            pair.indexA = i;
            pair.indexB = j;
            pair.imagePathA = images[static_cast<std::size_t>(i)].imagePath;
            pair.imagePathB = images[static_cast<std::size_t>(j)].imagePath;
            pair.bowScore = histograms.row(i).dot(histograms.row(j));
            for (int word = 0; word < vocabulary_size; ++word)
            {
                if (word_presence[static_cast<std::size_t>(i)][static_cast<std::size_t>(word)] &&
                    word_presence[static_cast<std::size_t>(j)][static_cast<std::size_t>(word)])
                {
                    ++pair.sharedWordCount;
                }
            }

            pair.accepted = true;
            if (config.geometryCheck)
            {
                pair.geometricInliers = geometryInlierCount(images[static_cast<std::size_t>(i)],
                                                            images[static_cast<std::size_t>(j)],
                                                            descriptors[static_cast<std::size_t>(i)],
                                                            descriptors[static_cast<std::size_t>(j)],
                                                            config.ransacThreshold);
                if (pair.geometricInliers < config.minInliers)
                {
                    pair.accepted = false;
                    pair.rejectReason = "几何内点数不足";
                }
            }

            result->candidates.push_back(pair);
            if (pair.accepted)
            {
                result->acceptedPairs.push_back(pair);
            }
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
    std::sort(result->candidates.begin(), result->candidates.end(), pair_sort);
    std::sort(result->acceptedPairs.begin(), result->acceptedPairs.end(), pair_sort);

    std::ostringstream detail;
    detail << "images=" << image_count
           << " vocabulary=" << result->vocabularySize
           << " candidates=" << result->candidates.size()
           << " accepted=" << result->acceptedPairs.size();
    result->detail = detail.str();

    return true;
}

} // namespace xjw

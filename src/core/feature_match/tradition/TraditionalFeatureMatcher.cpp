#include "TraditionalFeatureMatcher.h"
#include "CudaSiftMatcher.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace
{

cv::Mat toDescriptorType(const cv::Mat &input, int cvType)
{
    if (input.empty())
    {
        return cv::Mat();
    }

    if (input.type() == cvType)
    {
        return input;
    }

    cv::Mat converted;
    input.convertTo(converted, cvType);
    return converted;
}

std::vector<cv::DMatch> filterKnnMatches(const std::vector<std::vector<cv::DMatch>> &knnMatches,
                                         float ratio)
{
    std::vector<cv::DMatch> filtered;
    filtered.reserve(knnMatches.size());

    for (const auto &candidate : knnMatches)
    {
        if (candidate.size() < 2)
        {
            continue;
        }

        const cv::DMatch &best = candidate[0];
        const cv::DMatch &second = candidate[1];
        if (best.distance < ratio * second.distance)
        {
            filtered.push_back(best);
        }
    }

    return filtered;
}

} // namespace

namespace xjw::feature_match::tradition
{

std::string TraditionalFeatureMatcher::normalizeAlgorithmName(const std::string &algorithmName)
{
    std::string normalized = algorithmName;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c)
    {
        return static_cast<char>(std::tolower(c));
    });

    if (normalized == "orb_bf_hamming" ||
        normalized == "sift_bf_l2" ||
        normalized == "sift_flann")
    {
        return normalized;
    }

    return "orb_bf_hamming";
}

xjw::feature_match::MatchResult TraditionalFeatureMatcher::match(const cv::Mat &descriptors0,
                                                                 const cv::Mat &descriptors1,
                                                                 int numKeypoints0,
                                                                 int numKeypoints1,
                                                                 const TraditionalMatchConfig &config)
{
    const std::vector<cv::DMatch> matches = matchDescriptors(descriptors0, descriptors1, config);

    xjw::feature_match::MatchResult result;
    result.cvMatches = matches;
    result.numMatches = static_cast<int>(matches.size());
    result.sourceAlgorithm = normalizeAlgorithmName(config.algorithmName);
    result.matches0.assign(std::max(0, numKeypoints0), -1);
    result.matches1.assign(std::max(0, numKeypoints1), -1);
    result.matchingScores0.assign(std::max(0, numKeypoints0), 0.0f);
    result.matchingScores1.assign(std::max(0, numKeypoints1), 0.0f);

    for (const cv::DMatch &dm : result.cvMatches)
    {
        if (dm.queryIdx < 0 || dm.queryIdx >= numKeypoints0 || dm.trainIdx < 0 || dm.trainIdx >= numKeypoints1)
        {
            continue;
        }

        const float score = 1.0f / (1.0f + std::max(0.0f, dm.distance));
        result.matches0[dm.queryIdx] = dm.trainIdx;
        result.matches1[dm.trainIdx] = dm.queryIdx;
        result.matchingScores0[dm.queryIdx] = score;
        result.matchingScores1[dm.trainIdx] = score;
    }

    return result;
}

std::vector<cv::DMatch> TraditionalFeatureMatcher::matchDescriptors(const cv::Mat &descriptors0,
                                                                    const cv::Mat &descriptors1,
                                                                    const TraditionalMatchConfig &config)
{
    if (descriptors0.empty() || descriptors1.empty())
    {
        return {};
    }

    const std::string normalizedAlgorithm = normalizeAlgorithmName(config.algorithmName);
    const float ratio = std::clamp(config.ratioTestThreshold, 0.1f, 0.99f);

    cv::Mat desc0;
    cv::Mat desc1;

    std::vector<std::vector<cv::DMatch>> knnForward;
    std::vector<std::vector<cv::DMatch>> knnReverse;

    if (normalizedAlgorithm == "orb_bf_hamming")
    {
        desc0 = toDescriptorType(descriptors0, CV_8U);
        desc1 = toDescriptorType(descriptors1, CV_8U);
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        matcher.knnMatch(desc0, desc1, knnForward, 2);
        matcher.knnMatch(desc1, desc0, knnReverse, 2);
    }
    else if (normalizedAlgorithm == "sift_bf_l2")
    {
        desc0 = toDescriptorType(descriptors0, CV_32F);
        desc1 = toDescriptorType(descriptors1, CV_32F);
        if (config.useCuda && CudaSiftMatcher::isAvailable())
        {
            knnForward = CudaSiftMatcher::knnMatchL2(desc0, desc1, 2, config.cudaDevice);
            knnReverse = CudaSiftMatcher::knnMatchL2(desc1, desc0, 2, config.cudaDevice);
        }
        else
        {
            cv::BFMatcher matcher(cv::NORM_L2, false);
            matcher.knnMatch(desc0, desc1, knnForward, 2);
            matcher.knnMatch(desc1, desc0, knnReverse, 2);
        }
    }
    else if (normalizedAlgorithm == "sift_flann")
    {
        desc0 = toDescriptorType(descriptors0, CV_32F);
        desc1 = toDescriptorType(descriptors1, CV_32F);
        cv::FlannBasedMatcher matcher;
        matcher.knnMatch(desc0, desc1, knnForward, 2);
        matcher.knnMatch(desc1, desc0, knnReverse, 2);
    }
    else
    {
        throw std::runtime_error("unsupported traditional matcher algorithm");
    }

    std::vector<cv::DMatch> filteredForward = filterKnnMatches(knnForward, ratio);
    if (!config.requireMutualConsistency)
    {
        return filteredForward;
    }

    std::vector<cv::DMatch> filteredReverse = filterKnnMatches(knnReverse, ratio);
    std::vector<int> reverseBest(desc1.rows, -1);
    for (const cv::DMatch &match : filteredReverse)
    {
        if (match.queryIdx >= 0 && match.queryIdx < static_cast<int>(reverseBest.size()))
        {
            reverseBest[match.queryIdx] = match.trainIdx;
        }
    }

    std::vector<cv::DMatch> mutualMatches;
    mutualMatches.reserve(filteredForward.size());
    for (const cv::DMatch &match : filteredForward)
    {
        if (match.trainIdx >= 0 && match.trainIdx < static_cast<int>(reverseBest.size()) &&
            reverseBest[match.trainIdx] == match.queryIdx)
        {
            mutualMatches.push_back(match);
        }
    }

    return mutualMatches;
}

} // namespace xjw::feature_match::tradition

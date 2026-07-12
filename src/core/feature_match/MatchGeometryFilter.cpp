#include "MatchGeometryFilter.h"

#include "OpenCvCompat.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/version.hpp>

namespace xjw::feature_match
{

namespace
{

MatchResult rebuildFilteredResult(const MatchResult &input,
                                  const std::vector<int> &keptIndices)
{
    MatchResult out;
    out.matches0.assign(input.matches0.size(), -1);
    out.matches1.assign(input.matches1.size(), -1);
    out.matchingScores0.assign(input.matches0.size(), 0.0f);
    out.matchingScores1.assign(input.matches1.size(), 0.0f);
    out.sourceAlgorithm = input.sourceAlgorithm;

    out.cvMatches.reserve(keptIndices.size());
    for (int idx : keptIndices)
    {
        if (idx < 0 || idx >= static_cast<int>(input.cvMatches.size()))
        {
            continue;
        }

        const cv::DMatch &match = input.cvMatches[idx];
        if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(out.matches0.size()))
        {
            continue;
        }
        if (match.trainIdx < 0 || match.trainIdx >= static_cast<int>(out.matches1.size()))
        {
            continue;
        }

        out.cvMatches.push_back(match);
        out.matches0[match.queryIdx] = match.trainIdx;
        out.matches1[match.trainIdx] = match.queryIdx;

        float score0 = 1.0f - match.distance;
        float score1 = score0;
        if (match.queryIdx < static_cast<int>(input.matchingScores0.size()))
        {
            score0 = input.matchingScores0[match.queryIdx];
        }
        if (match.trainIdx < static_cast<int>(input.matchingScores1.size()))
        {
            score1 = input.matchingScores1[match.trainIdx];
        }
        out.matchingScores0[match.queryIdx] = score0;
        out.matchingScores1[match.trainIdx] = score1;
    }

    out.numMatches = static_cast<int>(out.cvMatches.size());
    return out;
}

int minimumPointCount(OutlierMethod method)
{
    if (method == OutlierMethod::HomographyRansac || method == OutlierMethod::AffineRansac)
    {
        return 4;
    }
    return 8;
}

bool maskValueAt(const cv::Mat &mask, int index)
{
    if (mask.type() == CV_8U)
    {
        return mask.at<uchar>(index) != 0;
    }
    return mask.at<char>(index) != 0;
}

} // namespace

MatchResult MatchGeometryFilter::filter(const MatchResult &input,
                                        const std::vector<cv::KeyPoint> &keypoints0,
                                        const std::vector<cv::KeyPoint> &keypoints1,
                                        const OutlierFilterConfig &config,
                                        int *inlierCount)
{
    if (inlierCount)
    {
        *inlierCount = input.numMatches;
    }

    if (config.method == OutlierMethod::None || input.cvMatches.empty())
    {
        return input;
    }

    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    std::vector<int> validMatchIndices;
    points0.reserve(input.cvMatches.size());
    points1.reserve(input.cvMatches.size());
    validMatchIndices.reserve(input.cvMatches.size());

    for (int i = 0; i < static_cast<int>(input.cvMatches.size()); ++i)
    {
        const cv::DMatch &match = input.cvMatches[i];
        if (match.queryIdx < 0 || match.trainIdx < 0)
        {
            continue;
        }
        if (match.queryIdx >= static_cast<int>(keypoints0.size()) ||
            match.trainIdx >= static_cast<int>(keypoints1.size()))
        {
            continue;
        }

        points0.push_back(keypoints0[match.queryIdx].pt);
        points1.push_back(keypoints1[match.trainIdx].pt);
        validMatchIndices.push_back(i);
    }

    if (static_cast<int>(points0.size()) < minimumPointCount(config.method))
    {
        return input;
    }

    cv::Mat inlierMask;
    switch (config.method)
    {
    case OutlierMethod::FundamentalRansac:
        cv::findFundamentalMat(points0,
                               points1,
                               cv::FM_RANSAC,
                               config.reprojThreshold,
                               config.confidence,
                               config.maxIters,
                               inlierMask);
        break;
    case OutlierMethod::FundamentalUsacMagsac:
    {
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 5
        try
        {
            cv::UsacParams usacParams;
            usacParams.confidence = config.confidence;
            usacParams.maxIterations = config.maxIters;
            usacParams.threshold = config.reprojThreshold;
            usacParams.score = cv::SCORE_METHOD_MAGSAC;
            usacParams.loMethod = cv::LOCAL_OPTIM_SIGMA;
            usacParams.loIterations = 10;
            usacParams.loSampleSize = 14;
            usacParams.randomGeneratorState = config.randomSeed;
            // USAC 内部并行会使临界像对的内点数随线程调度波动。
            // 上层已按像对并行，这里串行采样以保证匹配图可重复。
            usacParams.isParallel = false;
            cv::findFundamentalMat(points0, points1, inlierMask, usacParams);
        }
        catch (const cv::Exception &)
        {
            inlierMask.release();
            cv::findFundamentalMat(points0,
                                   points1,
                                   cv::FM_RANSAC,
                                   config.reprojThreshold,
                                   config.confidence,
                                   config.maxIters,
                                   inlierMask);
        }
#else
        cv::findFundamentalMat(points0,
                               points1,
                               cv::FM_RANSAC,
                               config.reprojThreshold,
                               config.confidence,
                               config.maxIters,
                               inlierMask);
#endif
        break;
    }
    case OutlierMethod::HomographyRansac:
        cv::findHomography(points0,
                           points1,
                           cv::RANSAC,
                           config.reprojThreshold,
                           inlierMask,
                           config.maxIters,
                           config.confidence);
        break;
    case OutlierMethod::AffineRansac:
        cv::estimateAffine2D(points0,
                             points1,
                             inlierMask,
                             cv::RANSAC,
                             config.reprojThreshold,
                             config.maxIters,
                             config.confidence);
        break;
    case OutlierMethod::None:
    default:
        return input;
    }

    if (inlierMask.empty())
    {
        if (inlierCount)
        {
            *inlierCount = 0;
        }
        return rebuildFilteredResult(input, {});
    }

    std::vector<int> keptIndices;
    keptIndices.reserve(validMatchIndices.size());
    for (int i = 0; i < static_cast<int>(validMatchIndices.size()); ++i)
    {
        if (maskValueAt(inlierMask, i))
        {
            keptIndices.push_back(validMatchIndices[i]);
        }
    }

    if (inlierCount)
    {
        *inlierCount = static_cast<int>(keptIndices.size());
    }

    if (keptIndices.empty() || static_cast<int>(keptIndices.size()) < config.minInliers)
    {
        // 已请求几何验证时，低于最小内点数代表该像对验证失败。
        // 返回原始匹配会把 RANSAC 判定的粗差重新送入 SfM，污染全局轨迹。
        return rebuildFilteredResult(input, {});
    }

    return rebuildFilteredResult(input, keptIndices);
}

} // namespace xjw::feature_match

#include "MatchGeometryVerifier.h"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xjw::image_matching
{
namespace
{

bool maskAt(const cv::Mat &mask, int index)
{
    return mask.type() == CV_8U ? mask.at<unsigned char>(index) != 0
                                : mask.at<char>(index) != 0;
}

std::array<double, 9> matrix3x3(const cv::Mat &matrix)
{
    std::array<double, 9> result{};
    if (matrix.rows != 3 || matrix.cols != 3)
    {
        return result;
    }
    cv::Mat asDouble;
    matrix.convertTo(asDouble, CV_64F);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            result[static_cast<std::size_t>(row * 3 + column)] =
                asDouble.at<double>(row, column);
        }
    }
    return result;
}

float sampsonResidual(const std::array<double, 9> &fundamental,
                      const cv::Point2f &point0,
                      const cv::Point2f &point1)
{
    const double x0 = point0.x;
    const double y0 = point0.y;
    const double x1 = point1.x;
    const double y1 = point1.y;
    const double fx0x = fundamental[0] * x0 + fundamental[1] * y0 + fundamental[2];
    const double fx0y = fundamental[3] * x0 + fundamental[4] * y0 + fundamental[5];
    const double fx0z = fundamental[6] * x0 + fundamental[7] * y0 + fundamental[8];
    const double ftx1x = fundamental[0] * x1 + fundamental[3] * y1 + fundamental[6];
    const double ftx1y = fundamental[1] * x1 + fundamental[4] * y1 + fundamental[7];
    const double numerator = x1 * fx0x + y1 * fx0y + fx0z;
    const double denominator = fx0x * fx0x + fx0y * fx0y +
        ftx1x * ftx1x + ftx1y * ftx1y;
    if (!std::isfinite(denominator) || denominator <= 1e-15)
    {
        return -1.0f;
    }
    return static_cast<float>(std::sqrt((numerator * numerator) / denominator));
}

float homographyResidual(const std::array<double, 9> &homography,
                         const cv::Point2f &point0,
                         const cv::Point2f &point1)
{
    const double denominator = homography[6] * point0.x +
        homography[7] * point0.y + homography[8];
    if (std::abs(denominator) <= 1e-15)
    {
        return -1.0f;
    }
    const double projectedX = (homography[0] * point0.x +
                               homography[1] * point0.y + homography[2]) / denominator;
    const double projectedY = (homography[3] * point0.x +
                               homography[4] * point0.y + homography[5]) / denominator;
    return static_cast<float>(std::hypot(projectedX - point1.x, projectedY - point1.y));
}

cv::Mat estimateModel(const std::vector<cv::Point2f> &points0,
                      const std::vector<cv::Point2f> &points1,
                      const MatchGeometryOptions &options,
                      cv::Mat *inlierMask)
{
    if (options.model == GeometryModel::Homography)
    {
        return cv::findHomography(points0,
                                  points1,
                                  cv::RANSAC,
                                  options.reprojectionThresholdPixels,
                                  *inlierMask,
                                  options.maximumIterations,
                                  options.confidence);
    }
    if (options.model != GeometryModel::Fundamental)
    {
        throw std::invalid_argument("unsupported image matching geometry model");
    }

    try
    {
        cv::UsacParams params;
        params.confidence = options.confidence;
        params.maxIterations = options.maximumIterations;
        params.threshold = options.reprojectionThresholdPixels;
        params.score = cv::SCORE_METHOD_MAGSAC;
        params.loMethod = cv::LOCAL_OPTIM_SIGMA;
        params.loIterations = 10;
        params.loSampleSize = 14;
        params.randomGeneratorState = options.randomSeed;
        // 像对已经在任务层并行；USAC 自身保持串行才能让临界内点集合可重复。
        params.isParallel = false;
        return cv::findFundamentalMat(points0, points1, *inlierMask, params);
    }
    catch (const cv::Exception &)
    {
        inlierMask->release();
        return cv::findFundamentalMat(points0,
                                      points1,
                                      cv::FM_RANSAC,
                                      options.reprojectionThresholdPixels,
                                      options.confidence,
                                      options.maximumIterations,
                                      *inlierMask);
    }
}

} // namespace

MatchGeometryResult MatchGeometryVerifier::verify(const MatchResult &matches,
                                                  const FeatureSet &features0,
                                                  const FeatureSet &features1,
                                                  const MatchGeometryOptions &options)
{
    MatchGeometryResult result;
    result.model = options.model;
    result.inlierMask.assign(matches.cvMatches.size(), false);
    result.residualPixels.assign(matches.cvMatches.size(), -1.0f);

    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    std::vector<int> sourceIndices;
    points0.reserve(matches.cvMatches.size());
    points1.reserve(matches.cvMatches.size());
    sourceIndices.reserve(matches.cvMatches.size());
    for (int index = 0; index < static_cast<int>(matches.cvMatches.size()); ++index)
    {
        const cv::DMatch &match = matches.cvMatches[static_cast<std::size_t>(index)];
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= features0.size() || match.trainIdx >= features1.size())
        {
            continue;
        }
        points0.push_back(features0.keypoints[static_cast<std::size_t>(match.queryIdx)].pt);
        points1.push_back(features1.keypoints[static_cast<std::size_t>(match.trainIdx)].pt);
        sourceIndices.push_back(index);
    }
    result.validInputCount = static_cast<int>(points0.size());
    const int minimumModelPoints = options.model == GeometryModel::Homography ? 4 : 8;
    if (result.validInputCount < minimumModelPoints)
    {
        return result;
    }

    cv::Mat inlierMask;
    const cv::Mat model = estimateModel(points0, points1, options, &inlierMask);
    if (model.empty() || inlierMask.empty() || model.rows != 3 || model.cols != 3)
    {
        return result;
    }
    result.modelEstimated = true;
    result.matrix = matrix3x3(model);

    for (int validIndex = 0; validIndex < result.validInputCount; ++validIndex)
    {
        const int sourceIndex = sourceIndices[static_cast<std::size_t>(validIndex)];
        const bool inlier = maskAt(inlierMask, validIndex);
        result.inlierMask[static_cast<std::size_t>(sourceIndex)] = inlier;
        result.residualPixels[static_cast<std::size_t>(sourceIndex)] =
            options.model == GeometryModel::Homography
                ? homographyResidual(result.matrix,
                                     points0[static_cast<std::size_t>(validIndex)],
                                     points1[static_cast<std::size_t>(validIndex)])
                : sampsonResidual(result.matrix,
                                  points0[static_cast<std::size_t>(validIndex)],
                                  points1[static_cast<std::size_t>(validIndex)]);
        if (inlier)
        {
            ++result.inlierCount;
        }
    }
    result.passed = result.inlierCount >= std::max(0, options.minimumInliers);
    return result;
}

} // namespace xjw::image_matching

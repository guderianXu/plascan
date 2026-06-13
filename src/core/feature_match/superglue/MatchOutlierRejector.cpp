// =============================================================================
// 文件: MatchOutlierRejector.cpp
// 说明:
//   对 SuperGlue 匹配结果进行几何外点过滤。
//   支持：基础矩阵 RANSAC、基础矩阵 USAC_MAGSAC、单应矩阵 RANSAC、仿射变换 RANSAC。
//   USAC_MAGSAC 使用自适应阈值，对粗差剔除能力显著优于传统 RANSAC。
//   过滤后重建 MatchResult：外点匹配的 matches0/matches1 被置为 -1。
// =============================================================================
#include "MatchOutlierRejector.h"

#include "OpenCvCompat.h"
#include <opencv2/core/version.hpp>

namespace superglue {

// rebuildFilteredResult: 根据保留的匹配索引重建完整 MatchResult
// keptIndices: 内点匹配在 input.cv_matches 中的下标
// 外点匹配的 matches0/matches1 被重置为 -1，matching_scores 被置 0
static MatchResult rebuildFilteredResult(const MatchResult &input,
                                         const std::vector<int> &keptIndices)
{
    MatchResult out;
    out.matches0.assign(input.matches0.size(), -1);
    out.matches1.assign(input.matches1.size(), -1);
    out.matchingScores0.assign(input.matches0.size(), 0.0f);
    out.matchingScores1.assign(input.matches1.size(), 0.0f);
    out.sourceAlgorithm = input.sourceAlgorithm;

    out.cvMatches.reserve(keptIndices.size());
    for (int idx : keptIndices) {
        if (idx < 0 || idx >= static_cast<int>(input.cvMatches.size())) continue;
        const cv::DMatch &m = input.cvMatches[idx];
        if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(out.matches0.size())) continue;
        if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(out.matches1.size())) continue;

        out.cvMatches.push_back(m);
        out.matches0[m.queryIdx] = m.trainIdx;
        out.matches1[m.trainIdx] = m.queryIdx;

        float score0 = 1.0f - m.distance;
        float score1 = score0;
        if (m.queryIdx < static_cast<int>(input.matchingScores0.size())) {
            score0 = input.matchingScores0[m.queryIdx];
        }
        if (m.trainIdx < static_cast<int>(input.matchingScores1.size())) {
            score1 = input.matchingScores1[m.trainIdx];
        }
        out.matchingScores0[m.queryIdx] = score0;
        out.matchingScores1[m.trainIdx] = score1;
    }

    out.numMatches = static_cast<int>(out.cvMatches.size());
    return out;
}

MatchResult MatchOutlierRejector::filter(const MatchResult &input,
                                         const std::vector<cv::KeyPoint> &kpts0,
                                         const std::vector<cv::KeyPoint> &kpts1,
                                         const OutlierFilterConfig &config,
                                         int *inlierCount)
{
    if (inlierCount) *inlierCount = input.numMatches;
    // None 模式直接返回，不撤一匹配
    if (config.method == OutlierMethod::None) {
        return input;
    }
    if (input.cvMatches.empty()) {
        return input;
    }

    // 从 cv_matches 中提取有效匹配对的坐标，用于 RANSAC 几何估计
    std::vector<cv::Point2f> pts0;
    std::vector<cv::Point2f> pts1;
    std::vector<int> validMatchIndices; // 记录对应到 input.cv_matches 中的広常
    pts0.reserve(input.cvMatches.size());
    pts1.reserve(input.cvMatches.size());
    validMatchIndices.reserve(input.cvMatches.size());

    for (int i = 0; i < static_cast<int>(input.cvMatches.size()); ++i) {
        const cv::DMatch &m = input.cvMatches[i];
        if (m.queryIdx < 0 || m.trainIdx < 0) continue;
        if (m.queryIdx >= static_cast<int>(kpts0.size()) || m.trainIdx >= static_cast<int>(kpts1.size())) continue;
        pts0.push_back(kpts0[m.queryIdx].pt);
        pts1.push_back(kpts1[m.trainIdx].pt);
        validMatchIndices.push_back(i);
    }

    const int n = static_cast<int>(pts0.size());
    // 最少点数要求：基础矩阵需要 8 点，单应/仿射需要 4 点
    int minGeomPoints = 8;
    if (config.method == OutlierMethod::HomographyRansac || config.method == OutlierMethod::AffineRansac) 
    {
        minGeomPoints = 4;
    }
    if (n < minGeomPoints) 
    {
        return input; // 点数不足，无法运行 RANSAC，返回原始匹配
    }

    cv::Mat inlierMask; // 每个匹配对是否为内点的标记 (CV_8U 或 CV_8SC1)
    // 根据所选方法调用对应的 OpenCV 函数
    switch (config.method) 
    {
    case OutlierMethod::FundamentalRansac:
        // 基础矩阵 RANSAC：适用于非平面场景的对极几何关系
        cv::findFundamentalMat(pts0, pts1, cv::FM_RANSAC,
                               config.reprojThreshold,
                               config.confidence,
                               config.maxIters,
                               inlierMask);
        break;
    case OutlierMethod::FundamentalUsacMagsac:
    {
        // USAC_MAGSAC：自适应阈值的鲁棒估计，粗差剔除能力显著强于传统 RANSAC。
        // 对于 Fundamental 矩阵估计，MAGSAC 使用 sigma 自适应内点/外点边界，
        // 能在高外点比例（>50%）场景下仍稳定找到正确模型。
        // reprojThreshold 此处作为 sigma 参数（建议 1.0-2.0）。
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
            usacParams.isParallel = true;
            cv::findFundamentalMat(pts0, pts1, inlierMask, usacParams);
        }
        catch (const cv::Exception &)
        {
            // OpenCV 4.12+ 中 USAC 框架在最小样本求解失败时触发断言，
            // 此处捕获并降级到普通 RANSAC 以保证流程继续。
            inlierMask.release();
            cv::findFundamentalMat(pts0, pts1, cv::FM_RANSAC,
                                   config.reprojThreshold,
                                   config.confidence,
                                   config.maxIters,
                                   inlierMask);
        }
#else
        // OpenCV < 4.5 回退：使用 FM_RANSAC 配合更严格的阈值
        cv::findFundamentalMat(pts0, pts1, cv::FM_RANSAC,
                               config.reprojThreshold,
                               config.confidence,
                               config.maxIters,
                               inlierMask);
#endif
        break;
    }
    case OutlierMethod::HomographyRansac:
        // 单应矩阵 RANSAC：适用于平面场景或纯旋转影像匹配
        cv::findHomography(pts0, pts1, cv::RANSAC,
                           config.reprojThreshold,
                           inlierMask,
                           config.maxIters,
                           config.confidence);
        break;
    case OutlierMethod::AffineRansac:
        // 仿射变换 RANSAC：适用于远距离平行投影场景
        cv::estimateAffine2D(pts0, pts1,
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

    if (inlierMask.empty()) {
        return input; // RANSAC 失败，未能计算 inlierMask
    }

    // 收集内点匹配索引
    std::vector<int> kept;
    kept.reserve(validMatchIndices.size());
    for (int i = 0; i < static_cast<int>(validMatchIndices.size()); ++i) {
        bool isInlier = false;
        // inlierMask 元素类型可能是 CV_8U 或 CV_8SC1，需分别处理
        if (inlierMask.type() == CV_8U) {
            isInlier = inlierMask.at<uchar>(i) != 0;
        } else {
            isInlier = inlierMask.at<char>(i) != 0;
        }
        if (isInlier) {
            kept.push_back(validMatchIndices[i]); // 保留内点匹配对应的 cv_matches 下标
        }
    }

    if (inlierCount) *inlierCount = static_cast<int>(kept.size());
    if (kept.empty()) {
        return input; // RANSAC 未找到任何内点
    }
    // 内点数小于 minInliers 时认为几何关系质量不可靠，保持原始匹配
    if (static_cast<int>(kept.size()) < config.minInliers) {
        return input;
    }

    // 根据保留的内点重建 matchResult
    return rebuildFilteredResult(input, kept);
}

} // namespace superglue

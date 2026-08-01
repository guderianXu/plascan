#pragma once

/**
 * @file SfmPairPlanner.h
 * @brief 空三匹配候选对的统一规划、评分和去重策略。
 *
 * 规划器只决定“哪些 pair 值得读取或匹配”，不读取特征、不执行匹配，也不判断
 * 几何内点。候选来源按可靠性组合：
 * - 调用方显式限制的 pair；
 * - 已知相机视锥重叠 pair；
 * - 照片序列局部窗口和首尾闭环；
 * - 已知相机中心的空间近邻。
 *
 * 同一 pair 可被多个来源命中。规划器以规范化绝对路径作为无向键，将来源和分数
 * 合并后统一排序。输出顺序仅用于调度优先级，绝不能当作几何正确性的证明。
 */

#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <QString>
#include <QStringList>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace xjw::aerial_triangulation
{

struct SfmPairPlannerOptions
{
    bool restrictPairs = false; ///< true 时只使用 allowedPairs，不再自动推导。
    QStringList allowedPairs; ///< 规范 pair key，格式为两个绝对路径以换行分隔。
    bool autoRestrictKnownCameraPairs = true; ///< 大规模数据是否启用自动候选裁剪。
    int knownCameraPairWindow = 4; ///< 序列中每张影像向后连接的最大索引距离。
    int knownCameraAllPairsMaxImages = 20; ///< 不超过该规模时保留全量 pair。
    int knownCameraSpatialNeighborCount = 8; ///< 每台相机保留的空间最近邻数量。
    bool knownCameraSequenceLoopClosure = false; ///< 将序列首尾视为相邻并补闭环边。
    std::vector<std::array<double, 3>> knownCameraCenters; ///< 与 images 同序的世界坐标相机中心。
    std::vector<std::array<double, 3>> knownCameraViewingDirections; ///< 与 images 同序的世界系视线方向。
    std::vector<std::array<int, 2>> knownCameraOverlapPairs; ///< 已由视锥/基线模块确认的索引对。
    double knownCameraOverlapMaxExpansion = 2.0; ///< 重叠对相对局部预算的最大膨胀倍数。
};

/// 一条规范化候选 pair 及其可解释评分分量。
struct SfmPairCandidate
{
    int indexA = -1; ///< 较小的影像索引；手工 key 无法反解时保持 -1。
    int indexB = -1; ///< 较大的影像索引；手工 key 无法反解时保持 -1。
    QString pairKey; ///< 与路径顺序无关的稳定无向 pair key。
    QStringList sourceTypes; ///< 命中来源，用于日志和质量报告解释。
    double priorityScore = 0.0; ///< 各来源非负分数之和，越大越优先。
    double overlapScore = 0.0; ///< 已知视锥重叠置信度，本实现为 0 或 1。
    double sequenceScore = 0.0; ///< 序列距离越近越接近 1。
    double spatialScore = 0.0; ///< 在空间近邻排名中越靠前越接近 1。
    double baselineScore = 0.0; ///< 由相机中心距离压缩到 [0, 1] 的调度分数。
    double orientationScore = 0.0; ///< 两相机视线点积的非负部分。
    int sequenceDistance = 0; ///< 最短有效序列索引距离；未知时为 0。
    double centerDistance = -1.0; ///< 世界系相机中心距离；未知时为 -1。
    double orientationAngleDeg = -1.0; ///< 两视线夹角，单位度；未知时为 -1。
};

/// 一次候选对规划结果及实际采用的裁剪依据。
struct SfmPairPlan
{
    bool restrictPairs = false; ///< false 表示下游应扫描全量 pair。
    bool autoRestricted = false; ///< 是否由已知相机/序列信息自动裁剪。
    bool usedCameraOverlapPairs = false; ///< 是否最终仅采用已知视锥重叠集合。
    bool usedSpatialCameraCenters = false; ///< 是否加入相机中心近邻边。
    bool usedSequenceLoopClosure = false; ///< 是否实际加入跨越序列边界的闭环边。
    int allPairCount = 0; ///< N(N-1)/2，用于报告裁剪比例。
    int knownCameraPairWindow = 0; ///< 本轮实际使用的序列窗口。
    int knownCameraSpatialNeighborCount = 0; ///< 本轮实际使用的空间近邻数。
    int knownCameraOverlapPairCount = 0; ///< 接受前的去重重叠 pair 数。
    QStringList allowedPairKeys; ///< 按优先级排序、供匹配读取器直接过滤的 key。
    std::vector<SfmPairCandidate> pairCandidates; ///< 与 allowedPairKeys 同序的解释记录。
};

/// 将影像路径转换为 cleanPath 形式的绝对路径；空白输入返回空串。
inline QString canonicalSfmPath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

/**
 * @brief 构造与输入顺序无关的 pair key。
 * @return 两个规范绝对路径按字典序排列并以换行连接；非法或同一路径返回空串。
 */
inline QString canonicalSfmPairKey(const QString &pathA, const QString &pathB)
{
    const QString normA = canonicalSfmPath(pathA);
    const QString normB = canonicalSfmPath(pathB);
    if (normA.isEmpty() || normB.isEmpty() || normA == normB)
    {
        return QString();
    }
    return (normA < normB)
        ? (normA + QStringLiteral("\n") + normB)
        : (normB + QStringLiteral("\n") + normA);
}

/// 保持首次出现顺序，移除空 key 和重复 key。
inline QStringList uniqueNonEmptyPairKeys(const QStringList &pairKeys)
{
    QStringList result;
    QSet<QString> seen;
    for (const QString &pairKey : pairKeys)
    {
        const QString trimmedKey = pairKey.trimmed();
        if (trimmedKey.isEmpty() || seen.contains(trimmedKey))
        {
            continue;
        }
        seen.insert(trimmedKey);
        result.append(trimmedKey);
    }
    return result;
}

/// 检查 images 与 cameraPaths 是否一一对应且每个路径均可规范化。
inline bool hasCompleteCameraPathList(const QStringList &images, const QStringList &cameraPaths)
{
    if (images.isEmpty() || cameraPaths.size() != images.size())
    {
        return false;
    }

    for (int i = 0; i < images.size(); ++i)
    {
        if (canonicalSfmPath(images.at(i)).isEmpty() || canonicalSfmPath(cameraPaths.at(i)).isEmpty())
        {
            return false;
        }
    }
    return true;
}

/// 检查待处理影像列表是否完整，可用于构造稳定 pair key。
inline bool hasCompleteSfmImageList(const QStringList &images)
{
    if (images.isEmpty())
    {
        return false;
    }

    for (const QString &image : images)
    {
        if (canonicalSfmPath(image).isEmpty())
        {
            return false;
        }
    }
    return true;
}

/// 检查相机中心是否与影像一一对应且全部为有限值。
inline bool hasCompleteKnownCameraCenters(int imageCount, const std::vector<std::array<double, 3>> &centers)
{
    if (imageCount <= 0 || centers.size() != static_cast<std::size_t>(imageCount))
    {
        return false;
    }

    for (const auto &center : centers)
    {
        if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2]))
        {
            return false;
        }
    }

    return true;
}

/// 相机中心欧氏距离的平方，供近邻排序避免不必要的开方。
inline double squaredCenterDistance(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

/// 三维向量的 L2 范数。
inline double vectorNorm3(const std::array<double, 3> &v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/// 检查世界系视线方向是否完整、有限且非零。
inline bool hasCompleteKnownCameraViewingDirections(
    int imageCount,
    const std::vector<std::array<double, 3>> &directions)
{
    if (imageCount <= 0 || directions.size() != static_cast<std::size_t>(imageCount))
    {
        return false;
    }

    for (const auto &direction : directions)
    {
        if (!std::isfinite(direction[0]) ||
            !std::isfinite(direction[1]) ||
            !std::isfinite(direction[2]) ||
            vectorNorm3(direction) <= 1e-12)
        {
            return false;
        }
    }

    return true;
}

struct SfmPairGeometryScores
{
    double baselineScore = 0.0; ///< d/(d+1)，用于排序而非摄影测量基高比。
    double orientationScore = 0.0; ///< max(0, cos(theta))，反向观察不会获得奖励。
    double centerDistance = -1.0; ///< 原始相机中心距离。
    double orientationAngleDeg = -1.0; ///< 原始视线夹角，单位度。
};

/**
 * @brief 计算一对已知相机的轻量调度分数。
 *
 * 该分数不替代重叠检测、基线质量评估或本质矩阵验证。距离归一化中的常数 1
 * 取决于输入坐标尺度，因此只作为同一数据集内部的弱排序项。
 */
inline SfmPairGeometryScores computeSfmPairGeometryScores(const SfmPairPlannerOptions &options,
                                                          int imageCount,
                                                          int indexA,
                                                          int indexB)
{
    SfmPairGeometryScores scores;
    if (indexA < 0 || indexB < 0 || indexA >= imageCount || indexB >= imageCount || indexA == indexB)
    {
        return scores;
    }

    if (hasCompleteKnownCameraCenters(imageCount, options.knownCameraCenters))
    {
        scores.centerDistance = std::sqrt(std::max(0.0,
            squaredCenterDistance(options.knownCameraCenters[static_cast<std::size_t>(indexA)],
                                  options.knownCameraCenters[static_cast<std::size_t>(indexB)])));
        scores.baselineScore = scores.centerDistance > 0.0
            ? std::clamp(scores.centerDistance / (scores.centerDistance + 1.0), 0.0, 1.0)
            : 0.0;
    }

    if (hasCompleteKnownCameraViewingDirections(imageCount, options.knownCameraViewingDirections))
    {
        const auto &dirA = options.knownCameraViewingDirections[static_cast<std::size_t>(indexA)];
        const auto &dirB = options.knownCameraViewingDirections[static_cast<std::size_t>(indexB)];
        const double normA = vectorNorm3(dirA);
        const double normB = vectorNorm3(dirB);
        if (normA > 1e-12 && normB > 1e-12)
        {
            const double dot = std::clamp((dirA[0] * dirB[0] + dirA[1] * dirB[1] + dirA[2] * dirB[2]) /
                                          (normA * normB),
                                          -1.0,
                                          1.0);
            scores.orientationScore = std::max(0.0, dot);
            scores.orientationAngleDeg = std::acos(dot) * 180.0 / M_PI;
        }
    }

    return scores;
}

/// 将合法索引对转换为 key，并写入保持首次出现顺序的集合。
inline void appendUniqueSfmPairKey(const QStringList &images,
                                   int indexA,
                                   int indexB,
                                   QSet<QString> *seen,
                                   QStringList *pairKeys)
{
    if (!seen || !pairKeys || indexA == indexB)
    {
        return;
    }

    const QString pairKey = canonicalSfmPairKey(images.at(indexA), images.at(indexB));
    if (pairKey.isEmpty() || seen->contains(pairKey))
    {
        return;
    }

    seen->insert(pairKey);
    pairKeys->append(pairKey);
}

/// 在当前候选列表中按规范 pair key 查找；候选规模受裁剪策略限制。
inline SfmPairCandidate *findSfmPairCandidate(std::vector<SfmPairCandidate> *candidates,
                                              const QString &pairKey)
{
    if (!candidates)
    {
        return nullptr;
    }

    for (SfmPairCandidate &candidate : *candidates)
    {
        if (candidate.pairKey == pairKey)
        {
            return &candidate;
        }
    }
    return nullptr;
}

/// 向候选追加一个不重复的可解释来源标签。
inline void appendSfmPairSource(SfmPairCandidate *candidate, const QString &sourceType)
{
    if (!candidate || sourceType.isEmpty() || candidate->sourceTypes.contains(sourceType))
    {
        return;
    }
    candidate->sourceTypes.append(sourceType);
}

/**
 * @brief 新建候选或合并同一 pair 的多个来源。
 *
 * priorityScore 采用累加，使同时被序列、空间和重叠策略支持的 pair 优先；
 * 各归一化分量采用最大值，原始距离/夹角保存最小有效值用于诊断。
 */
inline void addOrUpdateSfmPairCandidate(SfmPairPlan *plan,
                                        const QStringList &images,
                                        int indexA,
                                        int indexB,
                                        const QString &sourceType,
                                        double priorityScore,
                                        double overlapScore = 0.0,
                                        double sequenceScore = 0.0,
                                        double spatialScore = 0.0,
                                        int sequenceDistance = 0,
                                        double centerDistance = -1.0,
                                        double baselineScore = 0.0,
                                        double orientationScore = 0.0,
                                        double orientationAngleDeg = -1.0)
{
    if (!plan || indexA == indexB || indexA < 0 || indexB < 0 ||
        indexA >= images.size() || indexB >= images.size())
    {
        return;
    }

    const QString pairKey = canonicalSfmPairKey(images.at(indexA), images.at(indexB));
    if (pairKey.isEmpty())
    {
        return;
    }

    SfmPairCandidate *candidate = findSfmPairCandidate(&plan->pairCandidates, pairKey);
    if (!candidate)
    {
        SfmPairCandidate created;
        created.indexA = std::min(indexA, indexB);
        created.indexB = std::max(indexA, indexB);
        created.pairKey = pairKey;
        plan->pairCandidates.push_back(created);
        candidate = &plan->pairCandidates.back();
    }

    appendSfmPairSource(candidate, sourceType);
    candidate->priorityScore += std::max(0.0, priorityScore);
    candidate->overlapScore = std::max(candidate->overlapScore, overlapScore);
    candidate->sequenceScore = std::max(candidate->sequenceScore, sequenceScore);
    candidate->spatialScore = std::max(candidate->spatialScore, spatialScore);
    candidate->baselineScore = std::max(candidate->baselineScore, baselineScore);
    candidate->orientationScore = std::max(candidate->orientationScore, orientationScore);
    if (sequenceDistance > 0 &&
        (candidate->sequenceDistance == 0 || sequenceDistance < candidate->sequenceDistance))
    {
        candidate->sequenceDistance = sequenceDistance;
    }
    if (centerDistance >= 0.0 &&
        (candidate->centerDistance < 0.0 || centerDistance < candidate->centerDistance))
    {
        candidate->centerDistance = centerDistance;
    }
    if (orientationAngleDeg >= 0.0 &&
        (candidate->orientationAngleDeg < 0.0 || orientationAngleDeg < candidate->orientationAngleDeg))
    {
        candidate->orientationAngleDeg = orientationAngleDeg;
    }
}

/// 将调用方提供的不可反解 key 作为最高约束层的手工候选保存。
inline void addManualSfmPairCandidate(SfmPairPlan *plan, const QString &pairKey)
{
    if (!plan || pairKey.trimmed().isEmpty())
    {
        return;
    }

    SfmPairCandidate candidate;
    candidate.pairKey = pairKey.trimmed();
    candidate.priorityScore = 10.0;
    candidate.sourceTypes.append(QStringLiteral("manual_restricted"));
    plan->pairCandidates.push_back(candidate);
}

/// 按总优先级和稳定 key 排序，并同步生成下游过滤列表。
inline void finalizeSfmPairPlan(SfmPairPlan *plan)
{
    if (!plan)
    {
        return;
    }

    std::sort(plan->pairCandidates.begin(), plan->pairCandidates.end(),
              [](const SfmPairCandidate &lhs, const SfmPairCandidate &rhs)
    {
        if (lhs.priorityScore != rhs.priorityScore)
        {
            return lhs.priorityScore > rhs.priorityScore;
        }
        return lhs.pairKey < rhs.pairKey;
    });

    plan->allowedPairKeys.clear();
    for (const SfmPairCandidate &candidate : plan->pairCandidates)
    {
        if (!candidate.pairKey.isEmpty())
        {
            plan->allowedPairKeys.append(candidate.pairKey);
        }
    }
}

/// 计算非闭环序列滑动窗口可生成的无向 pair 数。
inline int slidingWindowPairCount(int imageCount, int window)
{
    if (imageCount <= 1 || window <= 0)
    {
        return 0;
    }

    int count = 0;
    for (int i = 0; i < imageCount; ++i)
    {
        count += std::min(window, imageCount - i - 1);
    }
    return count;
}

/**
 * @brief 判断视锥重叠模块给出的 pair 数是否仍具有裁剪意义。
 *
 * 过度膨胀通常意味着重叠先验异常或接近全连接；此时放弃该集合，回退到可控的
 * 序列窗口与空间近邻，避免一次错误先验把匹配复杂度重新推回 O(N^2)。
 */
inline bool acceptsKnownCameraOverlapPairs(int imageCount,
                                           int window,
                                           int spatialNeighborCount,
                                           int overlapPairCount,
                                           double maxExpansion)
{
    if (overlapPairCount <= 0)
    {
        return false;
    }

    const int sequenceBudget = slidingWindowPairCount(imageCount, window);
    const int spatialBudget = std::max(0, imageCount * std::max(0, spatialNeighborCount));
    const int boundedBudget = std::max(1, sequenceBudget + spatialBudget);
    const double expansion = std::max(1.0, maxExpansion);
    return static_cast<double>(overlapPairCount) <= static_cast<double>(boundedBudget) * expansion;
}

/**
 * @brief 生成一次完整候选对计划。
 *
 * 决策顺序：
 * 1. 显式 restrictPairs：原样采用手工 key；
 * 2. 小数据或先验不完整：不限制，下游使用全量 pair；
 * 3. 已知视锥重叠集合规模合理：直接采用该集合；
 * 4. 否则合并序列窗口、可选首尾闭环和相机中心近邻。
 *
 * cameraPaths 只用于判断已知相机输入是否完整，pair key 始终基于 images，
 * 从而与特征/匹配缓存的影像路径约定保持一致。
 */
inline SfmPairPlan planSfmMatchPairs(
    const QStringList &images,
    const QStringList &cameraPaths,
    const SfmPairPlannerOptions &options)
{
    SfmPairPlan plan;
    const int imageCount = images.size();
    plan.allPairCount = imageCount > 1 ? (imageCount * (imageCount - 1)) / 2 : 0;

    if (options.restrictPairs)
    {
        plan.restrictPairs = true;
        plan.allowedPairKeys = uniqueNonEmptyPairKeys(options.allowedPairs);
        for (const QString &pairKey : plan.allowedPairKeys)
        {
            addManualSfmPairCandidate(&plan, pairKey);
        }
        return plan;
    }

    const bool hasSequenceRestriction = options.knownCameraSequenceLoopClosure;
    const bool hasRestrictionInputs =
        hasCompleteCameraPathList(images, cameraPaths) ||
        hasCompleteKnownCameraCenters(imageCount, options.knownCameraCenters) ||
        !options.knownCameraOverlapPairs.empty() ||
        hasSequenceRestriction;

    if (!options.autoRestrictKnownCameraPairs ||
        imageCount <= std::max(0, options.knownCameraAllPairsMaxImages) ||
        !hasCompleteSfmImageList(images) ||
        !hasRestrictionInputs)
    {
        return plan;
    }

    const int window = std::max(1, options.knownCameraPairWindow);
    const int spatialNeighborCount = std::max(0, options.knownCameraSpatialNeighborCount);
    plan.restrictPairs = true;
    plan.autoRestricted = true;
    plan.knownCameraPairWindow = window;
    plan.knownCameraSpatialNeighborCount = spatialNeighborCount;

    // 视锥重叠是最强的候选先验：规模合理时无需再叠加启发式近邻。
    for (const auto &pair : options.knownCameraOverlapPairs)
    {
        const int indexA = pair[0];
        const int indexB = pair[1];
        if (indexA < 0 || indexA >= imageCount || indexB < 0 || indexB >= imageCount)
        {
            continue;
        }
        const SfmPairGeometryScores geometry =
            computeSfmPairGeometryScores(options, imageCount, indexA, indexB);
        addOrUpdateSfmPairCandidate(&plan,
                                    images,
                                    indexA,
                                    indexB,
                                    QStringLiteral("known_camera_overlap"),
                                    100.0 + 15.0 * geometry.orientationScore + 5.0 * geometry.baselineScore,
                                    1.0,
                                    0.0,
                                    0.0,
                                    0,
                                    geometry.centerDistance,
                                    geometry.baselineScore,
                                    geometry.orientationScore,
                                    geometry.orientationAngleDeg);
    }

    if (!plan.pairCandidates.empty())
    {
        plan.knownCameraOverlapPairCount = static_cast<int>(plan.pairCandidates.size());
        if (acceptsKnownCameraOverlapPairs(imageCount,
                                           window,
                                           spatialNeighborCount,
                                           plan.knownCameraOverlapPairCount,
                                           options.knownCameraOverlapMaxExpansion))
        {
            plan.usedCameraOverlapPairs = true;
            finalizeSfmPairPlan(&plan);
            return plan;
        }

        plan.pairCandidates.clear();
        plan.knownCameraOverlapPairCount = 0;
    }

    // 序列窗口提供稳定的局部骨架，尤其适合绕目标连续拍摄的数据。
    for (int i = 0; i < imageCount; ++i)
    {
        const int last = std::min(imageCount - 1, i + window);
        for (int j = i + 1; j <= last; ++j)
        {
            const int sequenceDistance = j - i;
            const double sequenceScore =
                static_cast<double>(window - sequenceDistance + 1) / static_cast<double>(window);
            const SfmPairGeometryScores geometry =
                computeSfmPairGeometryScores(options, imageCount, i, j);
            addOrUpdateSfmPairCandidate(&plan,
                                        images,
                                        i,
                                        j,
                                        QStringLiteral("sequence_window"),
                                        40.0 + 10.0 * sequenceScore +
                                            15.0 * geometry.orientationScore +
                                            5.0 * geometry.baselineScore,
                                        0.0,
                                        sequenceScore,
                                        0.0,
                                        sequenceDistance,
                                        geometry.centerDistance,
                                        geometry.baselineScore,
                                        geometry.orientationScore,
                                        geometry.orientationAngleDeg);
        }
    }

    if (options.knownCameraSequenceLoopClosure && imageCount > 2)
    {
        bool loopClosureAdded = false;
        const int loopWindow = std::min(window, imageCount - 1);
        for (int i = 0; i < imageCount; ++i)
        {
            for (int distance = 1; distance <= loopWindow; ++distance)
            {
                const int j = (i + distance) % imageCount;
                if (j > i || j == i)
                {
                    continue;
                }
                if (i - j <= window)
                {
                    // 非跨界的序列窗口边已经在上面的 sliding window 中加入。
                    // 再作为闭环边累计分数会把半圈远距离 pair 排到相邻 pair 前面，破坏 SfM 初始化。
                    continue;
                }

                // 照片序列常见于绕目标一圈拍摄，首尾相邻不能只靠 BoW 召回。
                // 显式补充闭环候选可避免 SfM 在序列尾段和首段之间断开。
                const double sequenceScore =
                    static_cast<double>(loopWindow - distance + 1) / static_cast<double>(loopWindow);
                const SfmPairGeometryScores geometry =
                    computeSfmPairGeometryScores(options, imageCount, i, j);
                addOrUpdateSfmPairCandidate(&plan,
                                            images,
                                            i,
                                            j,
                                            QStringLiteral("sequence_loop"),
                                            45.0 + 10.0 * sequenceScore +
                                                15.0 * geometry.orientationScore +
                                                5.0 * geometry.baselineScore,
                                            0.0,
                                            sequenceScore,
                                            0.0,
                                            distance,
                                            geometry.centerDistance,
                                            geometry.baselineScore,
                                            geometry.orientationScore,
                                            geometry.orientationAngleDeg);
                loopClosureAdded = true;
            }
        }
        plan.usedSequenceLoopClosure = loopClosureAdded;
    }

    // 空间近邻补充非序列拍摄或序列索引不可靠时的局部重叠候选。
    if (spatialNeighborCount > 0 && hasCompleteKnownCameraCenters(imageCount, options.knownCameraCenters))
    {
        plan.usedSpatialCameraCenters = true;
        for (int i = 0; i < imageCount; ++i)
        {
            std::vector<std::pair<double, int>> neighbors;
            neighbors.reserve(static_cast<std::size_t>(imageCount - 1));
            for (int j = 0; j < imageCount; ++j)
            {
                if (i == j)
                {
                    continue;
                }

                neighbors.emplace_back(squaredCenterDistance(options.knownCameraCenters[static_cast<std::size_t>(i)],
                                                             options.knownCameraCenters[static_cast<std::size_t>(j)]),
                                       j);
            }

            std::sort(neighbors.begin(), neighbors.end(), [](const auto &lhs, const auto &rhs)
            {
                if (lhs.first == rhs.first)
                {
                    return lhs.second < rhs.second;
                }
                return lhs.first < rhs.first;
            });

            const int keep = std::min(spatialNeighborCount, static_cast<int>(neighbors.size()));
            for (int n = 0; n < keep; ++n)
            {
                const auto &neighbor = neighbors[static_cast<std::size_t>(n)];
                const double spatialScore =
                    static_cast<double>(keep - n) / static_cast<double>(std::max(1, keep));
                const SfmPairGeometryScores geometry =
                    computeSfmPairGeometryScores(options, imageCount, i, neighbor.second);
                addOrUpdateSfmPairCandidate(&plan,
                                            images,
                                            i,
                                            neighbor.second,
                                            QStringLiteral("known_camera_spatial_neighbors"),
                                            60.0 + 10.0 * spatialScore +
                                                25.0 * geometry.orientationScore +
                                                5.0 * geometry.baselineScore,
                                            0.0,
                                            0.0,
                                            spatialScore,
                                            0,
                                            geometry.centerDistance,
                                            geometry.baselineScore,
                                            geometry.orientationScore,
                                            geometry.orientationAngleDeg);
            }
        }
    }

    finalizeSfmPairPlan(&plan);
    return plan;
}

} // namespace xjw::aerial_triangulation

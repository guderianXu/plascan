#pragma once

#include "FramePinholeCamera.h"
#include "MeshTypes.h"

#include <QVector>

#include <opencv2/core/mat.hpp>

#include <cstdint>

namespace xjw::mesh
{

struct MeshColorView
{
    FramePinholeCamera camera; ///< 深度、置信度与掩模所在分辨率的相机。
    FramePinholeCamera colorCamera; ///< 可选，颜色图所在分辨率的相机；无效时回退到 camera。
    cv::Mat colorBgr;
    cv::Mat depth;
    cv::Mat confidence;
    cv::Mat depthValidMask;
    cv::Mat supportMask;
    cv::Mat colorForegroundMask; ///< 可选，颜色图分辨率下可安全取色的前景区域。
    float qualityWeight = 1.0f;
};

struct MeshColorOptions
{
    float maximumVoxelSize = 0.0f;
    float depthToleranceVoxels = 6.0f;
    float relativeDepthTolerance = 0.005f;
    float visibilityToleranceVoxels = 4.0f;
    float minimumConfidence = 0.25f;
    float minimumViewCosine = 0.15f;
    int minimumConsistentViews = 1;
    int maximumBlendedViews = 3;
    int propagationPasses = 6;
    float propagationNormalCosine = 0.80f;
    int speckleCleanupPasses = 2;
    float speckleNormalCosine = 0.85f;
    float speckleMinimumColorDistance = 55.0f;
    float speckleMaximumNeighborDeviation = 28.0f;
    bool compensateExposure = false;
    bool coherentFacePrimaryViews = false;
    bool allowVisibilityOnlyFallback = false;
    int minimumVisibilityOnlyViews = 2;
    float visibilityOnlyMinimumViewCosine = 0.35f;
    int workerCount = 0;
};

struct MeshColorStatistics
{
    std::uint64_t candidateObservationCount = 0;
    std::uint64_t rejectedProjectionCount = 0;
    std::uint64_t rejectedMaskCount = 0;
    std::uint64_t rejectedDepthCount = 0;
    std::uint64_t rejectedVisibilityCount = 0;
    std::uint64_t rejectedViewAngleCount = 0;
    std::uint64_t rejectedColorOutlierCount = 0;
    std::uint64_t visibilityOnlyAttemptedObservationCount = 0;
    std::uint64_t visibilityOnlyCandidateObservationCount = 0;
    std::uint64_t visibilityOnlyRejectedForegroundCount = 0;
    std::uint64_t visibilityOnlyRejectedMissingForegroundCount = 0;
    std::uint64_t visibilityOnlyRejectedVisibilityCount = 0;
    std::uint64_t visibilityOnlyRejectedViewAngleCount = 0;
    int reliablyColoredVertexCount = 0;
    int bestViewFallbackVertexCount = 0;
    int visibilityOnlyFallbackVertexCount = 0;
    int propagatedVertexCount = 0;
    int fallbackVertexCount = 0;
    int cleanedSpeckleVertexCount = 0;
    int coherentPrimaryViewFaceCount = 0;
    int coherentPrimaryViewVertexCount = 0;
    int colorForegroundViewCount = 0;
    bool visibilityOnlyFallbackEnabled = false;
    int effectiveWorkerCount = 1;
    std::int64_t elapsedMs = 0;
};

class MeshColorizer
{
public:
    static MeshColorStatistics colorize(TriMesh *mesh,
                                        const QVector<MeshColorView> &views,
                                        const MeshColorOptions &options);
};

} // namespace xjw::mesh

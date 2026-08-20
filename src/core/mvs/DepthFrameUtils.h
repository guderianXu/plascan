#pragma once

#include "result/OperationResult.h"

#include "FramePinholeCamera.h"
#include "DepthMapFusion.h"
#include "MvsWorkspaceManifest.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <cstdint>
#include <vector>

namespace cv
{
class Mat;
}

namespace xjw::core::project
{

struct StoredDepthFrameRecord
{
    QString refImage;
    QString depthPng;
    QString rawDepthPath;
    QString rawConfidencePath;
    QString rawGeometrySupportPath;
    QString rawInverseDepthSpreadPath;
    QString rawAdaptiveGeometrySupportWeightPath;
    QString rawAdaptiveGeometryEffectiveViewCountPath;
    QString rawAdaptiveGeometryConflictRatioPath;
    QStringList sourceImages;
    QString device;
    QString configHash;
    QString projectInputSignature;
    QString reconstructionGenerationId;
    QJsonObject cameraModel;
    QString sceneProfile;
    QString acceptance;
    bool fusionEligible = false;
    bool fusionEligibilityKnown = false;
    bool useDiscreteGeometryFallback = false;
    int algorithmRevision = 0;
    int gridWidth = 0;
    int gridHeight = 0;
};

struct StoredDepthFramesResult
{
    xjw::common::OperationResult status;
    std::vector<StoredDepthFrameRecord> frames;
    QString batchDir;
};

struct FusionFrameBuildResult
{
    xjw::common::OperationResult status;
    xjw::mvs::FusionFrameInput frame;
    double readMs = 0.0;
    double postprocessMs = 0.0;
    double resizeMs = 0.0;
    double totalMs = 0.0;
};

std::uint64_t estimateFusionFrameWorkingSetBytes(int width,
                                                 int height,
                                                 int fusionMaxImageDim);
int recommendedDepthFrameLoadWorkers(int requestedWorkers,
                                     std::uint64_t availableMemoryBytes,
                                     std::uint64_t frameWorkingSetBytes);

QString rawDepthStoragePath(const QString &pngPath);
QString rawConfidenceStoragePath(const QString &pngPath);
QString rawGeometrySupportStoragePath(const QString &pngPath);
QString rawInverseDepthSpreadStoragePath(const QString &pngPath);
QString rawAdaptiveGeometrySupportWeightStoragePath(const QString &pngPath);
QString rawAdaptiveGeometryEffectiveViewCountStoragePath(const QString &pngPath);
QString rawAdaptiveGeometryConflictRatioStoragePath(const QString &pngPath);
xjw::common::OperationResult loadDepthMatStorage(const QString &path, cv::Mat *matrix);
xjw::common::OperationResult writeDepthMatStorage(const QString &path, const cv::Mat &matrix);
bool depthFrameArtifactsExist(const QString &pngPath, bool requireConfidence = false);
bool depthFrameArtifactsExist(const StoredDepthFrameRecord &frame, bool requireConfidence = false);

StoredDepthFramesResult collectLatestStoredDepthFrames(const QJsonObject &projectMeta);
StoredDepthFramesResult collectStoredDepthFramesForDirectory(const QJsonObject &projectMeta,
                                                             const QString &batchDirectory);
StoredDepthFramesResult selectFusionEligibleStoredDepthFrames(
    const StoredDepthFramesResult &storedFrames);
std::vector<int> storedFusionSourceIndices(const std::vector<StoredDepthFrameRecord> &frames,
                                           int referenceIndex);
bool downsampleFusionFrameForMaxDimension(xjw::mvs::FusionFrameInput *frame,
                                          int fusionMaxImageDim);
FusionFrameBuildResult buildStoredFusionFrame(const StoredDepthFrameRecord &stored,
                                              const xjw::FramePinholeCamera &camera,
                                              const xjw::mvs::FusionConfig &fusionConfig,
                                              int viewCount,
                                              int fusionMaxImageDim = 0);

} // namespace xjw::core::project

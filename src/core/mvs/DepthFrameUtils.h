#pragma once

#include "result/OperationResult.h"

#include "Camera.h"
#include "DepthMapFusion.h"

#include <QJsonObject>
#include <QString>

#include <vector>

namespace xjw::core::project
{

struct StoredDepthFrameRecord
{
    QString refImage;
    QString depthPng;
    QString rawDepthPath;
    QString rawConfidencePath;
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
};

QString rawDepthStoragePath(const QString &pngPath);
QString rawConfidenceStoragePath(const QString &pngPath);
bool depthFrameArtifactsExist(const QString &pngPath, bool requireConfidence = false);
bool depthFrameArtifactsExist(const StoredDepthFrameRecord &frame, bool requireConfidence = false);

StoredDepthFramesResult collectLatestStoredDepthFrames(const QJsonObject &projectMeta);
FusionFrameBuildResult buildStoredFusionFrame(const StoredDepthFrameRecord &stored,
                                              const xjw::Camera &camera,
                                              float confidenceThreshold,
                                              int viewCount);

} // namespace xjw::core::project

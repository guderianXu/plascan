#pragma once

#include "MvsTypes.h"

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project {

struct DenseGenerationSettings
{
    int atIndex = -1;
    QString outputDir;
    double resScale = 0.5;
    int iterations = 6;
    int threads = 8;
    int patchSize = 11;
    int minViews = 3;
    float patchMatchConfidence = 0.20f;
    bool useCuda = true;
    float fusionMinConfidence = 0.20f;
    int minConsistentViews = 2;
    float depthConsistency = 2.0f;
    float maxReprojError = 2.0f;
    bool pipelineMode = false;  // 流水线模式：跳过所有交互对话框
};

DenseGenerationSettings denseGenerationSettingsFromJson(const QJsonObject &settings);

xjw::mvs::DepthGenConfig buildDepthGenConfig(const DenseGenerationSettings &settings,
                                             int viewCount);

struct DenseRefineSettings
{
    bool sorEnabled = true;
    int sorK = 30;
    double sorStdDev = 2.0;
    bool voxelEnabled = false;
    double voxelSize = 0.005;
    bool normalsEnabled = true;
    int normalK = 30;
    bool smoothNormals = false;
    int threads = 8;
};

DenseRefineSettings denseRefineSettingsFromJson(const QJsonObject &settings);

} // namespace xjw::gui::project
#pragma once

#include "Camera.h"
#include "common/SfmTypes.h"

#include <QJsonObject>
#include <QMap>
#include <QStringList>

#include <array>
#include <vector>

namespace xjw::core::project
{

struct ProjectMatchObservationPair
{
    std::array<double, 2> pixelA{{0.0, 0.0}};
    std::array<double, 2> pixelB{{0.0, 0.0}};
    xjw::FeatureIdx featureA = xjw::kInvalidFeatureIdx;
    xjw::FeatureIdx featureB = xjw::kInvalidFeatureIdx;
    double score = 1.0;
};

struct ProjectMatchPair
{
    int cameraIndexA = -1;
    int cameraIndexB = -1;
    bool indexed = false;
    std::vector<ProjectMatchObservationPair> observations;
};

struct ProjectMatchInput
{
    std::vector<xjw::Camera> cameras;
    QStringList imagePathByIndex;
    QMap<QString, QJsonObject> beforeCamMeta;
    QMap<QString, int> cameraIndexByPath;
    std::vector<ProjectMatchPair> pairs;
    int sidecarV2PairCount = 0;
};

bool readProjectMatchInput(const QJsonObject &meta,
                           const QStringList &selectedImages,
                           int minMatches,
                           ProjectMatchInput *input);

int cameraIndexForImageToken(const QString &imageToken,
                             const QMap<QString, int> &cameraIndexByPath);

} // namespace xjw::core::project

#pragma once

#include "Camera.h"
#include "BundleAdjust.h"

#include <QJsonObject>
#include <QMap>
#include <QStringList>

#include <vector>

namespace xjw::core::project
{

enum class BaInputBuildStatus
{
    Ok,
    NotEnoughCameras,
    NoTracks
};

struct BaInputBuildResult
{
    std::vector<xjw::Camera> cameras;
    QStringList imagePathByIndex;
    QMap<QString, QJsonObject> beforeCamMeta;
    std::vector<xjw::BATrack> tracks;
};

BaInputBuildStatus buildBaInputFromMeta(const QJsonObject &meta,
                                        const QStringList &selectedImages,
                                        int minMatches,
                                        BaInputBuildResult *result);

} // namespace xjw::core::project

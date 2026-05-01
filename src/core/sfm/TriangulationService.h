#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::core::project
{

struct TriangulationServiceOptions
{
    QString outputDir;
    double minTriAngleDeg = 2.0;
    double maxReprojErrorPx = 2.0;
    int minObservations = 2;
    bool ignoreTwoViewTracks = false;
    int minTrackLength = 2;
};

struct TriangulationServiceResult
{
    bool success = false;
    QString errorMessage;
    QString sparseCloudPath;
    int exportedPointCount = 0;
    int candidateTrackCount = 0;
    int rejectedByObservationCount = 0;
    int rejectedByTriAngleCount = 0;
    int rejectedByReprojCount = 0;
    QJsonObject resultJson;
};

class TriangulationService
{
public:
    static TriangulationServiceResult run(const QJsonObject &meta,
                                          const QStringList &selectedImages,
                                          const TriangulationServiceOptions &options);
};

} // namespace xjw::core::project

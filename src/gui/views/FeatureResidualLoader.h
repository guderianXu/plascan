#pragma once

#include <QPointF>
#include <QString>
#include <QVector>

#include <opencv2/core/types.hpp>

#include <vector>

namespace xjw::gui::views
{

struct FeatureResidualVector
{
    QPointF observed;
    QPointF projected;
    double magnitudePx = 0.0;
};

struct ValidTiePointDiagnostics
{
    std::vector<cv::KeyPoint> keypoints;
    QVector<FeatureResidualVector> residuals;
    QString sidecarPath;
    QString message;
    qint64 loadMilliseconds = 0;
    bool available = false;
    bool usedUniqueNameFallback = false;
    bool loadedFromCache = false;
};

ValidTiePointDiagnostics loadValidTiePointDiagnosticsFromSidecar(
    const QString &sidecarPath,
    const QString &imagePath);

ValidTiePointDiagnostics loadValidTiePointDiagnosticsForImage(
    const QString &projectPath,
    const QString &imagePath);

QVector<FeatureResidualVector> loadFeatureResidualsForImage(const QString &projectPath,
                                                            const QString &imagePath,
                                                            QString *errorMessage = nullptr);

} // namespace xjw::gui::views

#pragma once

#include <QColor>
#include <QString>
#include <QVector>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace xjw::gui::tie_points
{

enum class ColorMode
{
    Color,
    Elevation,
    ImageCount
};

struct ScalarRange
{
    double minimum = 0.0;
    double maximum = 0.0;

    bool isValid() const;
    double normalize(double value) const;
};

struct ImageCountMetadata
{
    QVector<int> counts;
    QString errorMessage;

    bool isValidFor(qsizetype pointCount) const;
};

enum class QualityCriterion
{
    ReprojectionError,
    ReconstructionUncertainty,
    ImageCount,
    ProjectionAccuracy,
    MinimumTriangulationAngle
};

struct PrunePreviewQuery
{
    QualityCriterion criterion = QualityCriterion::ReprojectionError;
    double threshold = 0.0;

    bool isValid() const;
};

struct QualityMetadata
{
    QVector<double> reprojectionErrors;
    QVector<double> reconstructionUncertainties;
    QVector<int> imageCounts;
    QVector<double> projectionAccuracies;
    QVector<double> minimumTriangulationAngles;
    ScalarRange reprojectionErrorRange;
    ScalarRange reconstructionUncertaintyRange;
    ScalarRange imageCountRange;
    ScalarRange projectionAccuracyRange;
    ScalarRange minimumTriangulationAngleRange;
    qsizetype sourcePointCount = 0;
    QString errorMessage;

    bool isValidFor(qsizetype pointCount) const;
    bool hasCriterion(QualityCriterion criterion,
                      qsizetype pointCount = -1) const;
    ScalarRange range(QualityCriterion criterion) const;
};

struct PruneCandidateQueryResult
{
    std::vector<std::uint32_t> indices;
    qsizetype candidateCount = 0;
    QString errorMessage;

    bool succeeded() const { return errorMessage.isEmpty(); }
};

QColor elevationColor(double elevation, const ScalarRange &range);
QColor imageCountColor(int imageCount, const ScalarRange &range);
QColor scalarRampColor(double normalizedValue);
float pointSizeForMode(ColorMode mode);
QString inferSidecarPath(const QString &pointCloudPath);
ImageCountMetadata loadImageCountMetadata(const QString &sidecarPath);
QualityMetadata loadQualityMetadata(
    const QString &sidecarPath,
    const std::atomic_bool *cancellationFlag = nullptr);
PruneCandidateQueryResult queryPruneCandidates(
    const QualityMetadata &metadata,
    const PrunePreviewQuery &query,
    qsizetype pointCount,
    const std::atomic_bool *cancellationFlag = nullptr,
    std::size_t maximumReturnedIndices =
        std::numeric_limits<std::size_t>::max(),
    const std::vector<std::uint32_t> *excludedIndices = nullptr);

} // namespace xjw::gui::tie_points

#pragma once

#include "MeshTypes.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::qc
{

struct ProcessingBaselineMeshMetrics
{
    int vertexCount = 0;
    int faceCount = 0;
    int boundaryEdgeCount = 0;
    int nonManifoldEdgeCount = 0;
    int componentCount = 0;
    double largestComponentFaceRatio = 0.0;
    double highAspectFaceRatio = 0.0;
    double extremeAspectFaceRatio = 0.0;
    double adjacentNormalAngleMedianDegrees = 0.0;
    double adjacentNormalAngleP90Degrees = 0.0;
    double adjacentNormalAngleOver30Ratio = 0.0;
    double surfaceArea = 0.0;
    double boundingBoxDiagonal = 0.0;
    double normalizedSurfaceArea = 0.0;
};

struct ProcessingBaselineThresholds
{
    double maximumFaceCountRatio = 1.25;
    double maximumBoundaryEdgeCountRatio = 2.0;
    double maximumNormalizedSurfaceAreaRatio = 1.25;
    double maximumHighAspectFaceRatio = 0.01;
    double maximumExtremeAspectFaceRatio = 0.002;
    double maximumAdjacentNormalAngleMedianDegrees = 12.0;
    double maximumAdjacentNormalAngleP90Degrees = 30.0;
    double maximumAdjacentNormalAngleOver30Ratio = 0.15;
    int maximumComponentCount = 1;
    double minimumLargestComponentFaceRatio = 0.995;
};

struct ProcessingBaselineDefinition
{
    QString name;
    QString sceneType;
    QString createdAtUtc;
    QJsonObject inputSnapshot;
    QString inputFingerprintSha256;
    ProcessingBaselineMeshMetrics referenceMesh;
    ProcessingBaselineThresholds thresholds;
};

struct ProcessingBaselineComparison
{
    bool inputMatches = false;
    bool passed = false;
    QStringList failures;
    ProcessingBaselineMeshMetrics candidateMesh;
    QJsonObject report;
};

class ProcessingBaselineManager
{
public:
    static QString inputFingerprint(const QJsonObject &inputSnapshot);

    static ProcessingBaselineMeshMetrics analyzeMesh(
        const xjw::mesh::TriMesh &mesh);

    static ProcessingBaselineDefinition create(
        const QString &name,
        const QString &sceneType,
        const QJsonObject &inputSnapshot,
        const xjw::mesh::TriMesh &referenceMesh,
        const ProcessingBaselineThresholds &thresholds = {});

    static ProcessingBaselineComparison compare(
        const ProcessingBaselineDefinition &baseline,
        const QJsonObject &candidateInputSnapshot,
        const xjw::mesh::TriMesh &candidateMesh);

    static QJsonObject toJson(const ProcessingBaselineDefinition &baseline);

    static bool fromJson(const QJsonObject &object,
                         ProcessingBaselineDefinition *baseline,
                         QString *errorMessage = nullptr);

    static bool save(const QString &path,
                     const ProcessingBaselineDefinition &baseline,
                     QString *errorMessage = nullptr);

    static bool load(const QString &path,
                     ProcessingBaselineDefinition *baseline,
                     QString *errorMessage = nullptr);
};

} // namespace xjw::qc

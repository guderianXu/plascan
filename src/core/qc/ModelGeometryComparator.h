#pragma once

#include "MeshTypes.h"
#include "PointCloudAlignment.h"

#include <QString>

#include <vector>

namespace xjw::qc
{

struct ModelGeometryQuality
{
    int componentCount = 0;
    double largestComponentFaceRatio = 0.0;
    double largestFloatingDiagonalRatio = 0.0;
};

struct ReferenceGeometryQuality
{
    bool available = false;
    std::size_t sourcePointCount = 0;
    std::size_t referencePointCount = 0;
    double rmse = 0.0;
    double p50 = 0.0;
    double p84 = 0.0;
    double p95 = 0.0;
    double distanceThreshold = 0.0;
    double sourceCoverage = 0.0;
    double referenceCoverage = 0.0;
    QString error;
};

class ModelGeometryComparator
{
public:
    static ModelGeometryQuality analyzeMesh(const xjw::mesh::TriMesh &mesh);

    static ReferenceGeometryQuality comparePointSets(
        const std::vector<Point3D> &source,
        const std::vector<Point3D> &reference);

    static ReferenceGeometryQuality compareAlignedPointSets(
        const std::vector<Point3D> &source,
        const std::vector<Point3D> &reference,
        const SimilarityTransform *sourceToReference,
        bool cropReferenceToSourceBounds);

    static ReferenceGeometryQuality compareReferenceCloud(
        const xjw::mesh::TriMesh &mesh,
        const QString &referenceCloudPath,
        bool alignReferenceCloud,
        const SimilarityTransform *sourceToReference = nullptr,
        bool cropReferenceToSourceBounds = false);
};

} // namespace xjw::qc

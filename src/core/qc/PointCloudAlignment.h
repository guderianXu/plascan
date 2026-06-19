#pragma once

#include <QString>

#include <vector>

namespace xjw::qc
{

struct Point3D
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct SimilarityTransform
{
    double scale = 1.0;
    Point3D translation;
};

struct ErrorSummary
{
    double mean = 0.0;
    double rmse = 0.0;
    double median = 0.0;
    double p95 = 0.0;
};

struct PointCloudAlignmentResult
{
    bool success = false;
    QString error;
    QString method;
    int pairCount = 0;
    SimilarityTransform transform;
    ErrorSummary before;
    ErrorSummary after;
};

class PointCloudAlignment
{
public:
    static PointCloudAlignmentResult alignPairedSimilarity(const std::vector<Point3D> &source,
                                                          const std::vector<Point3D> &reference);

    static PointCloudAlignmentResult alignNearestNeighborTranslation(const std::vector<Point3D> &source,
                                                                    const std::vector<Point3D> &reference,
                                                                    int maxIterations = 8);

    static Point3D apply(const SimilarityTransform &transform, const Point3D &point);
};

} // namespace xjw::qc

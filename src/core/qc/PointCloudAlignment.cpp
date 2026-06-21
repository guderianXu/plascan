#include "PointCloudAlignment.h"

#include <plapoint/search/spatial_kdtree.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace xjw::qc
{

namespace
{

using AlignmentKdTree = plapoint::search::SpatialKdTree<3, double>;

Point3D operator+(const Point3D &a, const Point3D &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Point3D operator-(const Point3D &a, const Point3D &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Point3D operator*(double scale, const Point3D &point)
{
    return {scale * point.x, scale * point.y, scale * point.z};
}

double squaredNorm(const Point3D &point)
{
    return point.x * point.x + point.y * point.y + point.z * point.z;
}

double distance(const Point3D &a, const Point3D &b)
{
    return std::sqrt(squaredNorm(a - b));
}

Point3D centroid(const std::vector<Point3D> &points)
{
    Point3D sum;
    for (const Point3D &point : points)
    {
        sum = sum + point;
    }

    const double inv = points.empty() ? 0.0 : 1.0 / static_cast<double>(points.size());
    return inv * sum;
}

double percentileSorted(const std::vector<double> &sortedValues, double q)
{
    if (sortedValues.empty())
    {
        return 0.0;
    }

    const double clamped = std::max(0.0, std::min(1.0, q));
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(clamped * static_cast<double>(sortedValues.size() - 1)));
    return sortedValues[std::min(index, sortedValues.size() - 1)];
}

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 0)
    {
        return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
}

ErrorSummary summarizeErrors(std::vector<double> values)
{
    ErrorSummary summary;
    if (values.empty())
    {
        return summary;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double sumSquared = 0.0;
    for (double value : values)
    {
        sumSquared += value * value;
    }

    std::sort(values.begin(), values.end());
    summary.mean = sum / static_cast<double>(values.size());
    summary.rmse = std::sqrt(sumSquared / static_cast<double>(values.size()));
    summary.median = percentileSorted(values, 0.5);
    summary.p95 = percentileSorted(values, 0.95);
    return summary;
}

AlignmentKdTree buildAlignmentTree(const std::vector<Point3D> &points)
{
    std::vector<AlignmentKdTree::Point> treePoints;
    treePoints.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const Point3D &point = points[index];
        treePoints.push_back(AlignmentKdTree::Point{{point.x, point.y, point.z}, static_cast<int>(index)});
    }

    return AlignmentKdTree(treePoints);
}

Point3D nearestPoint(const std::vector<Point3D> &points,
                     const AlignmentKdTree &tree,
                     const Point3D &query)
{
    const int index = tree.nearest({query.x, query.y, query.z});
    if (index < 0 || index >= static_cast<int>(points.size()))
    {
        return {};
    }
    return points[static_cast<std::size_t>(index)];
}

std::vector<double> nearestNeighborErrors(const std::vector<Point3D> &source,
                                          const std::vector<Point3D> &reference,
                                          const AlignmentKdTree &referenceTree,
                                          const SimilarityTransform *transform)
{
    std::vector<double> errors;
    errors.reserve(source.size());
    for (const Point3D &point : source)
    {
        const Point3D aligned = transform ? PointCloudAlignment::apply(*transform, point) : point;
        errors.push_back(distance(aligned, nearestPoint(reference, referenceTree, aligned)));
    }
    return errors;
}

} // namespace

Point3D PointCloudAlignment::apply(const SimilarityTransform &transform, const Point3D &point)
{
    return transform.scale * point + transform.translation;
}

PointCloudAlignmentResult PointCloudAlignment::alignPairedSimilarity(const std::vector<Point3D> &source,
                                                                     const std::vector<Point3D> &reference)
{
    PointCloudAlignmentResult result;
    if (source.size() != reference.size())
    {
        result.error = QStringLiteral("源点云与参考点云点数不一致");
        return result;
    }
    if (source.size() < 2)
    {
        result.error = QStringLiteral("点云配准至少需要 2 对点");
        return result;
    }

    const Point3D sourceCenter = centroid(source);
    const Point3D referenceCenter = centroid(reference);

    double sourceVariance = 0.0;
    double referenceVariance = 0.0;
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        sourceVariance += squaredNorm(source[i] - sourceCenter);
        referenceVariance += squaredNorm(reference[i] - referenceCenter);
    }

    if (sourceVariance <= 1e-15 || referenceVariance <= 1e-15)
    {
        result.error = QStringLiteral("点云退化，无法估计尺度");
        return result;
    }

    result.transform.scale = std::sqrt(referenceVariance / sourceVariance);
    result.transform.translation = referenceCenter - result.transform.scale * sourceCenter;
    result.method = QStringLiteral("paired_similarity");

    std::vector<double> beforeErrors;
    std::vector<double> afterErrors;
    beforeErrors.reserve(source.size());
    afterErrors.reserve(source.size());
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        beforeErrors.push_back(distance(source[i], reference[i]));
        afterErrors.push_back(distance(apply(result.transform, source[i]), reference[i]));
    }

    result.pairCount = static_cast<int>(source.size());
    result.before = summarizeErrors(beforeErrors);
    result.after = summarizeErrors(afterErrors);
    result.success = true;
    return result;
}

PointCloudAlignmentResult PointCloudAlignment::alignNearestNeighborTranslation(const std::vector<Point3D> &source,
                                                                               const std::vector<Point3D> &reference,
                                                                               int maxIterations)
{
    PointCloudAlignmentResult result;
    result.method = QStringLiteral("nearest_neighbor_translation");
    if (source.empty() || reference.empty())
    {
        result.error = QStringLiteral("源点云和参考点云不能为空");
        return result;
    }

    SimilarityTransform transform;
    transform.scale = 1.0;
    const AlignmentKdTree referenceTree = buildAlignmentTree(reference);

    const int iterations = std::max(1, maxIterations);
    for (int iter = 0; iter < iterations; ++iter)
    {
        std::vector<double> dx;
        std::vector<double> dy;
        std::vector<double> dz;
        dx.reserve(source.size());
        dy.reserve(source.size());
        dz.reserve(source.size());

        for (const Point3D &point : source)
        {
            const Point3D aligned = apply(transform, point);
            const Point3D nearest = nearestPoint(reference, referenceTree, aligned);
            dx.push_back(nearest.x - aligned.x);
            dy.push_back(nearest.y - aligned.y);
            dz.push_back(nearest.z - aligned.z);
        }

        const Point3D delta{median(dx), median(dy), median(dz)};
        transform.translation = transform.translation + delta;
        if (squaredNorm(delta) <= 1e-18)
        {
            break;
        }
    }

    result.transform = transform;
    result.pairCount = static_cast<int>(source.size());
    result.before = summarizeErrors(nearestNeighborErrors(source, reference, referenceTree, nullptr));
    result.after = summarizeErrors(nearestNeighborErrors(source, reference, referenceTree, &result.transform));
    result.success = true;
    return result;
}

} // namespace xjw::qc

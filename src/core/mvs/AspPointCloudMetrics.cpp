#include "AspPointCloudMetrics.h"

#include "PointCloudTifIO.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>

#include <gdal_priv.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace xjw
{
namespace mvs
{
namespace
{

struct Point3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct LoadedCloud
{
    int width = 0;
    int height = 0;
    int valid = 0;
    std::array<double, 3> offset = {0.0, 0.0, 0.0};
    std::vector<Point3> points;
};

using PlaMetricCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

std::shared_ptr<PlaMetricCloud> toPlaCloud(const std::vector<Point3> &points)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> matrix(points.size(), 3);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        matrix(row, 0) = points[i].x;
        matrix(row, 1) = points[i].y;
        matrix(row, 2) = points[i].z;
    }
    return std::make_shared<PlaMetricCloud>(std::move(matrix));
}

bool readAspTif(const std::string &path, LoadedCloud &cloud, std::string *errorMessage)
{
    GDALAllRegister();
    GDALDataset *dataset = static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (!dataset)
    {
        if (errorMessage)
        {
            *errorMessage = "Failed to open ASP point cloud: " + path;
        }
        return false;
    }

    cloud.width = dataset->GetRasterXSize();
    cloud.height = dataset->GetRasterYSize();
    if (dataset->GetRasterCount() < 3)
    {
        GDALClose(dataset);
        if (errorMessage)
        {
            *errorMessage = "ASP point cloud has fewer than 3 bands: " + path;
        }
        return false;
    }

    const char *offsetText = dataset->GetMetadataItem("POINT_OFFSET");
    if (offsetText)
    {
        std::sscanf(offsetText, "%lf %lf %lf", &cloud.offset[0], &cloud.offset[1], &cloud.offset[2]);
    }

    cv::Mat x(cloud.height, cloud.width, CV_32FC1);
    cv::Mat y(cloud.height, cloud.width, CV_32FC1);
    cv::Mat z(cloud.height, cloud.width, CV_32FC1);
    dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cloud.width, cloud.height,
                                        x.data, cloud.width, cloud.height, GDT_Float32, 0, 0);
    dataset->GetRasterBand(2)->RasterIO(GF_Read, 0, 0, cloud.width, cloud.height,
                                        y.data, cloud.width, cloud.height, GDT_Float32, 0, 0);
    dataset->GetRasterBand(3)->RasterIO(GF_Read, 0, 0, cloud.width, cloud.height,
                                        z.data, cloud.width, cloud.height, GDT_Float32, 0, 0);
    GDALClose(dataset);

    cloud.points.reserve(static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height) / 2);
    for (int row = 0; row < cloud.height; ++row)
    {
        for (int col = 0; col < cloud.width; ++col)
        {
            const float px = x.at<float>(row, col);
            const float py = y.at<float>(row, col);
            const float pz = z.at<float>(row, col);
            if ((px == 0.0f && py == 0.0f && pz == 0.0f) ||
                !std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz))
            {
                continue;
            }
            cloud.points.push_back({static_cast<float>(px + cloud.offset[0]),
                                    static_cast<float>(py + cloud.offset[1]),
                                    static_cast<float>(pz + cloud.offset[2])});
        }
    }
    cloud.valid = static_cast<int>(cloud.points.size());
    return true;
}

bool readPlascanTif(const std::string &path, LoadedCloud &cloud, std::string *errorMessage)
{
    TriangulationResult tri;
    if (!PointCloudTifIO::readTif(path, tri, errorMessage))
    {
        return false;
    }

    cloud.width = tri.pointCloud.cols;
    cloud.height = tri.pointCloud.rows;
    cloud.offset = tri.pointOffset;
    cloud.points.reserve(static_cast<size_t>(std::max(0, tri.validPoints)));
    for (int row = 0; row < tri.pointCloud.rows; ++row)
    {
        for (int col = 0; col < tri.pointCloud.cols; ++col)
        {
            if (tri.validMask.at<uint8_t>(row, col) == 0)
            {
                continue;
            }
            const cv::Vec3d &p = tri.pointCloud.at<cv::Vec3d>(row, col);
            cloud.points.push_back({static_cast<float>(p[0] + tri.pointOffset[0]),
                                    static_cast<float>(p[1] + tri.pointOffset[1]),
                                    static_cast<float>(p[2] + tri.pointOffset[2])});
        }
    }
    cloud.valid = static_cast<int>(cloud.points.size());
    return true;
}

void updateBounds(const Point3 &p, Point3 &minP, Point3 &maxP)
{
    minP.x = std::min(minP.x, p.x);
    minP.y = std::min(minP.y, p.y);
    minP.z = std::min(minP.z, p.z);
    maxP.x = std::max(maxP.x, p.x);
    maxP.y = std::max(maxP.y, p.y);
    maxP.z = std::max(maxP.z, p.z);
}

void evaluatePass(const AspPointCloudMetricsThresholds &thresholds,
                  AspPointCloudMetricsResult &result)
{
    std::ostringstream failures;
    if (result.coverageRatio < thresholds.minCoverageRatio)
    {
        failures << "coverageRatio " << result.coverageRatio << " < " << thresholds.minCoverageRatio << "; ";
    }
    if (result.nnMedian > thresholds.maxMedianDistance)
    {
        failures << "nnMedian " << result.nnMedian << " > " << thresholds.maxMedianDistance << "; ";
    }
    if (result.nnP95 > thresholds.maxP95Distance)
    {
        failures << "nnP95 " << result.nnP95 << " > " << thresholds.maxP95Distance << "; ";
    }
    result.failureReason = failures.str();
    result.passed = result.failureReason.empty();
}

} // namespace

bool AspPointCloudMetrics::compare(const std::string &plascanTifPath,
                                   const std::string &aspTifPath,
                                   const AspPointCloudMetricsThresholds &thresholds,
                                   AspPointCloudMetricsResult &result,
                                   std::string *errorMessage)
{
    LoadedCloud asp;
    LoadedCloud plascan;
    if (!readAspTif(aspTifPath, asp, errorMessage) || !readPlascanTif(plascanTifPath, plascan, errorMessage))
    {
        return false;
    }
    if (asp.points.empty() || plascan.points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "Cannot compare empty point clouds";
        }
        return false;
    }

    result.aspWidth = asp.width;
    result.aspHeight = asp.height;
    result.aspValidPoints = asp.valid;
    result.plascanWidth = plascan.width;
    result.plascanHeight = plascan.height;
    result.plascanValidPoints = plascan.valid;
    result.coverageRatio = static_cast<double>(plascan.valid) / static_cast<double>(asp.valid);

    const float max_float = std::numeric_limits<float>::max();
    Point3 aspMin{max_float, max_float, max_float};
    Point3 aspMax{-max_float, -max_float, -max_float};
    for (const Point3 &p : asp.points)
    {
        updateBounds(p, aspMin, aspMax);
    }

    int insideBbox = 0;
    for (const Point3 &p : plascan.points)
    {
        const bool inside_x = p.x >= aspMin.x && p.x <= aspMax.x;
        const bool inside_y = p.y >= aspMin.y && p.y <= aspMax.y;
        const bool inside_z = p.z >= aspMin.z && p.z <= aspMax.z;
        if (inside_x && inside_y && inside_z)
        {
            ++insideBbox;
        }
    }
    result.bboxContainmentRatio = static_cast<double>(insideBbox) / static_cast<double>(plascan.points.size());

    auto aspCloud = toPlaCloud(asp.points);
    plapoint::search::KdTree<float, plamatrix::Device::CPU> aspTree;
    aspTree.setInputCloud(aspCloud);
    aspTree.build();

    const int sampleStep = std::max(1, static_cast<int>(plascan.points.size()) / 100000);
    std::vector<double> distances;
    double offsetX = 0.0;
    double offsetY = 0.0;
    double offsetZ = 0.0;
    int matched = 0;
    for (int index = 0; index < static_cast<int>(plascan.points.size()); index += sampleStep)
    {
        const Point3 &p = plascan.points[index];
        const auto neighbors = aspTree.nearestKSearch(plamatrix::Vec3<float>{p.x, p.y, p.z}, 1);
        if (neighbors.empty())
        {
            continue;
        }
        const int bestIndex = neighbors.front();
        if (bestIndex >= 0)
        {
            const Point3 &q = asp.points[bestIndex];
            const double ddx = static_cast<double>(p.x) - q.x;
            const double ddy = static_cast<double>(p.y) - q.y;
            const double ddz = static_cast<double>(p.z) - q.z;
            const double dist = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            distances.push_back(dist);
            offsetX += ddx;
            offsetY += ddy;
            offsetZ += ddz;
            ++matched;
        }
    }

    if (distances.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "No nearest-neighbor matches found";
        }
        return false;
    }

    std::sort(distances.begin(), distances.end());
    double sum = 0.0;
    for (double d : distances)
    {
        sum += d;
    }
    result.nnMean = sum / static_cast<double>(distances.size());
    result.nnMedian = distances[distances.size() / 2];
    result.nnP90 = distances[distances.size() * 90 / 100];
    result.nnP95 = distances[distances.size() * 95 / 100];
    result.nnMax = distances.back();
    result.meanOffset = {offsetX / matched, offsetY / matched, offsetZ / matched};
    evaluatePass(thresholds, result);
    return true;
}

std::string AspPointCloudMetrics::toTextReport(const AspPointCloudMetricsResult &result)
{
    std::ostringstream out;
    out << "ASP valid: " << result.aspValidPoints << "\n";
    out << "PlaScan valid: " << result.plascanValidPoints << "\n";
    out << "Coverage ratio: " << result.coverageRatio << "\n";
    out << "BBox containment: " << result.bboxContainmentRatio << "\n";
    out << "NN mean: " << result.nnMean << "\n";
    out << "NN median: " << result.nnMedian << "\n";
    out << "NN P90: " << result.nnP90 << "\n";
    out << "NN P95: " << result.nnP95 << "\n";
    out << "NN max: " << result.nnMax << "\n";
    out << "Mean offset: (" << result.meanOffset[0] << ", "
        << result.meanOffset[1] << ", " << result.meanOffset[2] << ")\n";
    out << "PASS: " << (result.passed ? "true" : "false") << "\n";
    if (!result.passed)
    {
        out << "Failure: " << result.failureReason << "\n";
    }
    return out.str();
}

std::string AspPointCloudMetrics::toJson(const AspPointCloudMetricsResult &result)
{
    std::ostringstream out;
    out << "{\n";
    out << "  \"asp_valid_points\": " << result.aspValidPoints << ",\n";
    out << "  \"plascan_valid_points\": " << result.plascanValidPoints << ",\n";
    out << "  \"coverage_ratio\": " << result.coverageRatio << ",\n";
    out << "  \"bbox_containment_ratio\": " << result.bboxContainmentRatio << ",\n";
    out << "  \"nn_mean\": " << result.nnMean << ",\n";
    out << "  \"nn_median\": " << result.nnMedian << ",\n";
    out << "  \"nn_p90\": " << result.nnP90 << ",\n";
    out << "  \"nn_p95\": " << result.nnP95 << ",\n";
    out << "  \"nn_max\": " << result.nnMax << ",\n";
    out << "  \"mean_offset\": [" << result.meanOffset[0] << ", "
        << result.meanOffset[1] << ", " << result.meanOffset[2] << "],\n";
    out << "  \"passed\": " << (result.passed ? "true" : "false") << ",\n";
    out << "  \"failure_reason\": \"" << result.failureReason << "\"\n";
    out << "}\n";
    return out.str();
}

} // namespace mvs
} // namespace xjw

#include "PointCloudDomGenerator.h"
#include "PointCloudDomInternal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct BodyReference
{
    double centerX = 0.0;
    double centerY = 0.0;
    double centerZ = 0.0;
    double radius = 0.0;
};

struct GridPlan
{
    DemGridData reference;
    OrthoGenerationOptions resolved;
    BodyReference body;
    double minEdgeX = 0.0;
    double minEdgeY = 0.0;
    double maxEdgeX = 0.0;
    double maxEdgeY = 0.0;
};

bool fail(QString *errorMsg, const QString &message)
{
    if (errorMsg)
    {
        *errorMsg = message;
    }
    return false;
}

bool finitePoint(const PlaPointCloud &cloud, std::size_t index)
{
    const auto point = cloud[index];
    return std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z());
}

BodyReference resolveBodyReference(const PlaPointCloud &cloud,
                                   const OrthoGenerationOptions &options)
{
    BodyReference body{options.bodyCenterX,
                       options.bodyCenterY,
                       options.bodyCenterZ,
                       options.referenceRadius};
    if (!options.bodyReferenceAuto)
    {
        return body;
    }
    body = {};

    qint64 count = 0;
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (!finitePoint(cloud, index))
        {
            continue;
        }
        const auto point = cloud[index];
        body.centerX += point.x();
        body.centerY += point.y();
        body.centerZ += point.z();
        ++count;
    }
    if (count <= 0)
    {
        return {};
    }
    body.centerX /= static_cast<double>(count);
    body.centerY /= static_cast<double>(count);
    body.centerZ /= static_cast<double>(count);
    body.radius = 0.0;
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        if (!finitePoint(cloud, index))
        {
            continue;
        }
        const auto point = cloud[index];
        const double dx = point.x() - body.centerX;
        const double dy = point.y() - body.centerY;
        const double dz = point.z() - body.centerZ;
        body.radius += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    body.radius /= static_cast<double>(count);
    return body;
}

QString planarWkt()
{
    return QStringLiteral(
        "LOCAL_CS[\"PlaScan_Point_Cloud_Planar\","
        "LOCAL_DATUM[\"PlaScan_Local_Datum\",32767],UNIT[\"metre\",1],"
        "AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH]]");
}

QString cylindricalWkt(double radius, double centralMeridian)
{
    return QString(
        "PROJCS[\"PlaScan_Asteroid_Simple_Cylindrical\","
        "GEOGCS[\"GCS_PlaScan_Asteroid\",DATUM[\"D_PlaScan_Asteroid\","
        "SPHEROID[\"PlaScan_Asteroid_Sphere\",%1,0]],PRIMEM[\"Reference_Meridian\",0],"
        "UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Equirectangular\"],"
        "PARAMETER[\"standard_parallel_1\",0],PARAMETER[\"central_meridian\",%2],"
        "PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1]]")
        .arg(radius, 0, 'g', 16)
        .arg(centralMeridian, 0, 'g', 16);
}

bool planGrid(const PlaPointCloud &cloud,
              const OrthoGenerationOptions &options,
              GridPlan *plan,
              QString *errorMsg)
{
    if (!plan || cloud.size() == 0)
    {
        return fail(errorMsg, QStringLiteral("输入点云为空"));
    }
    if (!cloud.hasColors())
    {
        return fail(errorMsg, QStringLiteral("点云不包含 RGB 颜色，无法从点颜色生成正射影像"));
    }

    GridPlan output;
    output.resolved = options;
    output.body = resolveBodyReference(cloud, options);
    if (options.projectionType == OrthoProjectionType::SimpleCylindrical)
    {
        if (!(output.body.radius > 0.0) || !std::isfinite(output.body.radius))
        {
            return fail(errorMsg, QStringLiteral("无法从点云估算有效的小天体参考半径"));
        }
        output.minEdgeX = -kPi * output.body.radius;
        output.maxEdgeX = kPi * output.body.radius;
        output.minEdgeY = -0.5 * kPi * output.body.radius;
        output.maxEdgeY = 0.5 * kPi * output.body.radius;
        output.reference.projection.coordinateSystem =
            QStringLiteral("PlaScan Asteroid Simple Cylindrical");
        output.reference.projection.projectionWkt =
            cylindricalWkt(output.body.radius, options.centralMeridian);
    }
    else
    {
        output.minEdgeX = std::numeric_limits<double>::max();
        output.minEdgeY = std::numeric_limits<double>::max();
        output.maxEdgeX = -std::numeric_limits<double>::max();
        output.maxEdgeY = -std::numeric_limits<double>::max();
        for (std::size_t index = 0; index < cloud.size(); ++index)
        {
            if (!finitePoint(cloud, index))
            {
                continue;
            }
            const auto point = cloud[index];
            output.minEdgeX = std::min(output.minEdgeX, static_cast<double>(point.x()));
            output.minEdgeY = std::min(output.minEdgeY, static_cast<double>(point.y()));
            output.maxEdgeX = std::max(output.maxEdgeX, static_cast<double>(point.x()));
            output.maxEdgeY = std::max(output.maxEdgeY, static_cast<double>(point.y()));
        }
        output.reference.projection.coordinateSystem = QStringLiteral("PlaScan Point Cloud Planar");
        output.reference.projection.projectionWkt = planarWkt();
    }
    if (options.bounds.enabled)
    {
        output.minEdgeX = options.bounds.minX;
        output.minEdgeY = options.bounds.minY;
        output.maxEdgeX = options.bounds.maxX;
        output.maxEdgeY = options.bounds.maxY;
    }

    const double spanX = output.maxEdgeX - output.minEdgeX;
    const double spanY = output.maxEdgeY - output.minEdgeY;
    if (!(spanX > 0.0) || !(spanY > 0.0))
    {
        return fail(errorMsg, QStringLiteral("点云投影范围无效"));
    }

    double stepX = options.pixelSizeX;
    double stepY = options.pixelSizeY;
    if (options.sizingMode == OrthoSizingMode::MaximumDimension)
    {
        const double step = std::max(spanX, spanY) / options.maximumDimension;
        stepX = step;
        stepY = step;
    }
    else if (!(stepX > 0.0) || !(stepY > 0.0))
    {
        const double targetPixels = std::max(1.0, static_cast<double>(cloud.size()) * 2.0);
        const double step = std::sqrt(spanX * spanY / targetPixels);
        stepX = step;
        stepY = step;
    }
    const double widthValue = std::max(1.0, std::ceil(spanX / stepX));
    const double heightValue = std::max(1.0, std::ceil(spanY / stepY));
    if (widthValue > std::numeric_limits<int>::max()
        || heightValue > std::numeric_limits<int>::max()
        || widthValue * heightValue > static_cast<double>(options.maximumPixelCount))
    {
        return fail(errorMsg,
                    QStringLiteral("正射输出尺寸超过安全像素预算 %1")
                        .arg(options.maximumPixelCount));
    }
    const int width = static_cast<int>(widthValue);
    const int height = static_cast<int>(heightValue);
    const qint64 pixelCount = static_cast<qint64>(width) * height;
    if (pixelCount > options.maximumPixelCount)
    {
        return fail(errorMsg,
                    QStringLiteral("正射输出包含 %1 个像素，超过安全上限 %2")
                        .arg(pixelCount).arg(options.maximumPixelCount));
    }
    output.reference.width = width;
    output.reference.height = height;
    output.reference.stepX = stepX;
    output.reference.stepY = stepY;
    output.reference.minX = output.minEdgeX + 0.5 * stepX;
    output.reference.minY = output.minEdgeY + 0.5 * stepY;
    output.resolved.pixelSizeX = stepX;
    output.resolved.pixelSizeY = stepY;
    output.resolved.bodyReferenceAuto = false;
    output.resolved.bodyCenterX = output.body.centerX;
    output.resolved.bodyCenterY = output.body.centerY;
    output.resolved.bodyCenterZ = output.body.centerZ;
    output.resolved.referenceRadius = output.body.radius;
    *plan = output;
    return true;
}

} // namespace

bool PointCloudDomGenerator::estimate(const PlaPointCloud &pointCloud,
                                      const OrthoGenerationOptions &options,
                                      QJsonObject *result,
                                      QString *errorMsg)
{
    GridPlan plan;
    if (!planGrid(pointCloud, options, &plan, errorMsg))
    {
        return false;
    }
    if (result)
    {
        QJsonObject output;
        output[QStringLiteral("resolved_settings")] = plan.resolved.toResolvedJson();
        output[QStringLiteral("coordinate_system")] = plan.reference.projection.coordinateSystem;
        output[QStringLiteral("projection_wkt")] = plan.reference.projection.projectionWkt;
        output[QStringLiteral("projection_wkt_present")] = true;
        output[QStringLiteral("dem_min_x")] = plan.minEdgeX;
        output[QStringLiteral("dem_min_y")] = plan.minEdgeY;
        output[QStringLiteral("dem_max_x")] = plan.maxEdgeX;
        output[QStringLiteral("dem_max_y")] = plan.maxEdgeY;
        output[QStringLiteral("dem_pixel_size_x")] = plan.reference.stepX;
        output[QStringLiteral("dem_pixel_size_y")] = plan.reference.stepY;
        output[QStringLiteral("min_x")] = plan.minEdgeX;
        output[QStringLiteral("min_y")] = plan.minEdgeY;
        output[QStringLiteral("max_x")] = plan.maxEdgeX;
        output[QStringLiteral("max_y")] = plan.maxEdgeY;
        output[QStringLiteral("pixel_size_x")] = plan.reference.stepX;
        output[QStringLiteral("pixel_size_y")] = plan.reference.stepY;
        output[QStringLiteral("width")] = plan.reference.width;
        output[QStringLiteral("height")] = plan.reference.height;
        output[QStringLiteral("estimated_memory_bytes")] =
            static_cast<double>(plan.reference.width) * plan.reference.height * 16.0;
        *result = output;
    }
    return true;
}

bool PointCloudDomGenerator::generate(const PlaPointCloud &pointCloud,
                                      const OrthoGenerationOptions &options,
                                      PointCloudDomResult *result,
                                      QString *errorMsg,
                                      const std::atomic_bool *cancelFlag,
                                      const ProgressCallback &progressCallback)
{
    if (!result)
    {
        return fail(errorMsg, QStringLiteral("点云正射输出对象为空"));
    }
    GridPlan plan;
    if (!planGrid(pointCloud, options, &plan, errorMsg))
    {
        return false;
    }
    cv::Mat image(plan.reference.height, plan.reference.width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask(plan.reference.height, plan.reference.width, CV_8U, cv::Scalar(0));
    cv::Mat depth(plan.reference.height, plan.reference.width, CV_64F,
                  cv::Scalar(-std::numeric_limits<double>::infinity()));
    qint64 projectedCount = 0;
    const double meridianRadians = options.centralMeridian * kPi / 180.0;
    for (std::size_t index = 0; index < pointCloud.size(); ++index)
    {
        if ((index & 0x3fffU) == 0U)
        {
            if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
            {
                return fail(errorMsg, QStringLiteral("点云正射影像生成已取消"));
            }
            if (progressCallback)
            {
                progressCallback(QStringLiteral("投影点云颜色"),
                                 static_cast<int>(80.0 * index / std::max<std::size_t>(1, pointCloud.size())));
            }
        }
        if (!finitePoint(pointCloud, index))
        {
            continue;
        }
        const auto point = pointCloud[index];
        double u = point.x();
        double v = point.y();
        double pointDepth = point.z();
        if (options.projectionType == OrthoProjectionType::SimpleCylindrical)
        {
            const double dx = point.x() - plan.body.centerX;
            const double dy = point.y() - plan.body.centerY;
            const double dz = point.z() - plan.body.centerZ;
            pointDepth = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (!(pointDepth > 0.0))
            {
                continue;
            }
            double longitude = std::atan2(dy, dx) - meridianRadians;
            longitude = std::remainder(longitude, 2.0 * kPi);
            const double latitude = std::atan2(dz, std::sqrt(dx * dx + dy * dy));
            u = plan.body.radius * longitude;
            v = plan.body.radius * latitude;
        }
        int col = static_cast<int>(std::floor((u - plan.minEdgeX) / plan.reference.stepX));
        int row = static_cast<int>(std::floor((v - plan.minEdgeY) / plan.reference.stepY));
        if (col == plan.reference.width)
        {
            col = plan.reference.width - 1;
        }
        if (row == plan.reference.height)
        {
            row = plan.reference.height - 1;
        }
        if (row < 0 || row >= plan.reference.height || col < 0 || col >= plan.reference.width)
        {
            continue;
        }
        double &storedDepth = depth.at<double>(row, col);
        if (pointDepth >= storedDepth)
        {
            storedDepth = pointDepth;
            image.at<cv::Vec3b>(row, col) = cv::Vec3b(
                cv::saturate_cast<uchar>(pointCloud.colors()->getValue(index, 2)),
                cv::saturate_cast<uchar>(pointCloud.colors()->getValue(index, 1)),
                cv::saturate_cast<uchar>(pointCloud.colors()->getValue(index, 0)));
            mask.at<uchar>(row, col) = 255;
        }
        ++projectedCount;
    }
    if (options.fillHoles)
    {
        point_cloud_dom_internal::fillSmallGaps(
            &image,
            &mask,
            std::clamp(static_cast<int>(std::ceil(options.holeFillRadius)), 1, 8));
    }
    plan.reference.validMask = mask;
    result->imageBgr = image;
    result->validMask = mask;
    result->reference = plan.reference;
    result->resolvedOptions = plan.resolved;
    result->inputPointCount = static_cast<qint64>(pointCloud.size());
    result->projectedPointCount = projectedCount;
    result->validPixelCount = cv::countNonZero(mask);
    result->coverageRatio = static_cast<double>(result->validPixelCount)
        / static_cast<double>(plan.reference.width * plan.reference.height);
    if (progressCallback)
    {
        progressCallback(QStringLiteral("点云投影完成"), 90);
    }
    return true;
}

} // namespace xjw

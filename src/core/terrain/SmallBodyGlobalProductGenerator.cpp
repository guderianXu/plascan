#include "SmallBodyGlobalProductGenerator.h"

#include "DemDomIO.h"
#include "GlobalTerrainReportRenderer.h"
#include "ObjMtlLoader.h"
#include "SmallBodyMeshRaycaster.h"
#include "io/PathIO.h"

#include <plapoint/io/ply_io.h>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QStringList>
#include <QTemporaryFile>
#include <QUuid>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <new>
#include <numeric>
#include <vector>

namespace xjw
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

void reportProgress(const SmallBodyProgressCallback &callback,
                    const QString &stage,
                    int percent)
{
    if (callback)
    {
        callback(stage, std::clamp(percent, 0, 100));
    }
}

bool cancellationRequested(const std::atomic_bool *cancelFlag)
{
    return cancelFlag && cancelFlag->load(std::memory_order_relaxed);
}

bool failIfCancelled(const std::atomic_bool *cancelFlag, QString *errorMessage)
{
    if (!cancellationRequested(cancelFlag))
    {
        return false;
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("全球地形产品生成已取消。");
    }
    return true;
}

class ArtifactTransaction final
{
public:
    ~ArtifactTransaction()
    {
        if (_committed)
        {
            return;
        }
        for (const Artifact &artifact : _artifacts)
        {
            QFile::remove(artifact.temporaryPath);
        }
        for (const QString &published_path : _publishedPaths)
        {
            QFile::remove(published_path);
        }
    }

    bool add(const QString &finalPath, QString *temporaryPath, QString *errorMessage)
    {
        if (QFileInfo::exists(finalPath))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "全球地形产物已存在，为避免混合覆盖旧成果，请选择新的输出目录：%1")
                                    .arg(finalPath);
            }
            return false;
        }
        const QFileInfo final_info(finalPath);
        const QString token = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QString temporary_name = QStringLiteral("%1.%2.tmp")
                                     .arg(final_info.completeBaseName(), token);
        if (!final_info.suffix().isEmpty())
        {
            temporary_name += QStringLiteral(".%1").arg(final_info.suffix());
        }
        const QString temporary_path =
            QDir(final_info.absolutePath()).filePath(temporary_name);
        _artifacts.push_back({temporary_path, finalPath});
        if (temporaryPath)
        {
            *temporaryPath = temporary_path;
        }
        return true;
    }

    bool publish(QString *errorMessage)
    {
        for (const Artifact &artifact : _artifacts)
        {
            if (!QFileInfo::exists(artifact.temporaryPath)
                || !QFile::rename(artifact.temporaryPath, artifact.finalPath))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("无法原子发布全球地形产物：%1 -> %2")
                                        .arg(artifact.temporaryPath, artifact.finalPath);
                }
                return false;
            }
            _publishedPaths.append(artifact.finalPath);
        }
        _committed = true;
        return true;
    }

private:
    struct Artifact
    {
        QString temporaryPath;
        QString finalPath;
    };

    std::vector<Artifact> _artifacts;
    QStringList _publishedPaths;
    bool _committed = false;
};

bool preflightOutputDirectory(const QString &outputDirectory,
                              bool includePreview,
                              QString *errorMessage)
{
    const QString product_directory =
        QDir(outputDirectory).filePath(QStringLiteral("products"));
    if (!QDir().mkpath(outputDirectory) || !QDir().mkpath(product_directory))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建全球地形输出目录：%1")
                                .arg(outputDirectory);
        }
        return false;
    }

    QStringList final_paths{
        QDir(product_directory).filePath(QStringLiteral("radial_dem.tif")),
        QDir(product_directory).filePath(QStringLiteral("elevation_dem.tif")),
        QDir(product_directory).filePath(QStringLiteral("dom.tif")),
        QDir(product_directory).filePath(QStringLiteral("reliability.tif")),
        QDir(product_directory).filePath(QStringLiteral("coverage_mask.tif")),
        QDir(product_directory).filePath(QStringLiteral("ambiguity_mask.tif")),
        QDir(outputDirectory).filePath(QStringLiteral("small_body_global_report.json"))};
    if (includePreview)
    {
        final_paths.append(
            QDir(outputDirectory).filePath(QStringLiteral("global_terrain_report.png")));
    }
    for (const QString &final_path : final_paths)
    {
        if (QFileInfo::exists(final_path))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "全球地形产物已存在，为避免混合覆盖旧成果，请选择新的输出目录：%1")
                                    .arg(final_path);
            }
            return false;
        }
    }

    for (const QString &directory : {outputDirectory, product_directory})
    {
        QTemporaryFile probe(
            QDir(directory).filePath(QStringLiteral(".plascan-write-check-XXXXXX")));
        if (!probe.open())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("全球地形输出目录不可写：%1（%2）")
                                    .arg(directory, probe.errorString());
            }
            return false;
        }
    }
    return true;
}

bool loadSurface(const QString &path, TerrainMeshInput *surface, QString *errorMessage)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix == QLatin1String("obj"))
    {
        return ObjMtlLoader::load(path, surface, errorMessage);
    }
    if (suffix != QLatin1String("ply"))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("全球小天体产品目前支持带三角面的 PLY 或 OBJ 模型：%1")
                                .arg(path);
        }
        return false;
    }

    try
    {
        const auto cloud = plapoint::io::readPly<float>(
            xjw::common::io::toNativeNarrowPath(path));
        if (!cloud || cloud->size() == 0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("PLY 模型为空：%1").arg(path);
            }
            return false;
        }
        surface->mesh = std::move(*cloud);
        return true;
    }
    catch (const std::exception &exception)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取 PLY 模型失败：%1 (%2)")
                                .arg(path, QString::fromUtf8(exception.what()));
        }
        return false;
    }
}

cv::Vec3d estimateCenter(const PlaPointCloud &mesh,
                         const std::atomic_bool *cancelFlag)
{
    cv::Vec3d center(0.0, 0.0, 0.0);
    const auto &points = mesh.points();
    for (plamatrix::Index row = 0; row < points.rows(); ++row)
    {
        if ((row & 4095) == 0 && cancellationRequested(cancelFlag))
        {
            return {};
        }
        center[0] += points.getValue(row, 0);
        center[1] += points.getValue(row, 1);
        center[2] += points.getValue(row, 2);
    }
    return center / static_cast<double>(std::max<plamatrix::Index>(1, points.rows()));
}

double estimateReferenceRadius(const PlaPointCloud &mesh,
                               const cv::Vec3d &center,
                               const std::atomic_bool *cancelFlag)
{
    std::vector<double> radii;
    radii.reserve(mesh.size());
    const auto &points = mesh.points();
    for (plamatrix::Index row = 0; row < points.rows(); ++row)
    {
        if ((row & 4095) == 0 && cancellationRequested(cancelFlag))
        {
            return 0.0;
        }
        const cv::Vec3d position(points.getValue(row, 0),
                                 points.getValue(row, 1),
                                 points.getValue(row, 2));
        const double radius = cv::norm(position - center);
        if (std::isfinite(radius) && radius > 0.0)
        {
            radii.push_back(radius);
        }
    }
    if (radii.empty())
    {
        return 0.0;
    }
    const auto middle = radii.begin() + static_cast<std::ptrdiff_t>(radii.size() / 2);
    std::nth_element(radii.begin(), middle, radii.end());
    return *middle;
}

QString detectDomColorSource(const TerrainMeshInput &surface)
{
    const bool has_texture = !surface.texture.empty()
        && surface.mesh.hasTextureCoords()
        && (surface.mesh.hasFaceTextureIndices()
            || surface.mesh.hasPointAlignedTextureCoords());
    if (has_texture)
    {
        return QStringLiteral("texture");
    }
    if (surface.mesh.hasColors())
    {
        return QStringLiteral("vertex_color");
    }
    return {};
}

QString safeWktName(QString name)
{
    name = name.trimmed();
    if (name.isEmpty())
    {
        name = QStringLiteral("Small_Body");
    }
    name.replace(QLatin1Char('"'), QLatin1Char('_'));
    return name;
}

DemProjectionParameters makeProjection(const SmallBodyGlobalOptions &options,
                                       const cv::Vec3d &center,
                                       double referenceRadius)
{
    DemProjectionParameters projection;
    const QString target = safeWktName(options.targetName);
    projection.coordinateSystem = QStringLiteral("%1 / %2").arg(target, options.bodyFixedFrame);
    projection.projectionWkt = QStringLiteral(
        "GEOGCS[\"%1 Body Fixed\",DATUM[\"D_%1\",SPHEROID[\"%1\",%2,0]],"
        "PRIMEM[\"Reference Meridian\",%3],UNIT[\"degree\",0.0174532925199433],"
        "AXIS[\"Longitude\",EAST],AXIS[\"Latitude\",NORTH]]")
        .arg(target)
        .arg(referenceRadius, 0, 'g', 15)
        .arg(options.centralMeridianDeg, 0, 'g', 15);
    projection.originX = 0.0;
    projection.originY = -90.0;
    projection.metadata.insert(QStringLiteral("TARGET_NAME"), options.targetName);
    projection.metadata.insert(QStringLiteral("BODY_FIXED_FRAME"), options.bodyFixedFrame);
    projection.metadata.insert(QStringLiteral("SOURCE_SURFACE_UNIT"),
                               options.surfaceCoordinateUnit.trimmed().toLower());
    projection.metadata.insert(QStringLiteral("LATITUDE_TYPE"), QStringLiteral("planetocentric"));
    projection.metadata.insert(QStringLiteral("LONGITUDE_DIRECTION"), QStringLiteral("positive_east"));
    projection.metadata.insert(QStringLiteral("LONGITUDE_DOMAIN"), QStringLiteral("0_360"));
    projection.metadata.insert(QStringLiteral("REFERENCE_RADIUS_M"),
                               QString::number(referenceRadius, 'g', 15));
    projection.metadata.insert(QStringLiteral("CENTER_XYZ_M"),
                               QStringLiteral("%1,%2,%3")
                                   .arg(center[0], 0, 'g', 15)
                                   .arg(center[1], 0, 'g', 15)
                                   .arg(center[2], 0, 'g', 15));
    projection.metadata.insert(QStringLiteral("VERTICAL_REFERENCE"),
                               QStringLiteral("radial_distance_from_body_center"));
    projection.metadata.insert(QStringLiteral("ALGORITHM"),
                               QStringLiteral("body_center_ray_triangle_bvh_v1"));
    projection.metadata.insert(QStringLiteral("INTERSECTION_SELECTION"),
                               QStringLiteral("nearest_positive"));
    return projection;
}

bool writeScalarProduct(const DemGridData &referenceGrid,
                        const cv::Mat &values,
                        double scale,
                        const QString &path,
                        const QString &productType,
                        const QString &valueSemantics,
                        bool includeUncoveredPixels,
                        QString *errorMessage)
{
    DemGridData product = referenceGrid;
    product.elevation.release();
    values.convertTo(product.elevation, CV_32F, scale);
    product.worldX.release();
    product.worldY.release();
    product.color.release();
    product.triangulationError.release();
    product.pointCount.release();
    product.confidence.release();
    product.coverageMask.release();
    if (includeUncoveredPixels)
    {
        product.validMask = cv::Mat(
            referenceGrid.height, referenceGrid.width, CV_8U, cv::Scalar(255));
    }
    product.projection.metadata[QStringLiteral("VERTICAL_REFERENCE")] =
        QStringLiteral("not_applicable");
    product.projection.metadata[QStringLiteral("PRODUCT_TYPE")] = productType;
    product.projection.metadata[QStringLiteral("VALUE_SEMANTICS")] = valueSemantics;
    product.projection.metadata[QStringLiteral("BAND_UNIT")] = QStringLiteral("1");
    return DemDomIO::writeDemRaster(
        product, path, DemRasterFormat::Float32Tiff, errorMessage);
}

QJsonObject makeReport(const SmallBodyGlobalProducts &products,
                       const SmallBodyGlobalOptions &options,
                       double minimumElevation,
                       double maximumElevation)
{
    QJsonObject frame;
    frame[QStringLiteral("target_name")] = options.targetName;
    frame[QStringLiteral("body_fixed_frame")] = options.bodyFixedFrame;
    const bool model_local_frame = options.bodyFixedFrame.trimmed().compare(
        QLatin1String("MODEL_LOCAL_BODY_FIXED"), Qt::CaseInsensitive) == 0;
    frame[QStringLiteral("frame_status")] = model_local_frame
        ? QStringLiteral("model_local_unverified")
        : QStringLiteral("user_declared_unverified");
    frame[QStringLiteral("latitude_type")] = QStringLiteral("planetocentric");
    frame[QStringLiteral("longitude_direction")] = QStringLiteral("positive_east");
    frame[QStringLiteral("longitude_domain")] = QStringLiteral("0_360");
    frame[QStringLiteral("central_meridian_deg")] = options.centralMeridianDeg;
    frame[QStringLiteral("reference_radius_m")] = products.referenceRadiusM;
    frame[QStringLiteral("center_xyz_m")] = QJsonArray{
        products.bodyCenter[0], products.bodyCenter[1], products.bodyCenter[2]};
    frame[QStringLiteral("source_surface_unit")] =
        options.surfaceCoordinateUnit.trimmed().toLower();
    frame[QStringLiteral("surface_scale_to_m")] =
        options.surfaceCoordinateUnit.trimmed().compare(QLatin1String("km"), Qt::CaseInsensitive) == 0
            ? 1000.0 : 1.0;

    QJsonObject grid;
    grid[QStringLiteral("width")] = products.radialDem.width;
    grid[QStringLiteral("height")] = products.radialDem.height;
    grid[QStringLiteral("angular_resolution_deg")] = products.radialDem.stepX;
    grid[QStringLiteral("north_up")] = true;

    QJsonObject artifacts;
    artifacts[QStringLiteral("radial_dem")] = products.radialDemPath;
    artifacts[QStringLiteral("elevation_dem")] = products.elevationDemPath;
    artifacts[QStringLiteral("dom")] = products.domPath;
    artifacts[QStringLiteral("reliability")] = products.reliabilityPath;
    artifacts[QStringLiteral("coverage")] = products.coveragePath;
    artifacts[QStringLiteral("ambiguity")] = products.ambiguityPath;
    artifacts[QStringLiteral("preview_png")] = products.previewPath;

    QJsonObject metrics;
    metrics[QStringLiteral("coverage_ratio")] = products.coverageRatio;
    metrics[QStringLiteral("solid_angle_weighted_coverage_ratio")] =
        products.solidAngleWeightedCoverageRatio;
    metrics[QStringLiteral("ambiguous_ratio")] = products.ambiguousRatio;
    metrics[QStringLiteral("minimum_elevation_m")] = minimumElevation;
    metrics[QStringLiteral("maximum_elevation_m")] = maximumElevation;

    QJsonArray warnings;
    if (model_local_frame)
    {
        warnings.append(QStringLiteral(
            "坐标轴仅声明为 PlaScan 局部体固系；未应用 SPICE/IAU 姿态，不能冒充官方天体经纬网。"));
    }
    else
    {
        warnings.append(QStringLiteral(
            "体固连坐标系名称由用户声明，PlaScan 未在本流程中执行 SPICE/IAU 姿态转换或验证。"));
    }
    if (options.automaticCenter)
    {
        warnings.append(QStringLiteral("体心由网格顶点均值自动估计；不完整或密度不均网格建议手动提供体心。"));
    }
    warnings.append(QStringLiteral("PLY/OBJ 本身不声明长度单位；输入顶点已按用户选择的 %1 解释。")
                        .arg(options.surfaceCoordinateUnit.trimmed().toLower()));
    if (products.solidAngleWeightedCoverageRatio < 0.999)
    {
        warnings.append(QStringLiteral(
            "全球径向覆盖不完整；请检查网格是否闭合、体心是否位于模型内部，以及坐标单位是否正确。"));
    }
    if (products.ambiguousRatio > 0.0)
    {
        warnings.append(QStringLiteral(
            "部分径向射线命中多个表面；径向 DEM 固定采用从体心出发的最近正交点，"
            "请结合 ambiguity_mask.tif 评估深凹或双叶区域。"));
    }
    warnings.append(QStringLiteral(
        "可靠性是面法向与径向夹角的几何代理，不等同于多视影像支持度或测量精度。"));
    warnings.append(QStringLiteral(
        "覆盖率按球面固体角 cos(latitude) 加权，不代表不规则天体三角网的真实表面积权重。"));
    warnings.append(QStringLiteral(
        "未提供外部参考 DEM；四联图右下角为 hillshade，不是官方表面误差。"));

    QJsonObject report;
    report[QStringLiteral("type")] = QStringLiteral("small_body_global_terrain");
    report[QStringLiteral("schema_version")] = 1;
    report[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    report[QStringLiteral("generator")] = QStringLiteral("PlaScan C++ native terrain core");
    report[QStringLiteral("algorithm")] = QStringLiteral("body_center_ray_triangle_bvh_v1");
    report[QStringLiteral("intersection_selection")] = QStringLiteral("nearest_positive");
    report[QStringLiteral("source_surface")] = products.sourceSurfacePath;
    report[QStringLiteral("dom_color_source")] = products.domColorSource;
    report[QStringLiteral("frame")] = frame;
    report[QStringLiteral("grid")] = grid;
    report[QStringLiteral("artifacts")] = artifacts;
    report[QStringLiteral("metrics")] = metrics;
    report[QStringLiteral("difference_available")] = false;
    report[QStringLiteral("warnings")] = warnings;
    return report;
}

bool writeJson(const QString &path, const QJsonObject &object, QString *errorMessage)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写出全球地形报告：%1").arg(path);
        }
        return false;
    }
    const QByteArray contents = QJsonDocument(object).toJson(QJsonDocument::Indented);
    if (file.write(contents) != contents.size() || !file.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("提交全球地形报告失败：%1").arg(path);
        }
        return false;
    }
    return true;
}

} // namespace

bool SmallBodyGlobalProductGenerator::generate(const QString &surfacePath,
                                                const QString &outputDirectory,
                                                const SmallBodyGlobalOptions &options,
                                                SmallBodyGlobalProducts *products,
                                                QString *errorMessage,
                                                const std::atomic_bool *cancelFlag,
                                                const SmallBodyProgressCallback &progressCallback)
{
    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    const QString absolute_surface_path =
        QDir::cleanPath(QFileInfo(surfacePath).absoluteFilePath());
    if (!QFileInfo::exists(absolute_surface_path))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("表面模型不存在：%1").arg(absolute_surface_path);
        }
        return false;
    }
    reportProgress(progressCallback, QStringLiteral("读取体固连表面模型"), 2);
    TerrainMeshInput surface;
    if (!loadSurface(absolute_surface_path, &surface, errorMessage))
    {
        return false;
    }
    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    return generateFromMesh(surface, absolute_surface_path, outputDirectory, options, products,
                            errorMessage, cancelFlag, progressCallback);
}

bool SmallBodyGlobalProductGenerator::generateFromMesh(
    const TerrainMeshInput &surface,
    const QString &sourceLabel,
    const QString &outputDirectory,
    const SmallBodyGlobalOptions &options,
    SmallBodyGlobalProducts *products,
    QString *errorMessage,
    const std::atomic_bool *cancelFlag,
    const SmallBodyProgressCallback &progressCallback) try
{
    if (!products)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("全球地形产品输出对象为空。");
        }
        return false;
    }
    *products = SmallBodyGlobalProducts();
    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    if (!options.validate(errorMessage))
    {
        return false;
    }
    if (outputDirectory.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("全球地形产品输出目录不能为空。");
        }
        return false;
    }
    const QString absolute_output_directory =
        QDir::cleanPath(QFileInfo(outputDirectory).absoluteFilePath());
    if (!surface.mesh.hasFaces() || !surface.mesh.faces()
        || surface.mesh.faces()->rows() == 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "全球径向 DEM/DOM 需要带三角面的闭合 PLY/OBJ 网格；纯点云可继续使用现有全球 DOM 投影。");
        }
        return false;
    }
    const QString dom_color_source = detectDomColorSource(surface);
    if (dom_color_source.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "生成真实 DOM 需要网格包含 RGB 顶点颜色，或 OBJ 包含可读取的纹理与 UV；"
                "PlaScan 不会用固定灰色占位图冒充 DOM。");
        }
        return false;
    }
    if (!preflightOutputDirectory(
            absolute_output_directory, options.writeReportPreview, errorMessage))
    {
        return false;
    }

    const double surface_scale_to_m =
        options.surfaceCoordinateUnit.trimmed().compare(
            QLatin1String("km"), Qt::CaseInsensitive) == 0 ? 1000.0 : 1.0;
    const cv::Vec3d center_in_surface_units = options.automaticCenter
        ? estimateCenter(surface.mesh, cancelFlag) : options.bodyCenter / surface_scale_to_m;
    const cv::Vec3d center_m = center_in_surface_units * surface_scale_to_m;
    const double reference_radius = options.referenceRadiusM > 0.0
        ? options.referenceRadiusM
        : estimateReferenceRadius(surface.mesh, center_in_surface_units, cancelFlag)
            * surface_scale_to_m;
    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    if (!std::isfinite(reference_radius) || reference_radius <= 0.0
        || reference_radius > static_cast<double>(std::numeric_limits<float>::max()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法从模型估计可由 Float32 GeoTIFF 表示的有效参考半径。");
        }
        return false;
    }

    reportProgress(progressCallback, QStringLiteral("建立三角网 BVH"), 5);
    SmallBodyMeshRaycaster raycaster;
    if (!raycaster.initialize(
            surface, center_in_surface_units, errorMessage, cancelFlag))
    {
        return false;
    }
    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }

    const int width = static_cast<int>(std::ceil(360.0 / options.angularResolutionDeg));
    const int height = static_cast<int>(std::ceil(180.0 / options.angularResolutionDeg));
    const double step_x = 360.0 / static_cast<double>(width);
    const double step_y = 180.0 / static_cast<double>(height);
    const DemProjectionParameters projection = makeProjection(options, center_m, reference_radius);

    SmallBodyGlobalProducts generated;
    generated.sourceSurfacePath = sourceLabel;
    generated.domColorSource = dom_color_source;
    generated.bodyCenter = center_m;
    generated.referenceRadiusM = reference_radius;
    generated.radialDem.width = width;
    generated.radialDem.height = height;
    generated.radialDem.minX = step_x * 0.5;
    generated.radialDem.minY = -90.0 + step_y * 0.5;
    generated.radialDem.stepX = step_x;
    generated.radialDem.stepY = step_y;
    generated.radialDem.projection = projection;
    generated.radialDem.projection.metadata[QStringLiteral("DOM_COLOR_SOURCE")] =
        dom_color_source;
    generated.radialDem.projection.metadata[QStringLiteral("PRODUCT_TYPE")] =
        QStringLiteral("small_body_global_radial_dem");
    generated.radialDem.projection.metadata[QStringLiteral("BAND_UNIT")] =
        QStringLiteral("m");
    generated.radialDem.elevation = cv::Mat(height, width, CV_32F, cv::Scalar(0.0f));
    generated.radialDem.validMask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
    generated.radialDem.confidence = cv::Mat(height, width, CV_32F, cv::Scalar(0.0f));
    generated.radialDem.coverageMask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
    generated.domBgr = cv::Mat(height, width, CV_8UC3, cv::Scalar(48, 48, 48));
    generated.validMask = generated.radialDem.validMask;
    generated.reliability = generated.radialDem.confidence;
    generated.ambiguousMask = cv::Mat(height, width, CV_8U, cv::Scalar(0));

    std::atomic_int completed_rows{0};
    std::atomic_bool cancelled{false};
    std::atomic_bool invalid_radial_value{false};
    int reported_progress = 5;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int row = 0; row < height; ++row)
    {
        if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
        {
            cancelled.store(true, std::memory_order_relaxed);
            continue;
        }
        const double latitude_deg = -90.0 + (static_cast<double>(row) + 0.5) * step_y;
        const double latitude = latitude_deg * kPi / 180.0;
        const double cos_latitude = std::cos(latitude);
        const double sin_latitude = std::sin(latitude);
        for (int col = 0; col < width; ++col)
        {
            if ((col & 255) == 0 && cancellationRequested(cancelFlag))
            {
                cancelled.store(true, std::memory_order_relaxed);
                break;
            }
            const double longitude_deg = (static_cast<double>(col) + 0.5) * step_x
                + options.centralMeridianDeg;
            const double longitude = longitude_deg * kPi / 180.0;
            const cv::Vec3d direction(cos_latitude * std::cos(longitude),
                                      cos_latitude * std::sin(longitude),
                                      sin_latitude);
            SmallBodyMeshRaycaster::Hit hit;
            if (!raycaster.intersect(direction, &hit))
            {
                continue;
            }
            const double radial_distance_m = hit.radius * surface_scale_to_m;
            if (!std::isfinite(radial_distance_m) || radial_distance_m <= 0.0
                || radial_distance_m
                    > static_cast<double>(std::numeric_limits<float>::max()))
            {
                invalid_radial_value.store(true, std::memory_order_relaxed);
                continue;
            }
            generated.radialDem.elevation.at<float>(row, col) =
                static_cast<float>(radial_distance_m);
            generated.radialDem.validMask.at<uchar>(row, col) = 255;
            generated.radialDem.coverageMask.at<uchar>(row, col) = 255;
            generated.radialDem.confidence.at<float>(row, col) = static_cast<float>(hit.reliability);
            generated.domBgr.at<cv::Vec3b>(row, col) = hit.colorBgr;
            generated.ambiguousMask.at<uchar>(row, col) = hit.ambiguous ? 255 : 0;
        }
        const int rows_done = completed_rows.fetch_add(1, std::memory_order_relaxed) + 1;
        if (progressCallback && (rows_done == height || rows_done % std::max(1, height / 50) == 0))
        {
#if defined(_OPENMP)
#pragma omp critical(SmallBodyGlobalProgress)
#endif
            {
                const int percent = 5 + rows_done * 75 / height;
                if (percent > reported_progress)
                {
                    reported_progress = percent;
                    reportProgress(progressCallback,
                                   QStringLiteral("生成全球径向 DEM/DOM"),
                                   percent);
                }
            }
        }
    }
    if (cancelled.load(std::memory_order_relaxed)
        || cancellationRequested(cancelFlag))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("全球地形产品生成已取消。");
        }
        return false;
    }
    if (invalid_radial_value.load(std::memory_order_relaxed))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "网格径向距离超出 Float32 GeoTIFF 可表示范围；请检查输入坐标单位和体心。");
        }
        return false;
    }

    generated.validMask = generated.radialDem.validMask;
    generated.reliability = generated.radialDem.confidence;
    generated.elevationDem = generated.radialDem;
    generated.elevationDem.elevation = generated.radialDem.elevation.clone();
    for (int row = 0; row < height; ++row)
    {
        if (cancellationRequested(cancelFlag))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("全球地形产品生成已取消。");
            }
            return false;
        }
        for (int col = 0; col < width; ++col)
        {
            if (generated.validMask.at<uchar>(row, col) != 0)
            {
                const double relative_elevation = static_cast<double>(
                    generated.radialDem.elevation.at<float>(row, col)) - reference_radius;
                if (!std::isfinite(relative_elevation)
                    || std::abs(relative_elevation)
                        > static_cast<double>(std::numeric_limits<float>::max()))
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral(
                            "相对高程超出 Float32 GeoTIFF 可表示范围；请检查参考半径。");
                    }
                    return false;
                }
                generated.elevationDem.elevation.at<float>(row, col) =
                    static_cast<float>(relative_elevation);
            }
        }
    }
    generated.elevationDem.projection.metadata[QStringLiteral("VERTICAL_REFERENCE")] =
        QStringLiteral("elevation_above_reference_radius");
    generated.elevationDem.projection.metadata[QStringLiteral("PRODUCT_TYPE")] =
        QStringLiteral("small_body_global_elevation_dem");
    generated.elevationDem.projection.metadata[QStringLiteral("BAND_UNIT")] =
        QStringLiteral("m");

    const int valid_count = cv::countNonZero(generated.validMask);
    if (valid_count == 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "体心发出的全部径向射线均未命中网格；请检查体心、体固连坐标和网格坐标单位。");
        }
        return false;
    }
    const int ambiguous_count = cv::countNonZero(generated.ambiguousMask);
    generated.coverageRatio = static_cast<double>(valid_count)
        / (static_cast<double>(width) * static_cast<double>(height));
    generated.ambiguousRatio = valid_count > 0
        ? static_cast<double>(ambiguous_count) / valid_count : 0.0;
    double valid_area = 0.0;
    double total_area = 0.0;
    for (int row = 0; row < height; ++row)
    {
        if (cancellationRequested(cancelFlag))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("全球地形产品生成已取消。");
            }
            return false;
        }
        const double latitude = (-90.0 + (static_cast<double>(row) + 0.5) * step_y)
            * kPi / 180.0;
        const double row_area = std::max(0.0, std::cos(latitude));
        total_area += row_area * width;
        valid_area += row_area * cv::countNonZero(generated.validMask.row(row));
    }
    generated.solidAngleWeightedCoverageRatio =
        total_area > 0.0 ? valid_area / total_area : 0.0;

    reportProgress(progressCallback, QStringLiteral("写出全球 GeoTIFF"), 82);
    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    const QString product_dir = QDir(absolute_output_directory).filePath(
        QStringLiteral("products"));
    if (!QDir().mkpath(product_dir))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建全球地形输出目录：%1").arg(product_dir);
        }
        return false;
    }
    generated.radialDemPath = QDir(product_dir).filePath(QStringLiteral("radial_dem.tif"));
    generated.elevationDemPath = QDir(product_dir).filePath(QStringLiteral("elevation_dem.tif"));
    generated.domPath = QDir(product_dir).filePath(QStringLiteral("dom.tif"));
    generated.reliabilityPath = QDir(product_dir).filePath(QStringLiteral("reliability.tif"));
    generated.coveragePath = QDir(product_dir).filePath(QStringLiteral("coverage_mask.tif"));
    generated.ambiguityPath = QDir(product_dir).filePath(
        QStringLiteral("ambiguity_mask.tif"));
    generated.previewPath = options.writeReportPreview
        ? QDir(absolute_output_directory).filePath(
              QStringLiteral("global_terrain_report.png"))
        : QString();
    generated.reportPath = QDir(absolute_output_directory).filePath(
        QStringLiteral("small_body_global_report.json"));

    ArtifactTransaction transaction;
    QString radial_dem_temporary;
    QString elevation_dem_temporary;
    QString dom_temporary;
    QString reliability_temporary;
    QString coverage_temporary;
    QString ambiguity_temporary;
    QString preview_temporary;
    QString report_temporary;
    if (!transaction.add(generated.radialDemPath, &radial_dem_temporary, errorMessage)
        || !transaction.add(
            generated.elevationDemPath, &elevation_dem_temporary, errorMessage)
        || !transaction.add(generated.domPath, &dom_temporary, errorMessage)
        || !transaction.add(
            generated.reliabilityPath, &reliability_temporary, errorMessage)
        || !transaction.add(generated.coveragePath, &coverage_temporary, errorMessage)
        || !transaction.add(generated.ambiguityPath, &ambiguity_temporary, errorMessage)
        || (options.writeReportPreview
            && !transaction.add(generated.previewPath, &preview_temporary, errorMessage))
        || !transaction.add(generated.reportPath, &report_temporary, errorMessage))
    {
        return false;
    }

    if (!DemDomIO::writeDemRaster(generated.radialDem, radial_dem_temporary,
                                  DemRasterFormat::Float32Tiff, errorMessage)
        || failIfCancelled(cancelFlag, errorMessage)
        || !DemDomIO::writeDemRaster(generated.elevationDem, elevation_dem_temporary,
                                     DemRasterFormat::Float32Tiff, errorMessage)
        || failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }

    DemGridData dom_grid = generated.radialDem;
    dom_grid.projection.metadata[QStringLiteral("VERTICAL_REFERENCE")] =
        QStringLiteral("not_applicable");
    dom_grid.projection.metadata[QStringLiteral("PRODUCT_TYPE")] =
        QStringLiteral("surface_colour_dom");
    dom_grid.projection.metadata[QStringLiteral("VALUE_SEMANTICS")] =
        QStringLiteral("surface_colour_rgb_with_coverage_alpha");
    dom_grid.projection.metadata.remove(QStringLiteral("BAND_UNIT"));
    if (!DemDomIO::writeDomGeoTiff(
            generated.domBgr, generated.validMask, dom_grid, dom_temporary, errorMessage)
        || failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    if (!writeScalarProduct(
            generated.radialDem, generated.reliability, 1.0,
            reliability_temporary, QStringLiteral("radial_geometry_reliability_proxy"),
            QStringLiteral("0=radial_grazing,1=surface_normal_parallel_to_radial"),
            false, errorMessage)
        || failIfCancelled(cancelFlag, errorMessage)
        || !writeScalarProduct(
            generated.radialDem, generated.radialDem.coverageMask, 1.0 / 255.0,
            coverage_temporary, QStringLiteral("surface_coverage_mask"),
            QStringLiteral("0=no_radial_surface,1=valid_radial_surface"), true, errorMessage)
        || failIfCancelled(cancelFlag, errorMessage)
        || !writeScalarProduct(
            generated.radialDem, generated.ambiguousMask, 1.0 / 255.0,
            ambiguity_temporary, QStringLiteral("radial_multi_surface_ambiguity_mask"),
            QStringLiteral("0=single_surface,1=multiple_radial_surfaces"), false,
            errorMessage)
        || failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }

    double minimum_elevation = 0.0;
    double maximum_elevation = 0.0;
    if (valid_count > 0)
    {
        cv::minMaxLoc(generated.elevationDem.elevation, &minimum_elevation, &maximum_elevation,
                      nullptr, nullptr, generated.validMask);
    }

    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    if (options.writeReportPreview
        && !GlobalTerrainReportRenderer::writePreview(
            generated, options, preview_temporary, errorMessage))
    {
        return false;
    }
    generated.report = makeReport(generated, options, minimum_elevation, maximum_elevation);
    if (failIfCancelled(cancelFlag, errorMessage))
    {
        return false;
    }
    if (!writeJson(report_temporary, generated.report, errorMessage)
        || failIfCancelled(cancelFlag, errorMessage)
        || !transaction.publish(errorMessage))
    {
        return false;
    }

    *products = std::move(generated);
    reportProgress(progressCallback, QStringLiteral("全球 DEM/DOM 与报告完成"), 100);
    return true;
}
catch (const cv::Exception &exception)
{
    if (products)
    {
        *products = SmallBodyGlobalProducts();
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("全球 DEM/DOM 内存或栅格处理失败：%1")
                            .arg(QString::fromUtf8(exception.what()));
    }
    return false;
}
catch (const std::bad_alloc &)
{
    if (products)
    {
        *products = SmallBodyGlobalProducts();
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral(
            "全球 DEM/DOM 内存分配失败；请增大角分辨率或降低最大像元数。");
    }
    return false;
}
catch (const std::exception &exception)
{
    if (products)
    {
        *products = SmallBodyGlobalProducts();
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("全球 DEM/DOM 生成异常：%1")
                            .arg(QString::fromUtf8(exception.what()));
    }
    return false;
}

} // namespace xjw

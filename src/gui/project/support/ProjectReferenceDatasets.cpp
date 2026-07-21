#include "ProjectReferenceDatasets.h"

#include "DemDifference.h"
#include "DemDomIO.h"
#include "PointCloudAlignment.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"

#include <QDateTime>
#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>
#include <QtEndian>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace xjw::gui::project {
namespace {

QString utcNowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}

bool failWith(QString *errorMsg, const QString &message)
{
    if (errorMsg)
    {
        *errorMsg = message;
    }
    return false;
}

QString valuePath(const QJsonObject &record, const QStringList &keys)
{
    for (const QString &key : keys)
    {
        const QString path = record.value(key).toString().trimmed();
        if (!path.isEmpty())
        {
            return path;
        }
    }
    return {};
}

QString firstReferenceDemPath(const QJsonArray &references)
{
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        if (reference.value(QStringLiteral("type")).toString().toLower() != QLatin1String("dem"))
        {
            continue;
        }
        const QString path = valuePath(reference,
                                      {QStringLiteral("path"),
                                       QStringLiteral("dem_path"),
                                       QStringLiteral("file_path")});
        if (!path.isEmpty() && QFileInfo::exists(path))
        {
            return path;
        }
    }
    return {};
}

QString firstProjectDemPath(const QJsonArray &demResults)
{
    for (const QJsonValue &value : demResults)
    {
        const QJsonObject dem = value.toObject();
        const QString path = valuePath(dem,
                                      {QStringLiteral("path"),
                                       QStringLiteral("dem_path"),
                                       QStringLiteral("output_path"),
                                       QStringLiteral("raster_path")});
        if (!path.isEmpty() && QFileInfo::exists(path))
        {
            return path;
        }
    }
    return {};
}

QString firstReferenceCloudPath(const QJsonArray &references)
{
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        const QString type = reference.value(QStringLiteral("type")).toString().toLower();
        if (type != QLatin1String("point_cloud") && type != QLatin1String("lidar"))
        {
            continue;
        }
        const QString path = valuePath(reference,
                                      {QStringLiteral("path"),
                                       QStringLiteral("cloud_path"),
                                       QStringLiteral("lidar_path"),
                                       QStringLiteral("file_path")});
        if (!path.isEmpty() && QFileInfo::exists(path))
        {
            return path;
        }
    }
    return {};
}

QString firstProjectCloudPath(const QJsonArray &denseResults)
{
    for (const QJsonValue &value : denseResults)
    {
        const QJsonObject cloud = value.toObject();
        const QString path = valuePath(cloud,
                                      {QStringLiteral("path"),
                                       QStringLiteral("cloud_path"),
                                       QStringLiteral("dense_cloud_path"),
                                       QStringLiteral("point_cloud_path"),
                                       QStringLiteral("output_path")});
        if (!path.isEmpty() && QFileInfo::exists(path))
        {
            return path;
        }
    }
    return {};
}

xjw::qc::DemGrid demGridFromRaster(const DemGridData &data)
{
    xjw::qc::DemGrid grid;
    grid.width = data.width;
    grid.height = data.height;
    grid.nodata = -9999.0;
    grid.projection = data.projection.projectionWkt.isEmpty()
        ? data.projection.coordinateSystem
        : data.projection.projectionWkt;
    grid.values.assign(static_cast<std::size_t>(std::max(0, data.width) * std::max(0, data.height)),
                       grid.nodata);

    if (!data.isValid() || data.elevation.type() != CV_32FC1)
    {
        return grid;
    }

    for (int row = 0; row < data.height; ++row)
    {
        for (int col = 0; col < data.width; ++col)
        {
            const std::size_t index = static_cast<std::size_t>(row * data.width + col);
            const bool valid = data.validMask.empty() || data.validMask.at<uchar>(row, col) != 0;
            const float z = data.elevation.at<float>(row, col);
            grid.values[index] = valid && std::isfinite(static_cast<double>(z))
                ? static_cast<double>(z)
                : grid.nodata;
        }
    }
    return grid;
}

DemGridData demDifferenceRasterFromResult(const DemGridData &source,
                                          const xjw::qc::DemDifferenceResult &result,
                                          bool absoluteDifference)
{
    DemGridData raster = source;
    raster.elevation = cv::Mat(source.height, source.width, CV_32FC1, cv::Scalar(-9999.0f));
    raster.validMask = cv::Mat(source.height, source.width, CV_8UC1, cv::Scalar(0));
    raster.worldX.release();
    raster.worldY.release();
    raster.color.release();
    raster.triangulationError.release();
    raster.pointCount.release();
    raster.confidence.release();
    raster.coverageMask.release();

    for (int row = 0; row < source.height; ++row)
    {
        for (int col = 0; col < source.width; ++col)
        {
            const std::size_t index = static_cast<std::size_t>(row * source.width + col);
            if (index >= result.differences.size())
            {
                continue;
            }

            const double diff = result.differences[index];
            if (!std::isfinite(diff) || std::abs(diff - (-9999.0)) <= 1e-12)
            {
                continue;
            }

            raster.elevation.at<float>(row, col) = static_cast<float>(
                absoluteDifference ? std::abs(diff) : diff);
            raster.validMask.at<uchar>(row, col) = 255;
        }
    }
    return raster;
}

double pointDistance(const xjw::qc::Point3D &a, const xjw::qc::Point3D &b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

xjw::qc::Point3D nearestReferencePoint(const std::vector<xjw::qc::Point3D> &reference,
                                       const xjw::qc::Point3D &query)
{
    xjw::qc::Point3D best;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const xjw::qc::Point3D &point : reference)
    {
        const double d = pointDistance(point, query);
        if (d < bestDistance)
        {
            bestDistance = d;
            best = point;
        }
    }
    return best;
}

QString numberForCsv(double value)
{
    return QString::number(value, 'g', 17);
}

bool writePointCloudErrorCsv(const QString &path,
                             const std::vector<xjw::qc::Point3D> &source,
                             const std::vector<xjw::qc::Point3D> &reference,
                             const xjw::qc::SimilarityTransform *transform,
                             QString *errorMessage)
{
    if (source.empty() || reference.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("点云误差 CSV 缺少可用点: %1").arg(path);
        }
        return false;
    }

    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入点云误差 CSV: %1").arg(path);
        }
        return false;
    }

    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    if (transform)
    {
        stream << "index,source_x,source_y,source_z,reference_x,reference_y,reference_z,"
               << "aligned_x,aligned_y,aligned_z,error_m\n";
    }
    else
    {
        stream << "index,source_x,source_y,source_z,reference_x,reference_y,reference_z,error_m\n";
    }

    for (std::size_t i = 0; i < source.size(); ++i)
    {
        const xjw::qc::Point3D aligned = transform
            ? xjw::qc::PointCloudAlignment::apply(*transform, source[i])
            : source[i];
        const xjw::qc::Point3D referencePoint = source.size() == reference.size()
            ? reference[i]
            : nearestReferencePoint(reference, aligned);
        stream << i << ','
               << numberForCsv(source[i].x) << ','
               << numberForCsv(source[i].y) << ','
               << numberForCsv(source[i].z) << ','
               << numberForCsv(referencePoint.x) << ','
               << numberForCsv(referencePoint.y) << ','
               << numberForCsv(referencePoint.z) << ',';
        if (transform)
        {
            stream << numberForCsv(aligned.x) << ','
                   << numberForCsv(aligned.y) << ','
                   << numberForCsv(aligned.z) << ',';
        }
        stream << numberForCsv(pointDistance(aligned, referencePoint)) << '\n';
    }
    return true;
}

bool writePointCloudTransformJson(const QString &path,
                                  const xjw::qc::SimilarityTransform &transform,
                                  QString *errorMessage)
{
    QJsonObject translation;
    translation[QStringLiteral("x")] = transform.translation.x;
    translation[QStringLiteral("y")] = transform.translation.y;
    translation[QStringLiteral("z")] = transform.translation.z;

    QJsonObject object;
    object[QStringLiteral("type")] = QStringLiteral("sim3_point_to_point");
    object[QStringLiteral("scale")] = transform.scale;
    object[QStringLiteral("translation")] = translation;
    QJsonArray rotation;
    for (double value : transform.rotation)
    {
        rotation.append(value);
    }
    object[QStringLiteral("rotation")] = rotation;

    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入点云配准变换 JSON: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

QJsonObject buildSameGridDemDifferenceMetrics(const QString &candidatePath,
                                              const QString &referencePath,
                                              const QString &reportsDir,
                                              const QString &artifactBaseName,
                                              QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (candidatePath.isEmpty() || referencePath.isEmpty())
    {
        return {};
    }

    DemGridData candidateRaster;
    DemGridData referenceRaster;
    QString ioError;
    if (!DemDomIO::readDemRaster(candidatePath, &candidateRaster, &ioError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取项目 DEM 失败: %1").arg(ioError);
        }
        return {};
    }
    if (!DemDomIO::readDemRaster(referencePath, &referenceRaster, &ioError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取参考 DEM 失败: %1").arg(ioError);
        }
        return {};
    }

    const auto diff = xjw::qc::DemDifference::compareSameGrid(
        demGridFromRaster(candidateRaster),
        demGridFromRaster(referenceRaster),
        true);
    if (!diff.success)
    {
        if (errorMessage)
        {
            *errorMessage = diff.error;
        }
        return {};
    }

    const QString differencePath = QDir(reportsDir).filePath(
        artifactBaseName + QStringLiteral("_dem_difference.tif"));
    const QString absDifferencePath = QDir(reportsDir).filePath(
        artifactBaseName + QStringLiteral("_dem_abs_difference.tif"));
    if (!DemDomIO::writeDemRaster(demDifferenceRasterFromResult(candidateRaster, diff, false),
                                  differencePath,
                                  DemRasterFormat::Float32Tiff,
                                  &ioError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("写出 DEM 差分栅格失败: %1").arg(ioError);
        }
        return {};
    }
    if (!DemDomIO::writeDemRaster(demDifferenceRasterFromResult(candidateRaster, diff, true),
                                  absDifferencePath,
                                  DemRasterFormat::Float32Tiff,
                                  &ioError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("写出 DEM 绝对差分栅格失败: %1").arg(ioError);
        }
        return {};
    }

    QJsonObject metrics;
    metrics[QStringLiteral("dem_difference_available")] = true;
    metrics[QStringLiteral("dem_candidate_path")] = QFileInfo(candidatePath).absoluteFilePath();
    metrics[QStringLiteral("dem_reference_path")] = QFileInfo(referencePath).absoluteFilePath();
    metrics[QStringLiteral("dem_difference_path")] = QFileInfo(differencePath).absoluteFilePath();
    metrics[QStringLiteral("dem_abs_difference_path")] = QFileInfo(absDifferencePath).absoluteFilePath();
    metrics[QStringLiteral("dem_valid_count")] = diff.validCount;
    metrics[QStringLiteral("dem_mean_error_m")] = diff.mean;
    metrics[QStringLiteral("dem_rmse_m")] = diff.rmse;
    metrics[QStringLiteral("dem_median_error_m")] = diff.median;
    metrics[QStringLiteral("dem_p95_m")] = diff.p95;
    metrics[QStringLiteral("rmse_m")] = diff.rmse;
    metrics[QStringLiteral("p95_distance_m")] = diff.p95;
    return metrics;
}

bool parsePointLine(QString line, xjw::qc::Point3D *point)
{
    if (!point)
    {
        return false;
    }
    line = line.trimmed();
    if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
    {
        return false;
    }
    line.replace(QLatin1Char(','), QLatin1Char(' '));
    line.replace(QLatin1Char('\t'), QLatin1Char(' '));
    const QStringList parts = line.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (parts.size() < 3)
    {
        return false;
    }

    bool okX = false;
    bool okY = false;
    bool okZ = false;
    const double x = parts.at(0).toDouble(&okX);
    const double y = parts.at(1).toDouble(&okY);
    const double z = parts.at(2).toDouble(&okZ);
    if (!okX || !okY || !okZ)
    {
        return false;
    }
    *point = {x, y, z};
    return true;
}

bool canReadBytes(const QByteArray &bytes, qsizetype offset, qsizetype size)
{
    return offset >= 0
        && size >= 0
        && offset <= bytes.size()
        && size <= bytes.size() - offset;
}

quint16 readU16Le(const QByteArray &bytes, qsizetype offset)
{
    return qFromLittleEndian<quint16>(reinterpret_cast<const uchar *>(bytes.constData() + offset));
}

quint32 readU32Le(const QByteArray &bytes, qsizetype offset)
{
    return qFromLittleEndian<quint32>(reinterpret_cast<const uchar *>(bytes.constData() + offset));
}

quint64 readU64Le(const QByteArray &bytes, qsizetype offset)
{
    return qFromLittleEndian<quint64>(reinterpret_cast<const uchar *>(bytes.constData() + offset));
}

qint32 readI32Le(const QByteArray &bytes, qsizetype offset)
{
    return qFromLittleEndian<qint32>(reinterpret_cast<const uchar *>(bytes.constData() + offset));
}

double readF64Le(const QByteArray &bytes, qsizetype offset)
{
    const quint64 raw = readU64Le(bytes, offset);
    double value = 0.0;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

std::vector<xjw::qc::Point3D> readLasPointCloud(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 LAS 点云: %1").arg(path);
        }
        return {};
    }

    const QByteArray bytes = file.readAll();
    if (!canReadBytes(bytes, 0, 227))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("LAS 点云文件头过短: %1").arg(path);
        }
        return {};
    }
    if (!bytes.startsWith("LASF"))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("LAS 点云签名无效: %1").arg(path);
        }
        return {};
    }
    if (!canReadBytes(bytes, 171, static_cast<qsizetype>(sizeof(double))))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("LAS 点云缺少坐标比例或偏移信息: %1").arg(path);
        }
        return {};
    }

    const quint16 pointRecordLength = readU16Le(bytes, 105);
    if (pointRecordLength < 12)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("LAS 点记录长度无效: %1").arg(path);
        }
        return {};
    }

    const quint32 pointDataOffset = readU32Le(bytes, 96);
    if (pointDataOffset >= static_cast<quint32>(bytes.size()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("LAS 点数据偏移超出文件长度: %1").arg(path);
        }
        return {};
    }

    quint64 pointCount = readU32Le(bytes, 107);
    if (pointCount == 0 && canReadBytes(bytes, 247, static_cast<qsizetype>(sizeof(quint64))))
    {
        pointCount = readU64Le(bytes, 247);
    }
    if (pointCount == 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("LAS 点云没有可用 XYZ 点: %1").arg(path);
        }
        return {};
    }

    const qsizetype pointDataOffsetSigned = static_cast<qsizetype>(pointDataOffset);
    const quint64 availableRecords = static_cast<quint64>(
        (bytes.size() - pointDataOffsetSigned) / static_cast<qsizetype>(pointRecordLength));
    if (pointCount > availableRecords)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("LAS 点记录数量与文件长度不一致: %1").arg(path);
        }
        return {};
    }

    const double xScale = readF64Le(bytes, 131);
    const double yScale = readF64Le(bytes, 139);
    const double zScale = readF64Le(bytes, 147);
    const double xOffset = readF64Le(bytes, 155);
    const double yOffset = readF64Le(bytes, 163);
    const double zOffset = readF64Le(bytes, 171);

    std::vector<xjw::qc::Point3D> points;
    points.reserve(static_cast<std::size_t>(pointCount));
    for (quint64 pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        const qsizetype recordOffset = pointDataOffsetSigned
            + static_cast<qsizetype>(pointIndex * pointRecordLength);
        const qint32 xRaw = readI32Le(bytes, recordOffset);
        const qint32 yRaw = readI32Le(bytes, recordOffset + 4);
        const qint32 zRaw = readI32Le(bytes, recordOffset + 8);
        points.push_back({
            static_cast<double>(xRaw) * xScale + xOffset,
            static_cast<double>(yRaw) * yScale + yOffset,
            static_cast<double>(zRaw) * zScale + zOffset
        });
    }
    return points;
}

std::vector<xjw::qc::Point3D> readAsciiPointCloud(const QString &path, QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }

    const QFileInfo info(path);
    const QString suffix = info.suffix().toLower();
    const QString completeSuffix = info.completeSuffix().toLower();
    if (suffix == QLatin1String("las"))
    {
        return readLasPointCloud(path, errorMessage);
    }
    if (suffix == QLatin1String("laz") || completeSuffix.contains(QStringLiteral("copc")))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("暂不支持压缩 LAS/LAZ/COPC 点云: %1").arg(path);
        }
        return {};
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取点云: %1").arg(path);
        }
        return {};
    }

    QTextStream stream(&file);
    bool inPlyHeader = suffix == QLatin1String("ply");
    bool plyIsAscii = suffix != QLatin1String("ply");
    int expectedVertexCount = -1;
    std::vector<xjw::qc::Point3D> points;

    while (!stream.atEnd())
    {
        const QString line = stream.readLine();
        if (inPlyHeader)
        {
            const QString trimmed = line.trimmed();
            if (trimmed.startsWith(QStringLiteral("format ")))
            {
                plyIsAscii = trimmed.contains(QStringLiteral("ascii"));
            }
            else if (trimmed.startsWith(QStringLiteral("element vertex ")))
            {
                expectedVertexCount = trimmed.mid(QStringLiteral("element vertex ").size()).toInt();
                if (expectedVertexCount > 0)
                {
                    points.reserve(static_cast<std::size_t>(expectedVertexCount));
                }
            }
            else if (trimmed == QLatin1String("end_header"))
            {
                inPlyHeader = false;
                if (!plyIsAscii)
                {
                    if (errorMessage)
                    {
                        *errorMessage = QStringLiteral("暂仅支持 ASCII PLY 点云: %1").arg(path);
                    }
                    return {};
                }
            }
            continue;
        }

        xjw::qc::Point3D point;
        if (parsePointLine(line, &point))
        {
            points.push_back(point);
            if (expectedVertexCount > 0
                && points.size() >= static_cast<std::size_t>(expectedVertexCount))
            {
                break;
            }
        }
    }

    if (points.empty() && errorMessage)
    {
        *errorMessage = QStringLiteral("点云没有可用 XYZ 点: %1").arg(path);
    }
    return points;
}

QJsonObject buildPairedPointCloudMetrics(const QString &candidatePath,
                                         const QString &referencePath,
                                         const QString &reportsDir,
                                         const QString &artifactBaseName,
                                         QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (candidatePath.isEmpty() || referencePath.isEmpty())
    {
        return {};
    }

    QString readError;
    const std::vector<xjw::qc::Point3D> candidate = readAsciiPointCloud(candidatePath, &readError);
    if (candidate.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取项目点云失败: %1").arg(readError);
        }
        return {};
    }
    const std::vector<xjw::qc::Point3D> reference = readAsciiPointCloud(referencePath, &readError);
    if (reference.empty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取参考点云失败: %1").arg(readError);
        }
        return {};
    }

    const auto alignment = candidate.size() == reference.size()
        ? xjw::qc::PointCloudAlignment::alignPairedSimilarity(candidate, reference)
        : xjw::qc::PointCloudAlignment::alignNearestNeighborTranslation(candidate, reference);
    if (!alignment.success)
    {
        if (errorMessage)
        {
            *errorMessage = alignment.error;
        }
        return {};
    }

    const QString beforeErrorsPath = QDir(reportsDir).filePath(
        artifactBaseName + QStringLiteral("_cloud_beg_errors.csv"));
    const QString afterErrorsPath = QDir(reportsDir).filePath(
        artifactBaseName + QStringLiteral("_cloud_end_errors.csv"));
    const QString transformPath = QDir(reportsDir).filePath(
        artifactBaseName + QStringLiteral("_cloud_transform.json"));
    if (!writePointCloudErrorCsv(beforeErrorsPath, candidate, reference, nullptr, errorMessage))
    {
        return {};
    }
    if (!writePointCloudErrorCsv(afterErrorsPath, candidate, reference, &alignment.transform, errorMessage))
    {
        return {};
    }
    if (!writePointCloudTransformJson(transformPath, alignment.transform, errorMessage))
    {
        return {};
    }

    QJsonObject metrics;
    metrics[QStringLiteral("cloud_difference_available")] = true;
    metrics[QStringLiteral("cloud_candidate_path")] = QFileInfo(candidatePath).absoluteFilePath();
    metrics[QStringLiteral("cloud_reference_path")] = QFileInfo(referencePath).absoluteFilePath();
    metrics[QStringLiteral("cloud_beg_errors_csv_path")] = QFileInfo(beforeErrorsPath).absoluteFilePath();
    metrics[QStringLiteral("cloud_end_errors_csv_path")] = QFileInfo(afterErrorsPath).absoluteFilePath();
    metrics[QStringLiteral("cloud_transform_json_path")] = QFileInfo(transformPath).absoluteFilePath();
    metrics[QStringLiteral("cloud_alignment_method")] = alignment.method;
    metrics[QStringLiteral("cloud_pair_count")] = alignment.pairCount;
    metrics[QStringLiteral("cloud_alignment_scale")] = alignment.transform.scale;
    metrics[QStringLiteral("cloud_alignment_tx_m")] = alignment.transform.translation.x;
    metrics[QStringLiteral("cloud_alignment_ty_m")] = alignment.transform.translation.y;
    metrics[QStringLiteral("cloud_alignment_tz_m")] = alignment.transform.translation.z;
    metrics[QStringLiteral("cloud_rmse_before_m")] = alignment.before.rmse;
    metrics[QStringLiteral("cloud_mean_before_m")] = alignment.before.mean;
    metrics[QStringLiteral("cloud_p95_before_m")] = alignment.before.p95;
    metrics[QStringLiteral("cloud_rmse_m")] = alignment.after.rmse;
    metrics[QStringLiteral("cloud_mean_m")] = alignment.after.mean;
    metrics[QStringLiteral("cloud_median_m")] = alignment.after.median;
    metrics[QStringLiteral("cloud_p95_m")] = alignment.after.p95;
    metrics[QStringLiteral("point_cloud_rmse_m")] = alignment.after.rmse;
    return metrics;
}

} // namespace

QString referenceDatasetTypeForPath(const QString &path)
{
    const QFileInfo fi(path);
    const QString suffix = fi.suffix().toLower();
    const QString completeSuffix = fi.completeSuffix().toLower();

    if (suffix == QLatin1String("tif")
        || suffix == QLatin1String("tiff")
        || suffix == QLatin1String("vrt"))
    {
        return QStringLiteral("dem");
    }
    if (suffix == QLatin1String("las")
        || suffix == QLatin1String("laz")
        || suffix == QLatin1String("copc")
        || completeSuffix.contains(QStringLiteral("copc")))
    {
        return QStringLiteral("lidar");
    }
    if (suffix == QLatin1String("ply")
        || suffix == QLatin1String("xyz")
        || suffix == QLatin1String("csv"))
    {
        return QStringLiteral("point_cloud");
    }
    return {};
}

QString normalizeReferenceDatasetType(const QString &type, const QString &path)
{
    QString normalized = type.trimmed().toLower();
    if (normalized.isEmpty())
    {
        normalized = referenceDatasetTypeForPath(path);
    }
    if (normalized == QLatin1String("las") || normalized == QLatin1String("laz"))
    {
        normalized = QStringLiteral("lidar");
    }
    if (normalized == QLatin1String("tiff") || normalized == QLatin1String("geotiff"))
    {
        normalized = QStringLiteral("dem");
    }
    return normalized;
}

bool registerReferenceDataset(ProjectData *projectData,
                              const QString &path,
                              const QString &type,
                              const QString &role,
                              QString *errorMsg)
{
    if (!projectData || !projectData->hasProject())
    {
        return failWith(errorMsg, QStringLiteral("请先打开项目，再导入参考数据。"));
    }

    const QFileInfo fi(path);
    if (path.trimmed().isEmpty())
    {
        return failWith(errorMsg, QStringLiteral("参考数据路径为空。"));
    }
    if (!fi.exists() || !fi.isFile())
    {
        return failWith(errorMsg, QStringLiteral("参考数据文件不存在: %1").arg(path));
    }

    const QString normalizedType = normalizeReferenceDatasetType(type, path);
    if (normalizedType.isEmpty())
    {
        return failWith(errorMsg, QStringLiteral("无法识别参考数据类型: %1").arg(fi.fileName()));
    }

    QString normalizedRole = role.trimmed().toLower();
    if (normalizedRole.isEmpty())
    {
        normalizedRole = QStringLiteral("validation");
    }

    QJsonObject record;
    record[QStringLiteral("path")] = QDir::cleanPath(fi.absoluteFilePath());
    record[QStringLiteral("type")] = normalizedType;
    record[QStringLiteral("role")] = normalizedRole;
    record[QStringLiteral("storage")] = QStringLiteral("reference");
    record[QStringLiteral("created_at")] = utcNowIso();

    if (!projectData->upsertResultRecordByPath(QStringLiteral("reference_datasets"),
                                               QStringLiteral("path"),
                                               record,
                                               true))
    {
        return failWith(errorMsg, QStringLiteral("写入参考数据记录失败: %1").arg(fi.absoluteFilePath()));
    }

    if (errorMsg)
    {
        errorMsg->clear();
    }
    return true;
}

ReferenceDatasetQualityReportResult writeReferenceDatasetQualityReport(ProjectData *projectData,
                                                                       const QString &baseName)
{
    ReferenceDatasetQualityReportResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目未打开，无法生成参考数据精度检查报告");
        return result;
    }

    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectData->currentProjectPath());
    if (assetsDir.isEmpty())
    {
        result.errorMessage = QStringLiteral("无法解析项目 assets 目录");
        return result;
    }

    const QJsonObject meta = projectData->metadata();
    const QJsonArray references = meta.value(QStringLiteral("reference_datasets")).toArray();
    const QJsonArray demResults = meta.value(QStringLiteral("dem_results")).toArray();
    const QJsonArray denseResults = meta.value(QStringLiteral("dense_cloud_results")).toArray();

    int demReferences = 0;
    int lidarReferences = 0;
    int pointCloudReferences = 0;
    for (const QJsonValue &value : references)
    {
        const QString type = value.toObject().value(QStringLiteral("type")).toString();
        if (type == QLatin1String("dem"))
        {
            ++demReferences;
        }
        else if (type == QLatin1String("lidar"))
        {
            ++lidarReferences;
        }
        else if (type == QLatin1String("point_cloud"))
        {
            ++pointCloudReferences;
        }
    }

    const bool demComparisonAvailable = demReferences > 0 && !demResults.isEmpty();
    const bool cloudComparisonAvailable =
        (lidarReferences > 0 || pointCloudReferences > 0) && !denseResults.isEmpty();
    const bool comparisonAvailable = demComparisonAvailable || cloudComparisonAvailable;

    QString status = QStringLiteral("ready");
    if (references.isEmpty())
    {
        status = QStringLiteral("missing_reference_datasets");
    }
    else if (!comparisonAvailable)
    {
        status = QStringLiteral("missing_project_products");
    }

    QJsonObject products;
    products[QStringLiteral("dem_count")] = demResults.size();
    products[QStringLiteral("dense_cloud_count")] = denseResults.size();

    const QString safeBaseName = baseName.trimmed().isEmpty()
        ? QStringLiteral("reference_quality_report")
        : baseName.trimmed();
    const QString reportsDir = QDir(assetsDir).filePath(QStringLiteral("reports"));
    QDir().mkpath(reportsDir);

    QString demDifferenceError;
    const QJsonObject demDifferenceMetrics = buildSameGridDemDifferenceMetrics(
        firstProjectDemPath(demResults),
        firstReferenceDemPath(references),
        reportsDir,
        safeBaseName,
        &demDifferenceError);
    QString cloudDifferenceError;
    const QJsonObject cloudDifferenceMetrics = buildPairedPointCloudMetrics(
        firstProjectCloudPath(denseResults),
        firstReferenceCloudPath(references),
        reportsDir,
        safeBaseName,
        &cloudDifferenceError);

    QJsonObject report;
    report[QStringLiteral("type")] = QStringLiteral("reference_quality");
    report[QStringLiteral("created_at")] = utcNowIso();
    report[QStringLiteral("status")] = status;
    report[QStringLiteral("reference_count")] = references.size();
    report[QStringLiteral("dem_reference_count")] = demReferences;
    report[QStringLiteral("lidar_reference_count")] = lidarReferences;
    report[QStringLiteral("point_cloud_reference_count")] = pointCloudReferences;
    report[QStringLiteral("comparison_available")] = comparisonAvailable;
    report[QStringLiteral("dem_comparison_available")] = demComparisonAvailable;
    report[QStringLiteral("cloud_comparison_available")] = cloudComparisonAvailable;
    report[QStringLiteral("reference_datasets")] = references;
    report[QStringLiteral("project_products")] = products;
    if (!demDifferenceMetrics.isEmpty())
    {
        for (auto it = demDifferenceMetrics.constBegin(); it != demDifferenceMetrics.constEnd(); ++it)
        {
            report.insert(it.key(), it.value());
        }
    }
    else if (demComparisonAvailable)
    {
        report[QStringLiteral("dem_difference_available")] = false;
        if (!demDifferenceError.isEmpty())
        {
            report[QStringLiteral("dem_difference_error")] = demDifferenceError;
        }
    }
    if (!cloudDifferenceMetrics.isEmpty())
    {
        for (auto it = cloudDifferenceMetrics.constBegin(); it != cloudDifferenceMetrics.constEnd(); ++it)
        {
            report.insert(it.key(), it.value());
        }
        if (demDifferenceMetrics.isEmpty())
        {
            report[QStringLiteral("rmse_m")] = cloudDifferenceMetrics.value(QStringLiteral("cloud_rmse_m"));
            report[QStringLiteral("p95_distance_m")] = cloudDifferenceMetrics.value(QStringLiteral("cloud_p95_m"));
        }
    }
    else if (cloudComparisonAvailable)
    {
        report[QStringLiteral("cloud_difference_available")] = false;
        if (!cloudDifferenceError.isEmpty())
        {
            report[QStringLiteral("cloud_difference_error")] = cloudDifferenceError;
        }
    }

    const QString jsonPath = QDir(reportsDir).filePath(safeBaseName + QStringLiteral(".json"));
    const QString csvPath = QDir(reportsDir).filePath(safeBaseName + QStringLiteral(".csv"));

    QFile jsonFile(jsonPath);
    if (!jsonFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        result.errorMessage = QStringLiteral("无法写入参考数据精度检查报告: %1").arg(jsonPath);
        return result;
    }
    jsonFile.write(QJsonDocument(report).toJson(QJsonDocument::Indented));
    jsonFile.close();

    QFile csvFile(csvPath);
    if (!csvFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        result.errorMessage = QStringLiteral("无法写入参考数据精度检查 CSV: %1").arg(csvPath);
        return result;
    }
    QTextStream stream(&csvFile);
    stream.setEncoding(QStringConverter::Utf8);
    stream << "metric,value\n";
    stream << "status," << status << "\n";
    stream << "reference_count," << references.size() << "\n";
    stream << "dem_reference_count," << demReferences << "\n";
    stream << "lidar_reference_count," << lidarReferences << "\n";
    stream << "point_cloud_reference_count," << pointCloudReferences << "\n";
    stream << "dem_count," << demResults.size() << "\n";
    stream << "dense_cloud_count," << denseResults.size() << "\n";
    stream << "comparison_available," << (comparisonAvailable ? 1 : 0) << "\n";
    if (!demDifferenceMetrics.isEmpty())
    {
        stream << "dem_valid_count," << demDifferenceMetrics.value(QStringLiteral("dem_valid_count")).toInt() << "\n";
        stream << "dem_mean_error_m," << demDifferenceMetrics.value(QStringLiteral("dem_mean_error_m")).toDouble() << "\n";
        stream << "dem_rmse_m," << demDifferenceMetrics.value(QStringLiteral("dem_rmse_m")).toDouble() << "\n";
        stream << "dem_median_error_m," << demDifferenceMetrics.value(QStringLiteral("dem_median_error_m")).toDouble() << "\n";
        stream << "dem_p95_m," << demDifferenceMetrics.value(QStringLiteral("dem_p95_m")).toDouble() << "\n";
    }
    else if (demComparisonAvailable && !demDifferenceError.isEmpty())
    {
        stream << "dem_difference_error," << demDifferenceError << "\n";
    }
    if (!cloudDifferenceMetrics.isEmpty())
    {
        stream << "cloud_pair_count," << cloudDifferenceMetrics.value(QStringLiteral("cloud_pair_count")).toInt() << "\n";
        stream << "cloud_alignment_scale," << cloudDifferenceMetrics.value(QStringLiteral("cloud_alignment_scale")).toDouble() << "\n";
        stream << "cloud_rmse_before_m," << cloudDifferenceMetrics.value(QStringLiteral("cloud_rmse_before_m")).toDouble() << "\n";
        stream << "cloud_rmse_m," << cloudDifferenceMetrics.value(QStringLiteral("cloud_rmse_m")).toDouble() << "\n";
        stream << "cloud_p95_m," << cloudDifferenceMetrics.value(QStringLiteral("cloud_p95_m")).toDouble() << "\n";
    }
    else if (cloudComparisonAvailable && !cloudDifferenceError.isEmpty())
    {
        stream << "cloud_difference_error," << cloudDifferenceError << "\n";
    }
    csvFile.close();

    QJsonObject record;
    record[QStringLiteral("created_at")] = report.value(QStringLiteral("created_at"));
    record[QStringLiteral("type")] = QStringLiteral("reference_quality");
    record[QStringLiteral("path")] = jsonPath;
    record[QStringLiteral("json_path")] = jsonPath;
    record[QStringLiteral("csv_path")] = csvPath;
    record[QStringLiteral("status")] = status;
    record[QStringLiteral("reference_count")] = references.size();
    record[QStringLiteral("comparison_available")] = comparisonAvailable;
    if (!demDifferenceMetrics.isEmpty())
    {
        for (auto it = demDifferenceMetrics.constBegin(); it != demDifferenceMetrics.constEnd(); ++it)
        {
            record.insert(it.key(), it.value());
        }
    }
    else if (demComparisonAvailable && !demDifferenceError.isEmpty())
    {
        record[QStringLiteral("dem_difference_available")] = false;
            record[QStringLiteral("dem_difference_error")] = demDifferenceError;
    }
    if (!cloudDifferenceMetrics.isEmpty())
    {
        for (auto it = cloudDifferenceMetrics.constBegin(); it != cloudDifferenceMetrics.constEnd(); ++it)
        {
            record.insert(it.key(), it.value());
        }
        if (demDifferenceMetrics.isEmpty())
        {
            record[QStringLiteral("rmse_m")] = cloudDifferenceMetrics.value(QStringLiteral("cloud_rmse_m"));
            record[QStringLiteral("p95_distance_m")] = cloudDifferenceMetrics.value(QStringLiteral("cloud_p95_m"));
        }
    }
    else if (cloudComparisonAvailable && !cloudDifferenceError.isEmpty())
    {
        record[QStringLiteral("cloud_difference_available")] = false;
        record[QStringLiteral("cloud_difference_error")] = cloudDifferenceError;
    }

    if (!projectData->upsertResultRecordByPath(QStringLiteral("report_results"),
                                               QStringLiteral("path"),
                                               record,
                                               true))
    {
        result.errorMessage = QStringLiteral("参考数据精度检查报告已写出，但写入项目 metadata 失败");
        return result;
    }

    result.saved = true;
    result.jsonPath = jsonPath;
    result.csvPath = csvPath;
    result.record = record;
    return result;
}

ReferenceDatasetQualityReportResult writeReferenceTerrainPriorPreflightReport(ProjectData *projectData,
                                                                              const QString &baseName)
{
    ReferenceDatasetQualityReportResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目未打开，无法生成参考地形平差前置检查报告");
        return result;
    }

    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectData->currentProjectPath());
    if (assetsDir.isEmpty())
    {
        result.errorMessage = QStringLiteral("无法解析项目 assets 目录");
        return result;
    }

    const QJsonObject meta = projectData->metadata();
    const QJsonArray references = meta.value(QStringLiteral("reference_datasets")).toArray();
    const QJsonArray atResults = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    const QJsonArray baResults = meta.value(QStringLiteral("bundle_adjust_results")).toArray();

    int baPriorReferences = 0;
    QJsonArray priorReferences;
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        const QString role = reference.value(QStringLiteral("role")).toString().toLower();
        if (role == QLatin1String("ba_prior")
            || role == QLatin1String("bundle_adjustment")
            || role == QLatin1String("reference_prior"))
        {
            ++baPriorReferences;
            priorReferences.append(reference);
        }
    }

    QString status = QStringLiteral("ready");
    if (references.isEmpty())
    {
        status = QStringLiteral("missing_reference_datasets");
    }
    else if (baPriorReferences == 0)
    {
        status = QStringLiteral("missing_ba_prior_reference");
    }
    else if (atResults.isEmpty())
    {
        status = QStringLiteral("missing_aerial_triangulation");
    }

    const bool ready = status == QLatin1String("ready");

    QJsonObject report;
    report[QStringLiteral("type")] = QStringLiteral("reference_terrain_prior_preflight");
    report[QStringLiteral("created_at")] = utcNowIso();
    report[QStringLiteral("status")] = status;
    report[QStringLiteral("ready")] = ready;
    report[QStringLiteral("reference_count")] = references.size();
    report[QStringLiteral("ba_prior_reference_count")] = baPriorReferences;
    report[QStringLiteral("aerial_triangulation_result_count")] = atResults.size();
    report[QStringLiteral("bundle_adjust_result_count")] = baResults.size();
    report[QStringLiteral("recommended_sigma_m")] = 1.0;
    report[QStringLiteral("recommended_huber_delta_m")] = 0.5;
    report[QStringLiteral("prior_references")] = priorReferences;

    const QString safeBaseName = baseName.trimmed().isEmpty()
        ? QStringLiteral("reference_terrain_prior_preflight")
        : baseName.trimmed();
    const QString reportsDir = QDir(assetsDir).filePath(QStringLiteral("reports"));
    QDir().mkpath(reportsDir);
    const QString jsonPath = QDir(reportsDir).filePath(safeBaseName + QStringLiteral(".json"));
    const QString csvPath = QDir(reportsDir).filePath(safeBaseName + QStringLiteral(".csv"));

    QFile jsonFile(jsonPath);
    if (!jsonFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        result.errorMessage = QStringLiteral("无法写入参考地形平差前置检查报告: %1").arg(jsonPath);
        return result;
    }
    jsonFile.write(QJsonDocument(report).toJson(QJsonDocument::Indented));
    jsonFile.close();

    QFile csvFile(csvPath);
    if (!csvFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        result.errorMessage = QStringLiteral("无法写入参考地形平差前置检查 CSV: %1").arg(csvPath);
        return result;
    }
    QTextStream stream(&csvFile);
    stream.setEncoding(QStringConverter::Utf8);
    stream << "metric,value\n";
    stream << "status," << status << "\n";
    stream << "ready," << (ready ? 1 : 0) << "\n";
    stream << "reference_count," << references.size() << "\n";
    stream << "ba_prior_reference_count," << baPriorReferences << "\n";
    stream << "aerial_triangulation_result_count," << atResults.size() << "\n";
    stream << "bundle_adjust_result_count," << baResults.size() << "\n";
    csvFile.close();

    QJsonObject record;
    record[QStringLiteral("created_at")] = report.value(QStringLiteral("created_at"));
    record[QStringLiteral("type")] = QStringLiteral("reference_terrain_prior_preflight");
    record[QStringLiteral("path")] = jsonPath;
    record[QStringLiteral("json_path")] = jsonPath;
    record[QStringLiteral("csv_path")] = csvPath;
    record[QStringLiteral("status")] = status;
    record[QStringLiteral("ready")] = ready;
    record[QStringLiteral("ba_prior_reference_count")] = baPriorReferences;
    record[QStringLiteral("recommended_sigma_m")] = report.value(QStringLiteral("recommended_sigma_m"));
    record[QStringLiteral("recommended_huber_delta_m")] = report.value(QStringLiteral("recommended_huber_delta_m"));

    if (!projectData->upsertResultRecordByPath(QStringLiteral("report_results"),
                                               QStringLiteral("path"),
                                               record,
                                               true))
    {
        result.errorMessage = QStringLiteral("参考地形平差前置检查报告已写出，但写入项目 metadata 失败");
        return result;
    }

    result.saved = true;
    result.jsonPath = jsonPath;
    result.csvPath = csvPath;
    result.record = record;
    return result;
}

} // namespace xjw::gui::project

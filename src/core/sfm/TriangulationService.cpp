#include "TriangulationService.h"

#include "BaInputBuilder.h"
#include "triangulation/InitialSparsePointCloudTriangulator.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QTextStream>

#include <cmath>

namespace xjw::core::project
{

namespace
{

QStringList projectImagePaths(const QJsonObject &meta)
{
    QStringList imagePaths;
    const QJsonArray imageEntries = meta.value(QStringLiteral("images")).toArray();
    imagePaths.reserve(imageEntries.size());

    for (const QJsonValue &imageValue : imageEntries)
    {
        const QString imagePath = imageValue.toObject().value(QStringLiteral("path")).toString();
        if (!imagePath.isEmpty())
        {
            imagePaths.push_back(imagePath);
        }
    }

    return imagePaths;
}

} // namespace

TriangulationServiceResult TriangulationService::run(const QJsonObject &meta,
                                                     const QStringList &selectedImages,
                                                     const TriangulationServiceOptions &options)
{
    TriangulationServiceResult result;

    if (options.outputDir.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("输出目录未指定");
        return result;
    }

    const QStringList images = selectedImages.isEmpty() ? projectImagePaths(meta) : selectedImages;
    if (images.size() < 2)
    {
        result.errorMessage = QStringLiteral("至少需要两张影像才能执行三角化");
        return result;
    }

    BaInputBuildResult buildResult;
    const BaInputBuildStatus buildStatus = buildBaInputFromMeta(meta, images, 1, &buildResult);
    if (buildStatus == BaInputBuildStatus::NotEnoughCameras)
    {
        result.errorMessage = QStringLiteral("所选影像中可用相机参数不足（至少需要两台相机）");
        return result;
    }
    if (buildStatus == BaInputBuildStatus::NoTracks)
    {
        result.errorMessage = QStringLiteral("未找到可用于三角化的匹配轨迹");
        return result;
    }

    xjw::InitialSparseTriangulationOptions coreOptions;
    coreOptions.minTriAngleDeg = options.minTriAngleDeg;
    coreOptions.maxReprojErrorPx = options.maxReprojErrorPx;
    coreOptions.minObservations = options.minObservations;
    coreOptions.ignoreTwoViewTracks = options.ignoreTwoViewTracks;
    coreOptions.minTrackLength = options.minTrackLength;

    const xjw::InitialSparseTriangulationResult coreResult =
        xjw::InitialSparsePointCloudFilter::triangulate(buildResult.cameras,
                                                        buildResult.tracks,
                                                        coreOptions);

    result.candidateTrackCount = coreResult.candidateTrackCount;
    result.rejectedByObservationCount = coreResult.rejectedByObservationCount;
    result.rejectedByTriAngleCount = coreResult.rejectedByTriAngleCount;
    result.rejectedByReprojCount = coreResult.rejectedByReprojCount;

    if (!coreResult.success)
    {
        result.errorMessage = QStringLiteral(
            "未生成有效稀疏点云。候选轨迹=%1，观测不足剔除=%2，交会角剔除=%3，重投影剔除=%4。"
            "请检查相机单位、匹配质量或放宽阈值。")
                                  .arg(result.candidateTrackCount)
                                  .arg(result.rejectedByObservationCount)
                                  .arg(result.rejectedByTriAngleCount)
                                  .arg(result.rejectedByReprojCount);
        return result;
    }

    QJsonArray pointsArray;
    QVector<std::array<double, 3>> exportedPoints;
    exportedPoints.reserve(static_cast<int>(coreResult.points.size()));

    const int camCount = buildResult.imagePathByIndex.size();
    QVector<QImage> camImages(camCount);
    auto sampleColor = [&](const xjw::InitialSparsePoint &point,
                           std::array<uint8_t, 3> &color) -> bool {
        if (point.sourceTrackIndex < 0
            || point.sourceTrackIndex >= static_cast<int>(buildResult.tracks.size()))
        {
            return false;
        }

        const xjw::BATrack &track = buildResult.tracks[
            static_cast<std::size_t>(point.sourceTrackIndex)];

        int sampleCount = 0;
        int sumR = 0;
        int sumG = 0;
        int sumB = 0;

        for (const xjw::BAObservation &observation : track.observations)
        {
            if (observation.cameraIndex < 0 || observation.cameraIndex >= camCount)
            {
                continue;
            }

            if (!std::isfinite(observation.u) || !std::isfinite(observation.v))
            {
                continue;
            }

            QImage &img = camImages[observation.cameraIndex];
            if (img.isNull())
            {
                const QString &path = buildResult.imagePathByIndex[observation.cameraIndex];
                img = QImage(path).convertToFormat(QImage::Format_RGB888);
            }
            if (img.isNull())
            {
                continue;
            }

            const int px = qBound(0, qRound(observation.u), img.width() - 1);
            const int py = qBound(0, qRound(observation.v), img.height() - 1);
            const QRgb pix = img.pixel(px, py);
            sumR += qRed(pix);
            sumG += qGreen(pix);
            sumB += qBlue(pix);
            ++sampleCount;
        }

        if (sampleCount <= 0)
        {
            return false;
        }

        color = {
            static_cast<uint8_t>(sumR / sampleCount),
            static_cast<uint8_t>(sumG / sampleCount),
            static_cast<uint8_t>(sumB / sampleCount)
        };
        return true;
    };

    struct ExportPoint
    {
        std::array<double, 3> xyz;
        std::array<uint8_t, 3> rgb;
    };

    QVector<ExportPoint> exportWithColor;
    exportWithColor.reserve(static_cast<int>(coreResult.points.size()));

    for (const xjw::InitialSparsePoint &point : coreResult.points)
    {
        exportedPoints.push_back(point.xyz);

        ExportPoint exportPoint;
        exportPoint.xyz = point.xyz;
        exportPoint.rgb = {128, 128, 128};
        sampleColor(point, exportPoint.rgb);
        exportWithColor.push_back(exportPoint);

        QJsonObject pointObject;
        pointObject[QStringLiteral("track_len")] = point.trackLength;
        pointObject[QStringLiteral("min_tri_angle_deg")] = point.minTriAngleDeg;
        pointObject[QStringLiteral("rms_reproj_px")] = point.rmsReprojPx;

        QJsonArray xyzArray;
        xyzArray.append(point.xyz[0]);
        xyzArray.append(point.xyz[1]);
        xyzArray.append(point.xyz[2]);
        pointObject[QStringLiteral("point_xyz")] = xyzArray;
        pointsArray.append(pointObject);
    }

    QDir().mkpath(options.outputDir);
    const QString sparseCloudPath = QDir(options.outputDir).filePath(QStringLiteral("sparse_cloud.ply"));
    QFile outputFile(sparseCloudPath);
    if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        result.errorMessage = QStringLiteral("无法写入稀疏点云文件: %1").arg(sparseCloudPath);
        return result;
    }

    QTextStream out(&outputFile);
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment generated by PlaScan triangulation\n";
    out << "element vertex " << exportWithColor.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const ExportPoint &exportPoint : exportWithColor)
    {
        out << QStringLiteral("%1 %2 %3 %4 %5 %6\n")
                   .arg(exportPoint.xyz[0], 0, 'f', 6)
                   .arg(exportPoint.xyz[1], 0, 'f', 6)
                   .arg(exportPoint.xyz[2], 0, 'f', 6)
                   .arg(exportPoint.rgb[0])
                   .arg(exportPoint.rgb[1])
                   .arg(exportPoint.rgb[2]);
    }
    outputFile.close();

    result.success = true;
    result.sparseCloudPath = sparseCloudPath;
    result.exportedPointCount = coreResult.exportedPointCount;

    QJsonObject summary;
    summary[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    summary[QStringLiteral("output_dir")] = options.outputDir;
    summary[QStringLiteral("selected_images")] = QJsonArray::fromStringList(images);
    summary[QStringLiteral("candidate_track_count")] = result.candidateTrackCount;
    summary[QStringLiteral("exported_point_count")] = result.exportedPointCount;
    summary[QStringLiteral("rejected_by_observation_count")] = result.rejectedByObservationCount;
    summary[QStringLiteral("rejected_by_tri_angle_count")] = result.rejectedByTriAngleCount;
    summary[QStringLiteral("rejected_by_reproj_count")] = result.rejectedByReprojCount;
    summary[QStringLiteral("sparse_cloud_path")] = result.sparseCloudPath;
    summary[QStringLiteral("points")] = pointsArray;
    result.resultJson = summary;

    return result;
}

} // namespace xjw::core::project

/**
 * @file AerialTriangulationResultWriter.cpp
 * @brief 胜出 SfM 模型的稀疏点云、质量 sidecar 和工程记录构建实现。
 *
 * PLY 使用 QSaveFile 原子提交；质量 JSON 使用通用原子 IO。相机更新仍保留在
 * pendingCamUpdates，由工程服务在本函数完全成功后统一应用。
 */

#include "reporting/AerialTriangulationResultWriter.h"

#include "io/PathIO.h"
#include "project/SparseResultQuality.h"
#include "reconstruction/SfmReconstruction.h"
#include "reporting/QualityReportWriter.h"

#include <QColor>
#include <QDataStream>
#include <QDir>
#include <QImage>
#include <QJsonDocument>
#include <QSaveFile>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace xjw::aerial_triangulation
{
namespace
{

struct ExportPoint
{
    std::array<float, 3> xyz{}; ///< 最终 BA 坐标系中的三维坐标。
    std::array<quint8, 3> color{{128, 128, 128}}; ///< 无可读影像时使用中性灰。
    ImageId colorImageId = kInvalidImageId; ///< 采样颜色所用的首个有效轨迹观测。
    FeatureIdx colorFeatureIndex = kInvalidFeatureIdx; ///< 对应影像内关键点索引。
};

bool fail(const QString &message, QString *errorMessage)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
    return false;
}

/**
 * @brief 从最终重建收集可发布点。
 *
 * 发布门槛与质量报告保持一致：轨迹至少两视、误差不高于 4 px、坐标有限。
 * 这里不修改 reconstruction，也不重新三角化。
 */
std::vector<ExportPoint> collectExportPoints(const SfmReconstruction &reconstruction)
{
    std::vector<ExportPoint> points;
    points.reserve(reconstruction.numPoints3D());
    for (const Point3DId pointId : reconstruction.allPoint3DIds())
    {
        if (!reconstruction.hasPoint3D(pointId))
        {
            continue;
        }
        const ScenePoint3D &point = reconstruction.point3D(pointId);
        if (point.error > 4.0 || point.track.length() < 2 ||
            !std::isfinite(point.xyz[0]) || !std::isfinite(point.xyz[1]) ||
            !std::isfinite(point.xyz[2]))
        {
            continue;
        }

        ExportPoint output;
        output.xyz = {static_cast<float>(point.xyz[0]),
                      static_cast<float>(point.xyz[1]),
                      static_cast<float>(point.xyz[2])};
        for (const TrackElement &element : point.track.elements)
        {
            if (reconstruction.hasImage(element.imageId) &&
                element.featureIdx < reconstruction.image(element.imageId).keypoints.size())
            {
                output.colorImageId = element.imageId;
                output.colorFeatureIndex = element.featureIdx;
                break;
            }
        }
        points.push_back(output);
    }
    return points;
}

/**
 * @brief 按影像分组读取颜色，避免每个点重复解码同一文件。
 *
 * 颜色只用于可视化，不参与点的有效性；影像读取失败时保留默认灰色。
 */
void samplePointColors(const SfmReconstruction &reconstruction,
                       std::vector<ExportPoint> *points)
{
    if (!points)
    {
        return;
    }
    QMap<ImageId, std::vector<std::size_t>> requestsByImage;
    for (std::size_t index = 0; index < points->size(); ++index)
    {
        if ((*points)[index].colorImageId != kInvalidImageId)
        {
            requestsByImage[(*points)[index].colorImageId].push_back(index);
        }
    }

    for (auto request = requestsByImage.cbegin(); request != requestsByImage.cend(); ++request)
    {
        if (!reconstruction.hasImage(request.key()))
        {
            continue;
        }
        const ImageData &imageData = reconstruction.image(request.key());
        const QImage image(QString::fromStdString(imageData.imagePath));
        if (image.isNull())
        {
            continue;
        }
        for (const std::size_t pointIndex : request.value())
        {
            ExportPoint &point = (*points)[pointIndex];
            if (point.colorFeatureIndex >= imageData.keypoints.size())
            {
                continue;
            }
            const FeatureKeypoint &keypoint = imageData.keypoints[point.colorFeatureIndex];
            const int x = std::clamp(qRound(keypoint.x), 0, image.width() - 1);
            const int y = std::clamp(qRound(keypoint.y), 0, image.height() - 1);
            const QColor color = image.pixelColor(x, y);
            point.color = {static_cast<quint8>(color.red()),
                           static_cast<quint8>(color.green()),
                           static_cast<quint8>(color.blue())};
        }
    }
}

/// 原子写入 little-endian binary PLY，失败时旧文件保持不变。
bool writeBinaryPly(const QString &path,
                    const std::vector<ExportPoint> &points,
                    QString *errorMessage)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        return fail(QStringLiteral("无法写入稀疏点云文件: %1").arg(path), errorMessage);
    }

    const QByteArray header = QByteArrayLiteral(
        "ply\nformat binary_little_endian 1.0\ncomment PlaScan aerial triangulation\n") +
        QByteArray("element vertex ") + QByteArray::number(points.size()) + QByteArrayLiteral(
        "\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
    if (file.write(header) != header.size())
    {
        return fail(QStringLiteral("写入稀疏点云头失败: %1").arg(path), errorMessage);
    }

    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    for (const ExportPoint &point : points)
    {
        stream << point.xyz[0] << point.xyz[1] << point.xyz[2];
        const char colors[3]{static_cast<char>(point.color[0]),
                             static_cast<char>(point.color[1]),
                             static_cast<char>(point.color[2])};
        if (stream.writeRawData(colors, 3) != 3)
        {
            return fail(QStringLiteral("写入稀疏点云数据失败: %1").arg(path), errorMessage);
        }
    }
    if (stream.status() != QDataStream::Ok || !file.commit())
    {
        return fail(QStringLiteral("提交稀疏点云文件失败: %1").arg(path), errorMessage);
    }
    return true;
}

} // namespace

bool AerialTriangulationResultWriter::write(
    const PreparedAerialTriangulationInput &input,
    SfmAttemptExecutionResult *execution,
    QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!execution || !execution->result.success || !execution->reconstruction)
    {
        return fail(QStringLiteral("没有可写出的 SfM 内存重建结果"), errorMessage);
    }
    if (input.outputDir.trimmed().isEmpty())
    {
        return fail(QStringLiteral("空三输出目录为空"), errorMessage);
    }
    if (!QDir().mkpath(input.outputDir))
    {
        return fail(QStringLiteral("无法创建空三输出目录: %1").arg(input.outputDir), errorMessage);
    }

    // 先在内存中完成点过滤和颜色采样，再开始任何正式文件提交。
    std::vector<ExportPoint> points = collectExportPoints(*execution->reconstruction);
    samplePointColors(*execution->reconstruction, &points);
    const QString plyPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse.ply"));
    if (!writeBinaryPly(plyPath, points, errorMessage))
    {
        return false;
    }

    // 质量 sidecar 与 PLY 基于同一最终 reconstruction，避免候选试算指标混入。
    const SparseQualityReport report = QualityReportWriter::build(
        input, *execution->reconstruction, execution->result);
    const QString sidecarPath = QDir(input.outputDir)
        .filePath(QStringLiteral("sfm_sparse_points.json"));
    QJsonObject sidecar = xjw::common::project::mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("points"), report.points},
                    {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")}},
        report.qualityMetadata);
    sidecar.insert(QStringLiteral("sfm_diagnostics"), report.diagnostics);
    QString writeError;
    if (!xjw::common::io::writeFileBytesAtomic(
            sidecarPath,
            // 稀疏点逐观测 sidecar 可能达到数十 MiB；消费者均按 JSON 解析，
            // 无需在关键路径为人工缩进额外分配和写盘。
            QJsonDocument(sidecar).toJson(QJsonDocument::Compact),
            &writeError))
    {
        return fail(writeError.isEmpty()
                        ? QStringLiteral("无法写入稀疏点云质量文件: %1").arg(sidecarPath)
                        : writeError,
                    errorMessage);
    }

    // 到达此处表示两个正式文件均成功，随后才更新返回结果记录。
    execution->result.sparseCloudPath = plyPath;
    execution->result.qualityMetadata = report.qualityMetadata;
    execution->result.sfmDiagnostics = report.diagnostics;
    execution->result.perCameraResiduals = report.perCameraResiduals;
    const QJsonObject files{{QStringLiteral("sparse_cloud_points_json"), sidecarPath}};
    execution->result.resultRecordExtra = xjw::common::project::mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("files"), files},
                    {QStringLiteral("source"), QStringLiteral("workflow_aerial_triangulation")},
                    {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")}},
        report.qualityMetadata);
    execution->result.resultRecordExtra.insert(QStringLiteral("sfm_diagnostics"),
                                               report.diagnostics);
    execution->result.summary = QStringLiteral("SFM 重建成功：注册 %1/%2 张影像，%3 个三维点")
        .arg(execution->result.numRegisteredImages)
        .arg(input.images.size())
        .arg(execution->result.numPoints3D);
    return true;
}

} // namespace xjw::aerial_triangulation

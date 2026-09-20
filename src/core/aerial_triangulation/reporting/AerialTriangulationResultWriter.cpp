/**
 * @file AerialTriangulationResultWriter.cpp
 * @brief 胜出 SfM 模型的稀疏点云、质量 sidecar 和工程记录构建实现。
 *
 * PLY 使用标准文件模块原子提交；质量 JSON 使用通用原子 IO。相机更新仍保留在
 * cameraInstanceUpdates，由工程服务在本函数完全成功后统一应用。
 */

#include "reporting/AerialTriangulationResultWriter.h"

#include "io/ImageIO.h"
#include "io/PathIO.h"
#include "log/Logger.h"

#include "project/SparseResultQuality.h"
#include "reconstruction/SfmReconstruction.h"
#include "reporting/QualityReportWriter.h"

#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include "reporting/SparsePlyWriter.h"
#include "file/FileIO.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace xjw::aerial_triangulation
{
    namespace
    {

        struct ExportPoint
        {
            std::array<float, 3> xyz{};                        ///< 最终 BA 坐标系中的三维坐标。
            std::array<quint8, 3> color{{128, 128, 128}};      ///< 无可读影像时使用中性灰。
            ImageId colorImageId = kInvalidImageId;            ///< 采样颜色所用的首个有效轨迹观测。
            FeatureIdx colorFeatureIndex = kInvalidFeatureIdx; ///< 对应影像内关键点索引。
        };

        bool fail(const QString& message, QString* errorMessage)
        {
            if (errorMessage)
            {
                *errorMessage = message;
            }
            return false;
        }

        bool validateCanonicalOutputIdentity(const PreparedAerialTriangulationInput& input,
                                             const SfmReconstruction& reconstruction,
                                             QString* errorMessage)
        {
            if (input.images.size() != static_cast<qsizetype>(input.imageIds.size()))
            {
                return fail(QStringLiteral("SfM 正式 sidecar 写出要求 images 与 canonical ImageId 一一对应"),
                            errorMessage);
            }

            std::unordered_set<std::string> imageIds;
            imageIds.reserve(input.imageIds.size());
            for (const camera_core::ImageId& imageId : input.imageIds)
            {
                if (imageId.value().empty() || !imageIds.insert(imageId.value()).second)
                {
                    return fail(QStringLiteral("SfM 正式 sidecar 写出发现空或重复的 canonical ImageId"), errorMessage);
                }
            }

            for (const ImageId imageIndex : reconstruction.registeredImageIds())
            {
                if (imageIndex >= input.imageIds.size() || !reconstruction.hasCamera(imageIndex))
                {
                    return fail(QStringLiteral("SfM 注册相机超出 canonical ImageId 输入范围"), errorMessage);
                }
                const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera =
                    reconstruction.camera(imageIndex);
                if (!camera.hasBoundIdentity() || camera.imageId() != input.imageIds.at(imageIndex))
                {
                    return fail(QStringLiteral("SfM 注册相机缺少与输入一致的 canonical ImageId，拒绝写出 sidecar"),
                                errorMessage);
                }
            }
            return true;
        }

        /**
         * @brief 按质量报告确认的点 ID 从最终重建收集可发布点。
         *
         * 发布门槛由 QualityReportWriter 统一执行；这里保持其顺序并补充颜色采样索引，
         * 不修改 reconstruction，也不重新三角化。
         */
        std::vector<ExportPoint> collectExportPoints(const SfmReconstruction& reconstruction,
                                                     const std::vector<Point3DId>& publishedPointIds)
        {
            std::vector<ExportPoint> points;
            points.reserve(publishedPointIds.size());
            for (const Point3DId pointId : publishedPointIds)
            {
                if (!reconstruction.hasPoint3D(pointId))
                {
                    continue;
                }
                const ScenePoint3D& point = reconstruction.point3D(pointId);
                ExportPoint output;
                output.xyz = {static_cast<float>(point.xyz[0]),
                              static_cast<float>(point.xyz[1]),
                              static_cast<float>(point.xyz[2])};
                for (const TrackElement& element : point.track.elements)
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
        void samplePointColors(const SfmReconstruction& reconstruction,
                               std::vector<ExportPoint>* points,
                               std::vector<ExportPoint>* secondaryPoints)
        {
            if (!points && !secondaryPoints)
            {
                return;
            }
            struct ColorRequest
            {
                std::vector<ExportPoint>* points = nullptr;
                std::size_t index = 0;
            };
            QMap<ImageId, std::vector<ColorRequest>> requestsByImage;
            const auto appendRequests = [&requestsByImage](std::vector<ExportPoint>* target)
            {
                if (!target)
                {
                    return;
                }
                for (std::size_t index = 0; index < target->size(); ++index)
                {
                    if ((*target)[index].colorImageId != kInvalidImageId)
                    {
                        requestsByImage[(*target)[index].colorImageId].push_back({target, index});
                    }
                }
            };
            appendRequests(points);
            appendRequests(secondaryPoints);

            for (auto request = requestsByImage.cbegin(); request != requestsByImage.cend(); ++request)
            {
                if (!reconstruction.hasImage(request.key()))
                {
                    continue;
                }
                const ImageData& imageData = reconstruction.image(request.key());
                cv::Mat image;
                QString read_error;
                try
                {
                    image = xjw::common::io::readImage(QString::fromStdString(imageData.imagePath),
                                                       cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION,
                                                       &read_error);
                }
                catch (const cv::Exception& exception)
                {
                    read_error = QString::fromUtf8(exception.what());
                }
                if (image.empty() || image.type() != CV_8UC3)
                {
                    if (read_error.isEmpty())
                    {
                        read_error = QStringLiteral("预期非空 8 位三通道影像");
                    }
                    LOG_WARN(QStringLiteral("空三点云颜色读取失败 %1: %2")
                                 .arg(QString::fromStdString(imageData.imagePath), read_error));
                    continue;
                }
                for (const ColorRequest& colorRequest : request.value())
                {
                    ExportPoint& point = (*colorRequest.points)[colorRequest.index];
                    if (point.colorFeatureIndex >= imageData.keypoints.size())
                    {
                        continue;
                    }
                    const FeatureKeypoint& keypoint = imageData.keypoints[point.colorFeatureIndex];
                    const int x = std::clamp(qRound(keypoint.x), 0, image.cols - 1);
                    const int y = std::clamp(qRound(keypoint.y), 0, image.rows - 1);
                    const cv::Vec3b color = image.at<cv::Vec3b>(y, x);
                    point.color = {color[2], color[1], color[0]};
                }
            }
        }

        /// 原子写入 little-endian binary PLY，失败时旧文件保持不变。
        bool writeBinaryPly(const QString& path, const std::vector<ExportPoint>& points, QString* errorMessage)
        {
            std::string error;
            const bool ok = writeSparsePly(
                xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(path)),
                points.size(),
                [&points](std::size_t index) { return SparsePlyVertex{points[index].xyz, points[index].color}; },
                "PlaScan aerial triangulation",
                &error);
            if (errorMessage)
            {
                *errorMessage = QString::fromStdString(error);
            }
            return ok;
        }

    } // namespace

    bool AerialTriangulationResultWriter::write(const PreparedAerialTriangulationInput& input,
                                                SfmAttemptExecutionResult* execution,
                                                QString* errorMessage) const
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
        if (!validateCanonicalOutputIdentity(input, *execution->reconstruction, errorMessage))
        {
            return false;
        }
        std::string directory_error;
        if (!xjw::common::file::ensureDirectory(
                xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(input.outputDir)), &directory_error))
        {
            return fail(QString::fromStdString(directory_error), errorMessage);
        }

        // 主 PLY 和 sidecar 保存全部算法有效点；清理策略只生成独立显示云，不能再覆盖
        // BA/SfM 的正式结构结果。
        const SparseQualityReport report =
            QualityReportWriter::build(input, *execution->reconstruction, execution->result);
        std::vector<ExportPoint> points = collectExportPoints(*execution->reconstruction, report.publishedPointIds);
        if (points.empty() || points.size() != static_cast<std::size_t>(report.points.size()))
        {
            return fail(QStringLiteral("空三结果没有通过最终多指标质量门禁的可发布连接点"), errorMessage);
        }
        const QString plyPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse.ply"));

        std::vector<ExportPoint> displayPoints =
            collectExportPoints(*execution->reconstruction, report.displayPointIds);
        samplePointColors(*execution->reconstruction, &points, &displayPoints);
        if (!writeBinaryPly(plyPath, points, errorMessage))
        {
            return false;
        }
        QString displayPlyPath;
        if (!displayPoints.empty())
        {
            displayPlyPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse_display.ply"));
            if (!writeBinaryPly(displayPlyPath, displayPoints, errorMessage))
            {
                return false;
            }
        }

        // 质量 sidecar 与 PLY 基于同一最终 reconstruction，避免候选试算指标混入。
        const QString sidecarPath = QDir(input.outputDir).filePath(QStringLiteral("sfm_sparse_points.json"));
        QJsonObject sidecar = xjw::common::project::mergeSparseQualityIntoRecord(
            QJsonObject{{QStringLiteral("points"), report.points},
                        {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")}},
            report.qualityMetadata);
        QJsonArray images;
        for (int imageId = 0; imageId < input.images.size(); ++imageId)
        {
            const QString& imagePath = input.images.at(imageId);
            images.append(QJsonObject{
                {QStringLiteral("camera_index"), imageId},
                {QStringLiteral("image_id"), QString::fromStdString(input.imageIds.at(imageId).value())},
                {QStringLiteral("image_path"), imagePath},
                {QStringLiteral("image_name"), QFileInfo(imagePath).fileName()},
            });
        }
        sidecar.insert(QStringLiteral("schema"), QStringLiteral("plascan.sfm_sparse_points.v3"));
        sidecar.insert(QStringLiteral("observation_fields"),
                       QJsonArray{
                           QStringLiteral("camera_index"),
                           QStringLiteral("image_id"),
                           QStringLiteral("feature_idx"),
                           QStringLiteral("xy"),
                           QStringLiteral("scale"),
                           QStringLiteral("projected_xy"),
                       });
        sidecar.insert(QStringLiteral("images"), images);
        sidecar.insert(QStringLiteral("clean_tie_points_metric_contract"),
                       QStringLiteral("metashape-2.3.2-build-22956"));
        sidecar.insert(QStringLiteral("sfm_diagnostics"), report.diagnostics);
        const QByteArray sidecar_bytes = QJsonDocument(sidecar).toJson(QJsonDocument::Compact);
        std::string file_error;
        const bool sidecar_written = xjw::common::file::writeFileAtomic(
            xjw::common::file::pathFromUtf8(xjw::common::io::toUtf8Path(sidecarPath)),
            std::string_view(sidecar_bytes.constData(), static_cast<std::size_t>(sidecar_bytes.size())),
            &file_error);
        const QString writeError = QString::fromStdString(file_error);
        if (!sidecar_written)
        {
            return fail(writeError.isEmpty() ? QStringLiteral("无法写入稀疏点云质量文件: %1").arg(sidecarPath)
                                             : writeError,
                        errorMessage);
        }

        // 到达此处表示两个正式文件均成功，随后才更新返回结果记录。
        execution->result.sparseCloudPath = plyPath;
        execution->result.displaySparseCloudPath = displayPlyPath;
        execution->result.numPoints3D = static_cast<int>(points.size());
        execution->result.qualityMetadata = report.qualityMetadata;
        execution->result.sfmDiagnostics = report.diagnostics;
        execution->result.perCameraResiduals = report.perCameraResiduals;
        QJsonObject files{{QStringLiteral("sparse_cloud_points_json"), sidecarPath}};
        if (!displayPlyPath.isEmpty())
        {
            files.insert(QStringLiteral("sparse_cloud_display_xyz"), displayPlyPath);
        }
        execution->result.resultRecordExtra = xjw::common::project::mergeSparseQualityIntoRecord(
            QJsonObject{{QStringLiteral("files"), files},
                        {QStringLiteral("source"), QStringLiteral("workflow_aerial_triangulation")},
                        {QStringLiteral("operation"), QStringLiteral("workflow_aerial_triangulation")}},
            report.qualityMetadata);
        execution->result.resultRecordExtra.insert(QStringLiteral("sfm_diagnostics"), report.diagnostics);
        execution->result.summary = QStringLiteral("SFM 重建成功：注册 %1/%2 张影像，%3 个三维点")
                                        .arg(execution->result.numRegisteredImages)
                                        .arg(input.images.size())
                                        .arg(execution->result.numPoints3D);
        return true;
    }

} // namespace xjw::aerial_triangulation

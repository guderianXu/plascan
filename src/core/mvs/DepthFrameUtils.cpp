#include "DepthFrameUtils.h"

#include "DepthFrameQualificationPolicy.h"
#include "depth_processing/DepthPostprocessor.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QRegularExpression>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace xjw::core::project
{

    namespace
    {

        QString firstExistingPath(const QStringList& paths)
        {
            for (const QString& path : paths)
            {
                if (!path.trimmed().isEmpty() && QFileInfo::exists(path))
                {
                    return path;
                }
            }
            return QString();
        }

        bool
        readPositiveSize(const QJsonObject& object, const QString& widthKey, const QString& heightKey, cv::Size* size)
        {
            if (!size || !object.contains(widthKey) || !object.contains(heightKey))
            {
                return false;
            }

            const int width = object.value(widthKey).toInt(0);
            const int height = object.value(heightKey).toInt(0);
            if (width <= 0 || height <= 0)
            {
                return false;
            }
            *size = cv::Size(width, height);
            return true;
        }

        bool readPreparedPngSize(const QString& path, cv::Size* size)
        {
            if (!size || path.trimmed().isEmpty())
            {
                return false;
            }
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly))
            {
                return false;
            }
            const QByteArray header = file.read(24);
            constexpr std::array<unsigned char, 8> signature{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
            constexpr std::array<unsigned char, 4> ihdr_length{0, 0, 0, 13};
            if (header.size() != 24 || std::memcmp(header.constData(), signature.data(), signature.size()) != 0 ||
                std::memcmp(header.constData() + 8, ihdr_length.data(), ihdr_length.size()) != 0 ||
                std::memcmp(header.constData() + 12, "IHDR", 4) != 0)
            {
                return false;
            }
            const auto read_dimension = [&header](int offset)
            {
                const auto* bytes = reinterpret_cast<const unsigned char*>(header.constData() + offset);
                return (static_cast<quint32>(bytes[0]) << 24) | (static_cast<quint32>(bytes[1]) << 16) |
                       (static_cast<quint32>(bytes[2]) << 8) | static_cast<quint32>(bytes[3]);
            };
            const quint32 width = read_dimension(16);
            const quint32 height = read_dimension(20);
            if (width == 0 || height == 0 || width > static_cast<quint32>(std::numeric_limits<int>::max()) ||
                height > static_cast<quint32>(std::numeric_limits<int>::max()))
            {
                return false;
            }
            *size = cv::Size(static_cast<int>(width), static_cast<int>(height));
            return true;
        }

        template <std::size_t Size>
        bool readFiniteArray(const QJsonValue& value, std::array<double, Size>* output)
        {
            if (!output || !value.isArray())
            {
                return false;
            }
            const QJsonArray array = value.toArray();
            if (array.size() != static_cast<qsizetype>(Size))
            {
                return false;
            }
            for (std::size_t index = 0; index < Size; ++index)
            {
                const QJsonValue element = array.at(static_cast<qsizetype>(index));
                if (!element.isDouble() || !std::isfinite(element.toDouble()))
                {
                    return false;
                }
                (*output)[index] = element.toDouble();
            }
            return true;
        }

        std::shared_ptr<const placamera::FramePinholeModel>
        depthCameraFromJson(const QJsonObject& object, const cv::Size& depthSize)
        {
            if (depthSize.width <= 0 || depthSize.height <= 0)
            {
                return {};
            }
            std::array<double, 9> world_to_camera{};
            std::array<double, 3> center{};
            std::array<double, 3> translation{};
            if (!readFiniteArray(object.value(QStringLiteral("rotation_world_to_camera")), &world_to_camera) ||
                !readFiniteArray(object.value(QStringLiteral("translation_world_to_camera")), &translation) ||
                !readFiniteArray(object.value(QStringLiteral("camera_center")), &center))
            {
                return {};
            }

            const QString instance_id = object.value(QStringLiteral("instance_id")).toString().trimmed();
            const QString image_id = object.value(QStringLiteral("image_id")).toString().trimmed();
            const QString frame_id = object.value(QStringLiteral("world_frame")).toString().trimmed();
            const QJsonValue fx_value = object.value(QStringLiteral("fx"));
            const QJsonValue fy_value = object.value(QStringLiteral("fy"));
            const QJsonValue cx_value = object.value(QStringLiteral("cx"));
            const QJsonValue cy_value = object.value(QStringLiteral("cy"));
            if (!fx_value.isDouble() || !fy_value.isDouble() || !cx_value.isDouble() || !cy_value.isDouble() ||
                instance_id.isEmpty() || image_id.isEmpty() || frame_id.isEmpty())
            {
                return {};
            }

            const std::array<double, 9> camera_to_world{world_to_camera[0],
                                                         world_to_camera[3],
                                                         world_to_camera[6],
                                                         world_to_camera[1],
                                                         world_to_camera[4],
                                                         world_to_camera[7],
                                                         world_to_camera[2],
                                                         world_to_camera[5],
                                                         world_to_camera[8]};
            for (int row = 0; row < 3; ++row)
            {
                const double expected_translation =
                    -(world_to_camera[static_cast<std::size_t>(row * 3)] * center[0] +
                      world_to_camera[static_cast<std::size_t>(row * 3 + 1)] * center[1] +
                      world_to_camera[static_cast<std::size_t>(row * 3 + 2)] * center[2]);
                const double tolerance = 1.0e-8 * std::max({1.0, std::abs(expected_translation),
                                                              std::abs(translation[static_cast<std::size_t>(row)])});
                if (std::abs(expected_translation - translation[static_cast<std::size_t>(row)]) > tolerance)
                {
                    return {};
                }
            }

            try
            {
                const placamera::FrameId ground_frame(frame_id.toStdString());
                placamera::FrameIntrinsics intrinsics;
                intrinsics.focalX = fx_value.toDouble();
                intrinsics.focalY = fy_value.toDouble();
                intrinsics.principalX = cx_value.toDouble();
                intrinsics.principalY = cy_value.toDouble();
                const auto definition = placamera::FramePinholeDefinition::create(
                    placamera::CameraDefinitionId(instance_id.toStdString() + "-depth-definition"),
                    intrinsics,
                    {},
                    placamera::PixelConvention::PixelCenter,
                    ground_frame);
                return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
                    placamera::CameraInstanceId(instance_id.toStdString()),
                    placamera::ImageId(image_id.toStdString()),
                    definition,
                    {depthSize.width, depthSize.height},
                    placamera::Pose::create(ground_frame, center, camera_to_world)));
            }
            catch (const std::exception&)
            {
                return {};
            }
        }

        std::shared_ptr<const placamera::FramePinholeModel>
        scaledDepthCamera(const placamera::FramePinholeModel& source, const cv::Size& targetSize)
        {
            const auto& source_size = source.imageSize();
            const auto definition = source.pinholeDefinition().scaledIntrinsics(
                placamera::CameraDefinitionId(source.definitionId().value() + "-scaled-" +
                                              std::to_string(targetSize.width) + "x" +
                                              std::to_string(targetSize.height)),
                static_cast<double>(targetSize.width) / source_size.samples,
                static_cast<double>(targetSize.height) / source_size.lines);
            return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
                source.instanceId(),
                source.imageId(),
                definition,
                {targetSize.width, targetSize.height},
                source.pose(),
                source.captureTime()));
        }

        bool validateNativePixelDomainContract(const StoredDepthFrameRecord& stored,
                                               const cv::Size& rawDepthSize,
                                               cv::Size* preparedRasterSize,
                                               QString* errorMessage)
        {
            if (!stored.effectiveNativeFinalDepthGrid)
            {
                return true;
            }

            cv::Size diagnostic_raster_size;
            cv::Size diagnostic_grid_size;
            const bool has_raster_size = readPositiveSize(stored.pixelDomainDiagnostics,
                                                          QStringLiteral("raster_width"),
                                                          QStringLiteral("raster_height"),
                                                          &diagnostic_raster_size);
            const bool has_grid_size = readPositiveSize(stored.pixelDomainDiagnostics,
                                                        QStringLiteral("grid_width"),
                                                        QStringLiteral("grid_height"),
                                                        &diagnostic_grid_size);
            const bool diagnostic_effective =
                stored.pixelDomainDiagnostics.value(QStringLiteral("effective_native_final_depth_grid")).toBool(false);
            if (!has_raster_size || !has_grid_size || !diagnostic_effective)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("原生深度网格记录缺少完整的 pixel_domain_diagnostics，"
                                                   "无法恢复 prepared raster 像素域");
                }
                return false;
            }

            const cv::Size stored_grid_size(stored.gridWidth, stored.gridHeight);
            if (stored_grid_size.width <= 0 || stored_grid_size.height <= 0 ||
                diagnostic_grid_size != stored_grid_size || diagnostic_grid_size != rawDepthSize ||
                diagnostic_raster_size.width < diagnostic_grid_size.width ||
                diagnostic_raster_size.height < diagnostic_grid_size.height)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("原生深度网格记录的 raster/grid 尺寸互相矛盾："
                                                   "diagnostic_raster=%1x%2 diagnostic_grid=%3x%4 "
                                                   "stored_grid=%5x%6 raw_depth=%7x%8")
                                        .arg(diagnostic_raster_size.width)
                                        .arg(diagnostic_raster_size.height)
                                        .arg(diagnostic_grid_size.width)
                                        .arg(diagnostic_grid_size.height)
                                        .arg(stored.gridWidth)
                                        .arg(stored.gridHeight)
                                        .arg(rawDepthSize.width)
                                        .arg(rawDepthSize.height);
                }
                return false;
            }

            if (preparedRasterSize)
            {
                *preparedRasterSize = diagnostic_raster_size;
            }
            return true;
        }

        QString resolveExistingRawDepthPath(const QString& pngPath, const QString& preferredPath = QString())
        {
            return firstExistingPath({preferredPath, rawDepthStoragePath(pngPath)});
        }

        QString resolveExistingRawConfidencePath(const QString& pngPath, const QString& preferredPath = QString())
        {
            return firstExistingPath({preferredPath, rawConfidenceStoragePath(pngPath)});
        }

        QString resolveStoredArtifactPath(const QString& rawDepthPath,
                                          const QString& artifactPath,
                                          const QString& fallbackPath = QString())
        {
            const QString normalized_path = QDir::fromNativeSeparators(artifactPath.trimmed());
            if (normalized_path.isEmpty())
            {
                return firstExistingPath({fallbackPath});
            }

            const QFileInfo artifact_info(normalized_path);
            if (artifact_info.isAbsolute())
            {
                const QString existing_path = firstExistingPath({artifact_info.absoluteFilePath(), fallbackPath});
                return existing_path.isEmpty() ? QDir::cleanPath(artifact_info.absoluteFilePath())
                                               : QDir::cleanPath(existing_path);
            }

            const QDir raw_depth_directory = QFileInfo(rawDepthPath).absoluteDir();
            const QString relative_candidate = raw_depth_directory.absoluteFilePath(normalized_path);
            const QString filename_candidate = raw_depth_directory.filePath(QFileInfo(normalized_path).fileName());
            const QString existing_path =
                firstExistingPath({normalized_path, relative_candidate, filename_candidate, fallbackPath});
            return existing_path.isEmpty() ? QDir::cleanPath(relative_candidate) : QDir::cleanPath(existing_path);
        }

        xjw::common::OperationResult loadStoredEvidenceMap(const QString& path,
                                                           const QString& label,
                                                           int expectedType,
                                                           cv::Size expectedSize,
                                                           bool required,
                                                           cv::Mat* matrix)
        {
            if (!matrix)
            {
                return {false, QStringLiteral("内部错误：%1输出参数无效").arg(label)};
            }
            matrix->release();
            if (path.isEmpty())
            {
                return required ? xjw::common::OperationResult{false,
                                                               QStringLiteral("缓存深度帧缺少%1，请重新计算深度图")
                                                                   .arg(label)}
                                : xjw::common::OperationResult{true, QString()};
            }

            xjw::common::OperationResult status = loadDepthMatStorage(path, matrix);
            if (!status.ok)
            {
                status.errorMessage = QStringLiteral("读取%1失败：%2").arg(label, status.errorMessage);
                return status;
            }
            if (matrix->type() != expectedType || matrix->size() != expectedSize)
            {
                const QString error =
                    QStringLiteral(
                        "%1类型或尺寸与深度图不匹配：path=%2 type=%3 size=%4x%5 expected_type=%6 expected_size=%7x%8")
                        .arg(label,
                             path,
                             QString::number(matrix->type()),
                             QString::number(matrix->cols),
                             QString::number(matrix->rows),
                             QString::number(expectedType),
                             QString::number(expectedSize.width),
                             QString::number(expectedSize.height));
                matrix->release();
                return {false, error};
            }
            return {true, QString()};
        }

        void resizeEvidenceMap(cv::Mat* matrix, cv::Size targetSize)
        {
            if (!matrix || matrix->empty() || matrix->size() == targetSize)
            {
                return;
            }

            cv::Mat resized;
            cv::resize(*matrix, resized, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
            *matrix = std::move(resized);
        }

        int frameIndexFromPath(const QString& path)
        {
            const QRegularExpression re(QStringLiteral("(\\d+)(?!.*\\d)"));
            const QRegularExpressionMatch match = re.match(QFileInfo(path).completeBaseName());
            return match.hasMatch() ? match.captured(1).toInt() : std::numeric_limits<int>::max();
        }

        QString normalizedDirectoryPath(const QString& path)
        {
            const QFileInfo path_info(path);
            const QString directory_path = path_info.isDir() ? path_info.absoluteFilePath() : path_info.absolutePath();
            return QDir::cleanPath(directory_path);
        }

        StoredDepthFramesResult collectStoredDepthFramesInDirectory(const QJsonArray& depth_results,
                                                                    const QString& batch_directory)
        {
            StoredDepthFramesResult result;
            const QString normalized_batch_directory = normalizedDirectoryPath(batch_directory);
            if (normalized_batch_directory.isEmpty())
            {
                result.status = {false, QStringLiteral("深度图批次目录为空")};
                return result;
            }

            for (const QJsonValue& value : depth_results)
            {
                const QJsonObject record = value.toObject();
                const QString depth_png = record.value(QStringLiteral("depth_png")).toString();
                const QString raw_depth_path =
                    resolveExistingRawDepthPath(depth_png, record.value(QStringLiteral("raw_depth_path")).toString());
                if (raw_depth_path.isEmpty() ||
                    normalizedDirectoryPath(raw_depth_path).compare(normalized_batch_directory, Qt::CaseInsensitive) !=
                        0)
                {
                    continue;
                }

                StoredDepthFrameRecord frame;
                frame.refIndex = record.value(QStringLiteral("ref_index")).toInt(-1);
                frame.refImage = record.value(QStringLiteral("ref_image")).toString();
                frame.preparedImage = record.value(QStringLiteral("prepared_image")).toString();
                frame.preparedValidMaskPath = record.value(QStringLiteral("prepared_valid_mask_path")).toString();
                frame.preparedCameraModel = record.value(QStringLiteral("prepared_camera_model")).toObject();
                frame.depthPng = depth_png;
                frame.rawDepthPath = raw_depth_path;
                frame.rawConfidencePath = resolveExistingRawConfidencePath(
                    frame.depthPng, record.value(QStringLiteral("raw_confidence_path")).toString());
                frame.rawGeometrySupportPath =
                    resolveStoredArtifactPath(frame.rawDepthPath,
                                              record.value(QStringLiteral("raw_geometry_support_path")).toString(),
                                              rawGeometrySupportStoragePath(frame.depthPng));
                frame.rawInverseDepthSpreadPath =
                    resolveStoredArtifactPath(frame.rawDepthPath,
                                              record.value(QStringLiteral("raw_inverse_depth_spread_path")).toString(),
                                              rawInverseDepthSpreadStoragePath(frame.depthPng));
                frame.rawAdaptiveGeometrySupportWeightPath = resolveStoredArtifactPath(
                    frame.rawDepthPath,
                    record.value(QStringLiteral("raw_adaptive_geometry_support_weight_path")).toString(),
                    rawAdaptiveGeometrySupportWeightStoragePath(frame.depthPng));
                frame.rawAdaptiveGeometryEffectiveViewCountPath = resolveStoredArtifactPath(
                    frame.rawDepthPath,
                    record.value(QStringLiteral("raw_adaptive_geometry_effective_view_count_path")).toString(),
                    rawAdaptiveGeometryEffectiveViewCountStoragePath(frame.depthPng));
                frame.rawAdaptiveGeometryConflictRatioPath = resolveStoredArtifactPath(
                    frame.rawDepthPath,
                    record.value(QStringLiteral("raw_adaptive_geometry_conflict_ratio_path")).toString(),
                    rawAdaptiveGeometryConflictRatioStoragePath(frame.depthPng));
                const QJsonArray source_images = record.value(QStringLiteral("source_images")).toArray();
                for (const QJsonValue& source_image : source_images)
                {
                    const QString path = source_image.toString().trimmed();
                    if (!path.isEmpty())
                    {
                        frame.sourceImages.push_back(QDir::cleanPath(path));
                    }
                }
                frame.device = record.value(QStringLiteral("device")).toString();
                frame.configHash = record.value(QStringLiteral("config_hash")).toString();
                frame.algorithmRevision = record.value(QStringLiteral("algorithm_revision")).toInt(0);
                frame.projectInputSignature = record.value(QStringLiteral("project_input_signature")).toString();
                frame.reconstructionGenerationId =
                    record.value(QStringLiteral("reconstruction_generation_id")).toString();
                frame.cameraModel = record.value(QStringLiteral("camera_model")).toObject();
                frame.sceneProfile = record.value(QStringLiteral("scene_profile")).toString();
                frame.status = record.value(QStringLiteral("status")).toString();
                const xjw::mvs::MvsDepthFrameQualification qualification =
                    xjw::mvs::qualifyMvsDepthFrameArtifact(record);
                frame.acceptance = qualification.acceptance;
                frame.fusionEligibilityKnown = qualification.fusionEligibilityKnown;
                frame.fusionEligible = qualification.fusionEligible;
                frame.role = qualification.role;
                frame.useDiscreteGeometryFallback = qualification.useDiscreteGeometryFallback;
                frame.effectiveNativeFinalDepthGrid =
                    record.value(QStringLiteral("effective_native_final_depth_grid")).toBool(false);
                frame.pixelDomainDiagnostics = record.value(QStringLiteral("pixel_domain_diagnostics")).toObject();
                frame.gridWidth = record.value(QStringLiteral("grid_width")).toInt();
                frame.gridHeight = record.value(QStringLiteral("grid_height")).toInt();
                if (!frame.refImage.isEmpty() && depthFrameArtifactsExist(frame))
                {
                    result.frames.push_back(std::move(frame));
                }
            }

            std::sort(result.frames.begin(),
                      result.frames.end(),
                      [](const StoredDepthFrameRecord& lhs, const StoredDepthFrameRecord& rhs)
                      { return frameIndexFromPath(lhs.rawDepthPath) < frameIndexFromPath(rhs.rawDepthPath); });
            result.batchDir = normalized_batch_directory;
            if (result.frames.empty())
            {
                result.status = {
                    false, QStringLiteral("所选目录不包含可复用的原始深度图：%1").arg(normalized_batch_directory)};
                return result;
            }

            result.status = {true, QString()};
            return result;
        }

    } // namespace

    std::uint64_t estimateFusionFrameWorkingSetBytes(int width, int height, int fusionMaxImageDim)
    {
        if (width <= 0 || height <= 0)
        {
            return 64ULL * 1024ULL * 1024ULL;
        }

        const std::uint64_t source_pixels = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
        std::uint64_t target_pixels = source_pixels;
        if (fusionMaxImageDim > 0 && std::max(width, height) > fusionMaxImageDim)
        {
            const double scale = static_cast<double>(fusionMaxImageDim) / static_cast<double>(std::max(width, height));
            const int target_width = std::max(1, static_cast<int>(std::lround(width * scale)));
            const int target_height = std::max(1, static_cast<int>(std::lround(height * scale)));
            target_pixels = static_cast<std::uint64_t>(target_width) * static_cast<std::uint64_t>(target_height);
        }

        // Depth, confidence, discrete geometry support, inverse-depth spread and the
        // optional three-map adaptive evidence bundle can coexist during resize.
        const std::uint64_t load_peak = source_pixels * 28ULL + target_pixels * 28ULL;
        const std::uint64_t postprocess_peak = target_pixels * 36ULL;
        return std::max(load_peak, postprocess_peak) + 16ULL * 1024ULL * 1024ULL;
    }

    int recommendedDepthFrameLoadWorkers(int requestedWorkers,
                                         std::uint64_t availableMemoryBytes,
                                         std::uint64_t frameWorkingSetBytes)
    {
        const int requested = std::clamp(requestedWorkers > 0 ? requestedWorkers : 4, 1, 4);
        if (availableMemoryBytes == 0 || frameWorkingSetBytes == 0)
        {
            return std::clamp(requested, 2, 4);
        }

        const std::uint64_t loading_budget = availableMemoryBytes / 2ULL;
        const std::uint64_t memory_workers = loading_budget / frameWorkingSetBytes;
        return static_cast<int>(
            std::clamp<std::uint64_t>(std::min<std::uint64_t>(requested, memory_workers), 1ULL, 4ULL));
    }

    QString rawDepthStoragePath(const QString& pngPath)
    {
        const QFileInfo info(pngPath);
        return info.dir().filePath(info.completeBaseName() + QStringLiteral(".bin"));
    }

    QString rawConfidenceStoragePath(const QString& pngPath)
    {
        const QFileInfo info(pngPath);
        return info.dir().filePath(info.completeBaseName() + QStringLiteral("_conf.bin"));
    }

    QString rawGeometrySupportStoragePath(const QString& pngPath)
    {
        const QFileInfo info(pngPath);
        return info.dir().filePath(info.completeBaseName() + QStringLiteral("_geometry_support.bin"));
    }

    QString rawInverseDepthSpreadStoragePath(const QString& pngPath)
    {
        const QFileInfo info(pngPath);
        return info.dir().filePath(info.completeBaseName() + QStringLiteral("_inverse_depth_spread.bin"));
    }

    QString rawAdaptiveGeometrySupportWeightStoragePath(const QString& pngPath)
    {
        const QFileInfo info(pngPath);
        return info.dir().filePath(info.completeBaseName() + QStringLiteral("_adaptive_geometry_support_weight.bin"));
    }

    QString rawAdaptiveGeometryEffectiveViewCountStoragePath(const QString& pngPath)
    {
        const QFileInfo info(pngPath);
        return info.dir().filePath(info.completeBaseName() +
                                   QStringLiteral("_adaptive_geometry_effective_view_count.bin"));
    }

    QString rawAdaptiveGeometryConflictRatioStoragePath(const QString& pngPath)
    {
        const QFileInfo info(pngPath);
        return info.dir().filePath(info.completeBaseName() + QStringLiteral("_adaptive_geometry_conflict_ratio.bin"));
    }

    bool depthFrameArtifactsExist(const QString& pngPath, bool requireConfidence)
    {
        if (pngPath.trimmed().isEmpty() || !QFileInfo::exists(pngPath))
        {
            return false;
        }

        const QString rawDepthPath = resolveExistingRawDepthPath(pngPath);
        if (rawDepthPath.isEmpty())
        {
            return false;
        }

        if (requireConfidence && resolveExistingRawConfidencePath(pngPath).isEmpty())
        {
            return false;
        }

        return true;
    }

    bool depthFrameArtifactsExist(const StoredDepthFrameRecord& frame, bool requireConfidence)
    {
        if (frame.depthPng.trimmed().isEmpty() || !QFileInfo::exists(frame.depthPng))
        {
            return false;
        }

        if (frame.rawDepthPath.trimmed().isEmpty() || !QFileInfo::exists(frame.rawDepthPath))
        {
            const QString fallbackRawDepthPath = resolveExistingRawDepthPath(frame.depthPng, frame.rawDepthPath);
            if (fallbackRawDepthPath.isEmpty())
            {
                return false;
            }
        }

        if (requireConfidence &&
            (frame.rawConfidencePath.trimmed().isEmpty() || !QFileInfo::exists(frame.rawConfidencePath)))
        {
            const QString fallbackConfidencePath =
                resolveExistingRawConfidencePath(frame.depthPng, frame.rawConfidencePath);
            if (fallbackConfidencePath.isEmpty())
            {
                return false;
            }
        }

        if (frame.algorithmRevision >= xjw::mvs::kMvsPreparedRasterProvenanceRevision)
        {
            cv::Size prepared_size;
            cv::Size mask_size;
            if (!readPreparedPngSize(frame.preparedImage, &prepared_size) ||
                !readPreparedPngSize(frame.preparedValidMaskPath, &mask_size) || prepared_size != mask_size ||
                !depthCameraFromJson(frame.preparedCameraModel, prepared_size))
            {
                return false;
            }
        }

        return true;
    }

    StoredDepthFramesResult collectLatestStoredDepthFrames(const QJsonObject& projectMeta)
    {
        StoredDepthFramesResult result;

        const QJsonArray depthResults = projectMeta.value(QStringLiteral("depth_map_results")).toArray();
        QString latestDir;
        for (int index = depthResults.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = depthResults.at(index).toObject();
            const QString depthPng = record.value(QStringLiteral("depth_png")).toString();
            const QString rawDepthPath =
                resolveExistingRawDepthPath(depthPng, record.value(QStringLiteral("raw_depth_path")).toString());
            if (!depthPng.isEmpty() && !rawDepthPath.isEmpty() && QFileInfo::exists(depthPng) &&
                QFileInfo::exists(rawDepthPath))
            {
                latestDir = QFileInfo(rawDepthPath).absolutePath();
                break;
            }
        }

        if (latestDir.isEmpty())
        {
            result.status = {false, QStringLiteral("未找到可复用的原始深度图，请先执行深度图估计")};
            return result;
        }

        return collectStoredDepthFramesInDirectory(depthResults, latestDir);
    }

    StoredDepthFramesResult collectStoredDepthFramesForDirectory(const QJsonObject& projectMeta,
                                                                 const QString& batchDirectory)
    {
        return collectStoredDepthFramesForDirectory(projectMeta.value(QStringLiteral("depth_map_results")).toArray(),
                                                    batchDirectory);
    }

    StoredDepthFramesResult collectStoredDepthFramesForDirectory(const QJsonArray& depthResults,
                                                                 const QString& batchDirectory)
    {
        return collectStoredDepthFramesInDirectory(depthResults, batchDirectory);
    }

    StoredDepthFramesResult selectFusionEligibleStoredDepthFrames(const StoredDepthFramesResult& storedFrames)
    {
        StoredDepthFramesResult result;
        result.batchDir = storedFrames.batchDir;
        if (!storedFrames.status.ok)
        {
            result.status = storedFrames.status;
            return result;
        }

        int accepted_count = 0;
        int fusion_eligible_count = 0;
        int validation_only_count = 0;
        result.frames.reserve(storedFrames.frames.size());
        for (const StoredDepthFrameRecord& frame : storedFrames.frames)
        {
            const QString acceptance = frame.acceptance.trimmed().toLower();
            if (acceptance == QStringLiteral("accepted"))
            {
                ++accepted_count;
            }
            else if (acceptance == QStringLiteral("validation_only"))
            {
                ++validation_only_count;
            }
            if (frame.fusionEligibilityKnown && frame.fusionEligible)
            {
                ++fusion_eligible_count;
            }
            if (xjw::mvs::isPrimaryFusionFrame(frame.role))
            {
                result.frames.push_back(frame);
            }
        }

        if (result.frames.size() < 2)
        {
            result.status = {false,
                             QStringLiteral("当前深度图批次没有足够的可融合主帧：accepted=%1，"
                                            "fusion_eligible=%2，同时满足=%3/%4，validation_only=%5；"
                                            "创建点云至少需要 2 帧同时满足 accepted 和 fusion_eligible。"
                                            "这不是目录权限问题；当前深度图缺少可靠的多视几何一致性，"
                                            "请检查环拍空三相机参数、相邻视角选择和深度估计质量后重新计算。")
                                 .arg(accepted_count)
                                 .arg(fusion_eligible_count)
                                 .arg(result.frames.size())
                                 .arg(storedFrames.frames.size())
                                 .arg(validation_only_count)};
            return result;
        }

        result.status = {true, QString()};
        return result;
    }

    std::vector<int> storedFusionSourceIndices(const std::vector<StoredDepthFrameRecord>& frames, int referenceIndex)
    {
        std::vector<int> indices;
        if (referenceIndex < 0 || referenceIndex >= static_cast<int>(frames.size()))
        {
            return indices;
        }

        const QStringList& source_images = frames[static_cast<std::size_t>(referenceIndex)].sourceImages;
        indices.reserve(static_cast<std::size_t>(source_images.size()));
        for (const QString& source_image : source_images)
        {
            const QString normalized_source = QDir::cleanPath(source_image);
            for (int index = 0; index < static_cast<int>(frames.size()); ++index)
            {
                if (index == referenceIndex)
                {
                    continue;
                }
                if (QDir::cleanPath(frames[static_cast<std::size_t>(index)].refImage)
                        .compare(normalized_source, Qt::CaseInsensitive) == 0)
                {
                    indices.push_back(index);
                    break;
                }
            }
        }
        return indices;
    }

    bool downsampleFusionFrameForMaxDimension(xjw::mvs::FusionFrameInput* frame, int fusionMaxImageDim)
    {
        if (!frame || frame->depthMap.empty() || fusionMaxImageDim <= 0)
        {
            return false;
        }

        const int oldWidth = frame->depthMap.cols;
        const int oldHeight = frame->depthMap.rows;
        const int oldMaxSide = std::max(oldWidth, oldHeight);
        if (oldWidth <= 0 || oldHeight <= 0 || oldMaxSide <= fusionMaxImageDim)
        {
            frame->imgW = oldWidth;
            frame->imgH = oldHeight;
            return false;
        }

        const double scale = static_cast<double>(fusionMaxImageDim) / static_cast<double>(oldMaxSide);
        const cv::Size targetSize(std::max(1, static_cast<int>(std::round(oldWidth * scale))),
                                  std::max(1, static_cast<int>(std::round(oldHeight * scale))));
        if (targetSize.width == oldWidth && targetSize.height == oldHeight)
        {
            frame->imgW = oldWidth;
            frame->imgH = oldHeight;
            return false;
        }
        if (!frame->cameraModel || frame->cameraModel->imageSize().samples != oldWidth ||
            frame->cameraModel->imageSize().lines != oldHeight)
        {
            return false;
        }

        const auto scaled_camera = scaledDepthCamera(*frame->cameraModel, targetSize);

        cv::Mat resizedDepth;
        cv::resize(frame->depthMap, resizedDepth, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
        frame->depthMap = std::move(resizedDepth);

        if (!frame->confidence.empty())
        {
            cv::Mat resizedConfidence;
            cv::resize(frame->confidence, resizedConfidence, targetSize, 0.0, 0.0, cv::INTER_AREA);
            frame->confidence = std::move(resizedConfidence);
        }

        frame->cameraModel = scaled_camera;
        frame->imgW = targetSize.width;
        frame->imgH = targetSize.height;
        return true;
    }

    FusionFrameBuildResult
    buildStoredFusionFrame(const StoredDepthFrameRecord& stored,
                           const placamera::FramePinholeModel& camera,
                           const xjw::mvs::FusionConfig& fusionConfig,
                           int viewCount,
                           int fusionMaxImageDim)
    {
        FusionFrameBuildResult result;
        const auto total_start = std::chrono::steady_clock::now();
        if (!xjw::mvs::isKnownDepthSceneProfile(stored.sceneProfile))
        {
            result.status = {false,
                             QStringLiteral("缓存深度帧的 scene_profile 缺失或无法识别: %1").arg(stored.sceneProfile)};
            return result;
        }
        cv::Size declared_raster_size;
        const bool has_declared_raster_size = readPositiveSize(stored.pixelDomainDiagnostics,
                                                               QStringLiteral("raster_width"),
                                                               QStringLiteral("raster_height"),
                                                               &declared_raster_size);
        result.frame.geometrySupportPrevalidated =
            stored.pixelDomainDiagnostics.value(QStringLiteral("producer")).toString() ==
            QStringLiteral("recovered_scene_d4");
        if (!readPreparedPngSize(stored.preparedImage, &result.frame.preparedRasterSize))
        {
            result.status = {false, QStringLiteral("缓存深度帧缺少有效的 prepared PNG 栅格")};
            return result;
        }
        result.frame.imagePath = xjw::common::io::toUtf8Path(stored.preparedImage);
        if (has_declared_raster_size && declared_raster_size != result.frame.preparedRasterSize)
        {
            result.status = {false,
                             QStringLiteral("缓存深度帧的 prepared raster 与 pixel_domain_diagnostics 尺寸不一致")};
            return result;
        }
        result.frame.cameraModel = depthCameraFromJson(stored.cameraModel, {stored.gridWidth, stored.gridHeight});
        if (!result.frame.cameraModel)
        {
            result.status = {false, QStringLiteral("缓存深度帧缺少有效的 PlaCamera 面阵针孔 identity/frame 或几何")};
            return result;
        }
        if (camera.imageId() != result.frame.cameraModel->imageId() ||
            camera.instanceId() != result.frame.cameraModel->instanceId() ||
            camera.groundFrame() != result.frame.cameraModel->groundFrame())
        {
            result.status = {false, QStringLiteral("缓存深度帧相机与当前 canonical 相机 identity/frame 不一致")};
            return result;
        }
        const auto prepared_camera = depthCameraFromJson(stored.preparedCameraModel, result.frame.preparedRasterSize);
        if (!prepared_camera || prepared_camera->imageId() != result.frame.cameraModel->imageId() ||
            prepared_camera->instanceId() != result.frame.cameraModel->instanceId() ||
            prepared_camera->groundFrame() != result.frame.cameraModel->groundFrame())
        {
            result.status = {false, QStringLiteral("缓存深度帧 prepared 相机与主相机 identity/frame 不一致")};
            return result;
        }
        result.frame.imgW = stored.gridWidth;
        result.frame.imgH = stored.gridHeight;
        const QString rawDepthPath = resolveExistingRawDepthPath(stored.depthPng, stored.rawDepthPath);
        const auto read_start = std::chrono::steady_clock::now();
        result.status = loadDepthMatStorage(rawDepthPath, &result.frame.depthMap);
        if (!result.status.ok)
        {
            return result;
        }

        cv::Size validated_native_raster_size;
        QString pixel_domain_error;
        if (!validateNativePixelDomainContract(
                stored, result.frame.depthMap.size(), &validated_native_raster_size, &pixel_domain_error))
        {
            result.status = {false, pixel_domain_error};
            return result;
        }
        if (stored.effectiveNativeFinalDepthGrid)
        {
            if (result.frame.preparedRasterSize != validated_native_raster_size)
            {
                result.status = {false, QStringLiteral("原生深度网格记录的 prepared raster 与 raster 尺寸不一致")};
                return result;
            }
        }

        if (stored.gridWidth != result.frame.depthMap.cols || stored.gridHeight != result.frame.depthMap.rows)
        {
            result.frame.cameraModel = scaledDepthCamera(*result.frame.cameraModel, result.frame.depthMap.size());
        }
        result.frame.imgW = result.frame.depthMap.cols;
        result.frame.imgH = result.frame.depthMap.rows;

        const QString rawConfidencePath = resolveExistingRawConfidencePath(stored.depthPng, stored.rawConfidencePath);
        if (!rawConfidencePath.isEmpty())
        {
            result.status = loadDepthMatStorage(rawConfidencePath, &result.frame.confidence);
            if (!result.status.ok)
            {
                result.status.errorMessage = QStringLiteral("读取缓存置信度图失败：%1").arg(result.status.errorMessage);
                return result;
            }
        }

        xjw::mvs::DepthPostProcessEvidence evidence;
        const bool require_geometry_evidence =
            stored.algorithmRevision == 0 || stored.algorithmRevision >= xjw::mvs::kMvsGeometryFusionSupportRevision;
        result.status = loadStoredEvidenceMap(stored.rawGeometrySupportPath,
                                              QStringLiteral("跨视几何支持图"),
                                              CV_16UC1,
                                              result.frame.depthMap.size(),
                                              require_geometry_evidence,
                                              &evidence.geometrySupportCount);
        if (!result.status.ok)
        {
            return result;
        }
        result.status = loadStoredEvidenceMap(stored.rawInverseDepthSpreadPath,
                                              QStringLiteral("逆深度相对离散度图"),
                                              CV_32FC1,
                                              result.frame.depthMap.size(),
                                              require_geometry_evidence,
                                              &evidence.inverseDepthRelativeSpread);
        if (!result.status.ok)
        {
            return result;
        }

        const bool require_adaptive_evidence =
            !stored.useDiscreteGeometryFallback &&
            stored.algorithmRevision >= xjw::mvs::kMvsAdaptiveGeometryEvidenceRevision &&
            xjw::mvs::isOrbitalDepthSceneProfile(stored.sceneProfile);
        const bool has_any_adaptive_evidence =
            !stored.useDiscreteGeometryFallback && (!stored.rawAdaptiveGeometrySupportWeightPath.isEmpty() ||
                                                    !stored.rawAdaptiveGeometryEffectiveViewCountPath.isEmpty() ||
                                                    !stored.rawAdaptiveGeometryConflictRatioPath.isEmpty());
        if (require_adaptive_evidence || has_any_adaptive_evidence)
        {
            result.status = loadStoredEvidenceMap(stored.rawAdaptiveGeometrySupportWeightPath,
                                                  QStringLiteral("连续几何支持权重图"),
                                                  CV_32FC1,
                                                  result.frame.depthMap.size(),
                                                  true,
                                                  &evidence.adaptiveSupportWeight);
            if (!result.status.ok)
            {
                return result;
            }
            result.status = loadStoredEvidenceMap(stored.rawAdaptiveGeometryEffectiveViewCountPath,
                                                  QStringLiteral("连续几何有效视图数图"),
                                                  CV_32FC1,
                                                  result.frame.depthMap.size(),
                                                  true,
                                                  &evidence.adaptiveEffectiveViewCount);
            if (!result.status.ok)
            {
                return result;
            }
            result.status = loadStoredEvidenceMap(stored.rawAdaptiveGeometryConflictRatioPath,
                                                  QStringLiteral("连续几何冲突比例图"),
                                                  CV_32FC1,
                                                  result.frame.depthMap.size(),
                                                  true,
                                                  &evidence.adaptiveConflictRatio);
            if (!result.status.ok)
            {
                return result;
            }
        }
        const auto read_done = std::chrono::steady_clock::now();

        const auto resize_start = read_done;
        downsampleFusionFrameForMaxDimension(&result.frame, fusionMaxImageDim);
        const cv::Size fusion_size = result.frame.depthMap.size();
        resizeEvidenceMap(&evidence.geometrySupportCount, fusion_size);
        resizeEvidenceMap(&evidence.inverseDepthRelativeSpread, fusion_size);
        resizeEvidenceMap(&evidence.adaptiveSupportWeight, fusion_size);
        resizeEvidenceMap(&evidence.adaptiveEffectiveViewCount, fusion_size);
        resizeEvidenceMap(&evidence.adaptiveConflictRatio, fusion_size);
        const auto resize_done = std::chrono::steady_clock::now();

        const cv::Size raster_pixel_domain_size = result.frame.preparedRasterSize;

        const auto postprocess_start = resize_done;
        result.frame.depthPostprocess =
            xjw::mvs::DepthPostprocessor::postprocessFusionDepthMap(result.frame.depthMap,
                                                                    result.frame.confidence,
                                                                    fusionConfig,
                                                                    frameIndexFromPath(stored.rawDepthPath),
                                                                    viewCount,
                                                                    nullptr,
                                                                    &evidence,
                                                                    raster_pixel_domain_size);
        const auto postprocess_done = std::chrono::steady_clock::now();
        result.frame.geometrySupportCount = std::move(evidence.geometrySupportCount);
        if (!result.frame.geometrySupportCount.empty())
        {
            result.frame.geometrySupportCount.setTo(cv::Scalar(0), result.frame.depthMap <= 0.0f);
        }
        result.frame.confidence.release();
        result.frame.imgW = result.frame.depthMap.cols;
        result.frame.imgH = result.frame.depthMap.rows;
        result.status = {true, QString()};
        result.readMs = std::chrono::duration<double, std::milli>(read_done - read_start).count();
        result.resizeMs = std::chrono::duration<double, std::milli>(resize_done - resize_start).count();
        result.postprocessMs = std::chrono::duration<double, std::milli>(postprocess_done - postprocess_start).count();
        result.totalMs = std::chrono::duration<double, std::milli>(postprocess_done - total_start).count();
        return result;
    }

} // namespace xjw::core::project

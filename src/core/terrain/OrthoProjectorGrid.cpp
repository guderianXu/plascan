#include "OrthoProjector.h"

#include "camera/project/CameraProjectRuntime.h"
#include "camera/models/CameraModelFactories.h"
#include "project/ProjectMetadata.h"

#include <QJsonArray>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{

    namespace
    {

        QString normalizedPath(const QString& path)
        {
            return xjw::common::project::normalizedImageToken(xjw::common::project::normalizePath(path));
        }

        QJsonObject findImageEntry(const QJsonArray& entries, const QString& selectedPath)
        {
            const QString selected_normalized = normalizedPath(selectedPath);
            if (selected_normalized.isEmpty())
            {
                return {};
            }
            QJsonObject exact_match;
            int exact_match_count = 0;
            for (const QJsonValue& value : entries)
            {
                const QJsonObject entry = value.toObject();
                const QString stored_path = entry.value(QStringLiteral("path")).toString();
                if (!stored_path.isEmpty() && normalizedPath(stored_path) == selected_normalized)
                {
                    exact_match = entry;
                    ++exact_match_count;
                }
            }
            if (exact_match_count == 1)
            {
                return exact_match;
            }
            return {};
        }

        bool finitePositive(double value)
        {
            return std::isfinite(value) && value > 0.0;
        }

    } // namespace

    bool OrthoProjector::planOutputGrid(const DemGridData& demGrid,
                                        const OrthoGenerationOptions& options,
                                        OrthoOutputGrid* outputGrid,
                                        QString* errorMsg)
    {
        if (!outputGrid || demGrid.width <= 0 || demGrid.height <= 0 || !finitePositive(demGrid.stepX) ||
            !finitePositive(demGrid.stepY))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("DEM 网格无效，无法估算正射输出");
            }
            return false;
        }
        if (!options.validate(errorMsg))
        {
            return false;
        }

        const double dem_min_x = demGrid.minX - 0.5 * demGrid.stepX;
        const double dem_min_y = demGrid.minY - 0.5 * demGrid.stepY;
        const double dem_max_x = demGrid.minX + (static_cast<double>(demGrid.width) - 0.5) * demGrid.stepX;
        const double dem_max_y = demGrid.minY + (static_cast<double>(demGrid.height) - 0.5) * demGrid.stepY;

        double min_x = dem_min_x;
        double min_y = dem_min_y;
        double max_x = dem_max_x;
        double max_y = dem_max_y;
        if (options.bounds.enabled)
        {
            min_x = std::max(min_x, options.bounds.minX);
            min_y = std::max(min_y, options.bounds.minY);
            max_x = std::min(max_x, options.bounds.maxX);
            max_y = std::min(max_y, options.bounds.maxY);
        }
        if (!(min_x < max_x) || !(min_y < max_y))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("指定正射区域与 DEM 范围没有交集");
            }
            return false;
        }

        double pixel_x = options.pixelSizeX > 0.0 ? options.pixelSizeX : demGrid.stepX;
        double pixel_y = options.pixelSizeY > 0.0 ? options.pixelSizeY : demGrid.stepY;
        if (options.sizingMode == OrthoSizingMode::MaximumDimension)
        {
            const double scale_x = (max_x - min_x) / (static_cast<double>(options.maximumDimension) * demGrid.stepX);
            const double scale_y = (max_y - min_y) / (static_cast<double>(options.maximumDimension) * demGrid.stepY);
            const double scale = std::max({1.0, scale_x, scale_y});
            pixel_x = demGrid.stepX * scale;
            pixel_y = demGrid.stepY * scale;
        }

        const double width_value = std::ceil((max_x - min_x) / pixel_x - 1e-12);
        const double height_value = std::ceil((max_y - min_y) / pixel_y - 1e-12);
        if (width_value < 1.0 || height_value < 1.0 ||
            width_value > static_cast<double>(std::numeric_limits<int>::max()) ||
            height_value > static_cast<double>(std::numeric_limits<int>::max()))
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射输出尺寸超出可表示范围");
            }
            return false;
        }

        const int width = static_cast<int>(width_value);
        const int height = static_cast<int>(height_value);
        const qint64 pixel_count = static_cast<qint64>(width) * static_cast<qint64>(height);
        if (pixel_count > options.maximumPixelCount)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射输出像素数 %1 超过限制 %2，请增大像元或缩小区域")
                                .arg(pixel_count)
                                .arg(options.maximumPixelCount);
            }
            return false;
        }

        OrthoOutputGrid planned;
        planned.minEdgeX = min_x;
        planned.minEdgeY = min_y;
        planned.maxEdgeX = min_x + static_cast<double>(width) * pixel_x;
        planned.maxEdgeY = min_y + static_cast<double>(height) * pixel_y;
        planned.reference.width = width;
        planned.reference.height = height;
        planned.reference.minX = min_x + 0.5 * pixel_x;
        planned.reference.minY = min_y + 0.5 * pixel_y;
        planned.reference.stepX = pixel_x;
        planned.reference.stepY = pixel_y;
        planned.reference.projection = demGrid.projection;
        planned.estimatedMemoryBytes = pixel_count * 16;
        planned.resolvedOptions = options;
        planned.resolvedOptions.pixelSizeX = pixel_x;
        planned.resolvedOptions.pixelSizeY = pixel_y;
        planned.resolvedOptions.bounds.minX = planned.minEdgeX;
        planned.resolvedOptions.bounds.minY = planned.minEdgeY;
        planned.resolvedOptions.bounds.maxX = planned.maxEdgeX;
        planned.resolvedOptions.bounds.maxY = planned.maxEdgeY;
        *outputGrid = std::move(planned);
        return true;
    }

    bool OrthoProjector::buildImageInputs(const QStringList& selectedImages,
                                          const QJsonObject& projectMeta,
                                          std::vector<OrthoImageInput>* inputs,
                                          QString* errorMsg)
    {
        if (!inputs)
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射影像输入对象为空");
            }
            return false;
        }

        inputs->clear();
        if (selectedImages.isEmpty())
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射投影至少需要一幅项目影像");
            }
            return false;
        }

        const QJsonObject project_files = xjw::common::project::projectFilesRootObject(projectMeta);
        const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
            project_files, xjw::camera_models::makeBuiltinCameraModelRegistry());
        if (!runtime.ok())
        {
            if (errorMsg)
            {
                *errorMsg =
                    QStringLiteral("正射投影的规范相机数据无效：%1").arg(runtime.errors.join(QStringLiteral("; ")));
            }
            return false;
        }

        const QJsonArray entries = xjw::common::project::projectImageEntries(project_files);
        std::vector<QJsonObject> selected_entries;
        std::vector<camera_core::ImageId> image_ids;
        selected_entries.reserve(static_cast<std::size_t>(selectedImages.size()));
        image_ids.reserve(static_cast<std::size_t>(selectedImages.size()));
        for (const QString& selected_path : selectedImages)
        {
            const QJsonObject entry = findImageEntry(entries, selected_path);
            if (entry.isEmpty())
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("无法唯一匹配所选影像的项目元数据: %1").arg(selected_path);
                }
                inputs->clear();
                return false;
            }
            const QString image_id = entry.value(QStringLiteral("image_uuid")).toString().trimmed();
            if (image_id.isEmpty())
            {
                if (errorMsg)
                {
                    *errorMsg = QStringLiteral("所选影像缺少稳定 ImageId: %1").arg(selected_path);
                }
                return false;
            }
            selected_entries.push_back(entry);
            image_ids.emplace_back(image_id.toStdString());
        }

        const camera_core::CameraOperationPlan plan =
            runtime.planOperationForImages(image_ids, camera_core::CameraOperation::OrthoProjection);
        if (!plan.ok())
        {
            if (errorMsg)
            {
                *errorMsg =
                    QStringLiteral("正射投影相机能力校验失败：%1").arg(QString::fromStdString(plan.failureMessage()));
            }
            return false;
        }

        std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> states;
        std::string state_error;
        if (!runtime.framePinholeStatesForImages(image_ids, &states, &state_error))
        {
            if (errorMsg)
            {
                *errorMsg =
                    QStringLiteral("正射投影面阵针孔数值状态解析失败：%1").arg(QString::fromStdString(state_error));
            }
            return false;
        }
        if (states.size() != selected_entries.size())
        {
            if (errorMsg)
            {
                *errorMsg = QStringLiteral("正射投影相机状态与所选影像数量不一致");
            }
            return false;
        }

        inputs->reserve(selected_entries.size());
        for (std::size_t index = 0; index < selected_entries.size(); ++index)
        {
            const QJsonObject& entry = selected_entries[index];
            OrthoImageInput input;
            input.imagePath = selectedImages.at(static_cast<qsizetype>(index));
            input.imageId = QString::fromStdString(image_ids[index].value());
            input.exclusionMaskPath = entry.value(QStringLiteral("mask_path")).toString();
            input.camera = std::move(states[index]);
            inputs->push_back(std::move(input));
        }
        return true;
    }

} // namespace xjw

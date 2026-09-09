#include "RecoveredModelColorizer.h"
#include "VertexColorOptions.h"
#include "io/PathIO.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include <QFileInfo>
#include <QJsonArray>
#include <QResource>
#include <cuda_runtime_api.h>

static void initializeRecoveredVertexColorResources()
{
    Q_INIT_RESOURCE(recovered_vertex_color);
}

namespace xjw::mesh
{
    QJsonObject colorizeRecoveredModel(metmodel::Mesh& mesh,
                                       metmodel::Scene& scene,
                                       const QJsonObject& settings,
                                       int device,
                                       const std::function<bool()>& is_cancelled,
                                       const std::function<void(const QString&, int)>& progress)
    {
        const auto images = settings.value(QStringLiteral("recovered_source_images")).toArray();
        if (static_cast<std::size_t>(images.size()) != scene.cameras.size())
            throw std::runtime_error("Recovered vertex colors require the complete original image list");
        if (progress)
            progress(QStringLiteral("加载参考取色使用的完整 RGB 相机影像…"), 90);
        for (std::size_t i = 0; i < scene.cameras.size(); ++i)
        {
            if (is_cancelled && is_cancelled())
                throw std::runtime_error("Recovered color loading cancelled");
            auto path = images[static_cast<qsizetype>(i)].toString();
            if (!scene.cameras[i].path.empty())
            {
                const auto stored = scene.cameras[i].path.u8string();
                path = QString::fromUtf8(reinterpret_cast<const char*>(stored.data()),
                                         static_cast<qsizetype>(stored.size()));
            }
            if (path.isEmpty() || !QFileInfo(path).isFile())
                throw std::runtime_error("Missing recovered color image: " + path.toStdString());
            auto image = metalign::load_rgb_image(xjw::common::io::toFilesystemPath(path));
            auto& camera = scene.cameras[i];
            if (image.width != camera.image.width || image.height != camera.image.height)
                throw std::runtime_error("Recovered color image and Brown camera dimensions differ: " +
                                         path.toStdString());
            // The reference RGB scene loader derives sensor offsets from the
            // absolute principal point. Older depth bundles did not populate
            // these redundant fields because PatchMatch only uses cx/cy.
            camera.model.cx_offset = camera.model.cx - static_cast<double>(image.width) * 0.5;
            camera.model.cy_offset = camera.model.cy - static_cast<double>(image.height) * 0.5;
            camera.image = std::move(image);
        }
        initializeRecoveredVertexColorResources();
        cudaDeviceProp properties{};
        const auto device_status = cudaGetDeviceProperties(&properties, device);
        if (device_status != cudaSuccess)
            throw std::runtime_error(std::string("Cannot identify selected CUDA device: ") +
                                     cudaGetErrorString(device_status));
        metmodel::RecoveredVertexColorVulkanOptions options;
        std::array<std::uint8_t, 16> uuid{};
        static_assert(sizeof(properties.uuid.bytes) == uuid.size());
        std::copy_n(reinterpret_cast<const std::uint8_t*>(properties.uuid.bytes), uuid.size(), uuid.begin());
        options.deviceUuid = uuid;
        options.isCancelled = is_cancelled;
        options.progress = [&](std::size_t completed, std::size_t total)
        {
            if (progress)
                progress(QStringLiteral("参考 Vulkan 七阶段顶点取色 %1/%2…").arg(completed).arg(total),
                         91 + static_cast<int>(completed * 7 / total));
        };
        const auto color =
            metmodel::colorize_mesh_recovered_vulkan(mesh, scene.cameras, ":/plascan/recovered_vertex_color", options);
        return {{"vertex_color_algorithm", "recovered_vulkan_seven_stage"},
                {"vertex_color_device", QString::fromStdString(color.device_name)},
                {"color_source_view_count", static_cast<qint64>(color.cameras)},
                {"reliably_colored_vertex_count", static_cast<qint64>(color.directly_colored_vertices)},
                {"extrapolated_vertex_color_count", static_cast<qint64>(color.extrapolated_vertices)},
                {"uncolored_vertex_count", static_cast<qint64>(color.uncolored_vertices)},
                {"vertex_color_gpu_seconds", color.gpu_seconds},
                {"vertex_color_finalize_seconds", color.finalize_seconds}};
    }
} // namespace xjw::mesh

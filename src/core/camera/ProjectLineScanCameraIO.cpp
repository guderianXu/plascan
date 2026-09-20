#include "ProjectCameraIO.h"

#include "camera/models/LineScanModelJson.h"

namespace xjw::common::project
{

    QJsonObject serializeLineScanInstance(const xjw::camera_models::linescan::LineScanInstance& camera)
    {
        const QJsonObject parameters =
            xjw::camera_models::lineScanDefinitionParametersToJson(camera.lineScanDefinition());
        const QJsonObject state = xjw::camera_models::lineScanInstanceStateToJson(camera);
        QJsonObject result{
            {QStringLiteral("model"), QStringLiteral("planetary_linescan")},
            {QStringLiteral("world_frame"), QString::fromStdString(camera.definition().worldFrame().value())},
            {QStringLiteral("image_samples"), camera.imageSize().samples},
            {QStringLiteral("image_lines"), camera.imageSize().lines},
            {QStringLiteral("optics"), parameters.value(QStringLiteral("optics"))},
            {QStringLiteral("pixel_convention"), parameters.value(QStringLiteral("pixel_convention"))},
            {QStringLiteral("trajectory"), state.value(QStringLiteral("trajectory"))},
            {QStringLiteral("line_timing"), state.value(QStringLiteral("line_timing"))}};
        if (state.contains(QStringLiteral("capture_time")))
        {
            result.insert(QStringLiteral("capture_time"), state.value(QStringLiteral("capture_time")));
        }
        return result;
    }

} // namespace xjw::common::project

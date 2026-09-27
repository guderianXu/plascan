#include "ProjectFramePinholeMetadataIO.h"

#include <QJsonArray>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace xjw::common::project
{
    namespace
    {

        bool readFiniteNumber(const QJsonObject& object, const QString& key, double* output)
        {
            const QJsonValue value = object.value(key);
            if (!output || !value.isDouble() || !std::isfinite(value.toDouble()))
            {
                return false;
            }
            *output = value.toDouble();
            return true;
        }

        template <std::size_t N>
        bool readFiniteArray(const QJsonObject& object, const QString& key, std::array<double, N>* output)
        {
            if (!output || !object.value(key).isArray())
            {
                return false;
            }
            const QJsonArray values = object.value(key).toArray();
            if (values.size() != static_cast<qsizetype>(N))
            {
                return false;
            }
            for (std::size_t index = 0; index < N; ++index)
            {
                const QJsonValue value = values.at(static_cast<qsizetype>(index));
                if (!value.isDouble() || !std::isfinite(value.toDouble()))
                {
                    return false;
                }
                (*output)[index] = value.toDouble();
            }
            return true;
        }

        void setError(QString* error, const QString& message)
        {
            if (error)
            {
                *error = message;
            }
        }

    } // namespace

    std::optional<ProjectFramePinhole>
    decodeFramePinholeMetadata(const QJsonObject& metadata, placamera::CameraDefinitionId definition_id, QString* error)
    {
        if (error)
        {
            error->clear();
        }
        if (metadata.value(QStringLiteral("model")).toString() != QStringLiteral("frame_pinhole") ||
            metadata.value(QStringLiteral("intrinsics_unit")).toString() != QStringLiteral("mm") ||
            metadata.value(QStringLiteral("camera_center_unit")).toString() != QStringLiteral("m") ||
            metadata.value(QStringLiteral("pixel_convention")).toString() != QStringLiteral("center"))
        {
            setError(error, QStringLiteral("模型、单位或像素约定无效"));
            return std::nullopt;
        }

        const QString frame_value = metadata.value(QStringLiteral("world_frame")).toString().trimmed();
        if (frame_value.isEmpty() || !metadata.value(QStringLiteral("depth_axis_flipped")).isBool())
        {
            setError(error, QStringLiteral("缺少世界坐标系或深度轴方向"));
            return std::nullopt;
        }

        std::array<double, 3> center{};
        std::array<double, 9> rotation{};
        double pitch = 0.0;
        double fu = 0.0;
        double fv = 0.0;
        double cu = 0.0;
        double cv = 0.0;
        placamera::BrownConradyDistortion distortion;
        double u_axis = 0.0;
        double v_axis = 0.0;
        if (!readFiniteArray(metadata, QStringLiteral("C"), &center) ||
            !readFiniteArray(metadata, QStringLiteral("R"), &rotation) ||
            !readFiniteNumber(metadata, QStringLiteral("pitch"), &pitch) || pitch <= 0.0 ||
            !readFiniteNumber(metadata, QStringLiteral("fu"), &fu) ||
            !readFiniteNumber(metadata, QStringLiteral("fv"), &fv) ||
            !readFiniteNumber(metadata, QStringLiteral("cu"), &cu) ||
            !readFiniteNumber(metadata, QStringLiteral("cv"), &cv) ||
            !readFiniteNumber(metadata, QStringLiteral("k1"), &distortion.radialK1) ||
            !readFiniteNumber(metadata, QStringLiteral("k2"), &distortion.radialK2) ||
            !readFiniteNumber(metadata, QStringLiteral("k3"), &distortion.radialK3) ||
            !readFiniteNumber(metadata, QStringLiteral("p1"), &distortion.tangentialP1) ||
            !readFiniteNumber(metadata, QStringLiteral("p2"), &distortion.tangentialP2) ||
            !readFiniteNumber(metadata, QStringLiteral("u_direction"), &u_axis) ||
            !readFiniteNumber(metadata, QStringLiteral("v_direction"), &v_axis) || (u_axis != -1.0 && u_axis != 1.0) ||
            (v_axis != -1.0 && v_axis != 1.0))
        {
            setError(error, QStringLiteral("相机内外参缺失或数值无效"));
            return std::nullopt;
        }

        const bool has_image_width = metadata.contains(QStringLiteral("image_width"));
        const bool has_image_height = metadata.contains(QStringLiteral("image_height"));
        if (has_image_width != has_image_height)
        {
            setError(error, QStringLiteral("影像宽高必须同时提供"));
            return std::nullopt;
        }
        if (has_image_width)
        {
            double image_width = 0.0;
            double image_height = 0.0;
            if (!readFiniteNumber(metadata, QStringLiteral("image_width"), &image_width) ||
                !readFiniteNumber(metadata, QStringLiteral("image_height"), &image_height) || image_width <= 0.0 ||
                image_height <= 0.0 || std::floor(image_width) != image_width ||
                std::floor(image_height) != image_height ||
                image_width > static_cast<double>(std::numeric_limits<int>::max()) ||
                image_height > static_cast<double>(std::numeric_limits<int>::max()))
            {
                setError(error, QStringLiteral("影像宽高必须是有效的正整数"));
                return std::nullopt;
            }
        }

        try
        {
            const placamera::FrameId frame(frame_value.toStdString());
            const placamera::FrameIntrinsics intrinsics{fu / pitch,
                                                        fv / pitch,
                                                        cu / pitch,
                                                        cv / pitch,
                                                        pitch,
                                                        static_cast<int>(u_axis),
                                                        static_cast<int>(v_axis)};
            auto definition = placamera::FramePinholeDefinition::create(
                std::move(definition_id),
                intrinsics,
                distortion,
                placamera::PixelConvention::PixelCenter,
                frame,
                metadata.value(QStringLiteral("depth_axis_flipped")).toBool());
            auto pose = placamera::Pose::create(frame, center, rotation);
            return ProjectFramePinhole{std::move(definition), std::move(pose)};
        }
        catch (const std::exception& exception)
        {
            setError(error, QString::fromUtf8(exception.what()));
            return std::nullopt;
        }
    }

} // namespace xjw::common::project

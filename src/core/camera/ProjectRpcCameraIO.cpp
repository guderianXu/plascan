#include "ProjectCameraIO.h"

#include "RpcCameraIO.h"
#include "io/PathIO.h"

#include <QJsonArray>

#include <cmath>
#include <memory>
#include <utility>

namespace xjw::common::project
{
    namespace
    {

        QJsonArray coefficientsToJson(const xjw::RpcCameraModel::Coefficients& coefficients)
        {
            QJsonArray result;
            for (double coefficient : coefficients)
            {
                result.append(coefficient);
            }
            return result;
        }

        bool coefficientsFromJson(const QJsonObject& object,
                                  const QString& key,
                                  xjw::RpcCameraModel::Coefficients* coefficients)
        {
            const QJsonArray values = object.value(key).toArray();
            if (!coefficients || values.size() != static_cast<int>(coefficients->size()))
            {
                return false;
            }
            for (int index = 0; index < values.size(); ++index)
            {
                if (!values.at(index).isDouble())
                {
                    return false;
                }
                (*coefficients)[static_cast<std::size_t>(index)] = values.at(index).toDouble();
            }
            return true;
        }

        bool requiredRpcNumber(const QJsonObject& object, const QString& key, double* value)
        {
            const QJsonValue field = object.value(key);
            if (!value || !field.isDouble())
            {
                return false;
            }
            *value = field.toDouble();
            return std::isfinite(*value);
        }

        bool hasImageCorrection(const xjw::RpcCameraModel::ImageCorrection& correction)
        {
            return correction.sampleOffsetPixels != 0.0 || correction.sampleSamplePixels != 0.0 ||
                   correction.sampleLinePixels != 0.0 || correction.lineOffsetPixels != 0.0 ||
                   correction.lineSamplePixels != 0.0 || correction.lineLinePixels != 0.0;
        }

        QJsonObject imageCorrectionToJson(const xjw::RpcCameraModel::ImageCorrection& correction)
        {
            return {{QStringLiteral("model"), QStringLiteral("affine_normalized_v1")},
                    {QStringLiteral("sample_offset_px"), correction.sampleOffsetPixels},
                    {QStringLiteral("sample_sample_px"), correction.sampleSamplePixels},
                    {QStringLiteral("sample_line_px"), correction.sampleLinePixels},
                    {QStringLiteral("line_offset_px"), correction.lineOffsetPixels},
                    {QStringLiteral("line_sample_px"), correction.lineSamplePixels},
                    {QStringLiteral("line_line_px"), correction.lineLinePixels}};
        }

        bool imageCorrectionFromJson(const QJsonObject& object, xjw::RpcCameraModel::ImageCorrection* correction)
        {
            if (!correction ||
                object.value(QStringLiteral("model")).toString() != QStringLiteral("affine_normalized_v1"))
            {
                return false;
            }
            return requiredRpcNumber(object, QStringLiteral("sample_offset_px"), &correction->sampleOffsetPixels) &&
                   requiredRpcNumber(object, QStringLiteral("sample_sample_px"), &correction->sampleSamplePixels) &&
                   requiredRpcNumber(object, QStringLiteral("sample_line_px"), &correction->sampleLinePixels) &&
                   requiredRpcNumber(object, QStringLiteral("line_offset_px"), &correction->lineOffsetPixels) &&
                   requiredRpcNumber(object, QStringLiteral("line_sample_px"), &correction->lineSamplePixels) &&
                   requiredRpcNumber(object, QStringLiteral("line_line_px"), &correction->lineLinePixels);
        }

    } // namespace

    QJsonObject cameraToJson(const xjw::RpcCameraModel& camera)
    {
        const xjw::RpcCameraModel::Parameters& parameters = camera.parameters();
        QJsonObject result{{QStringLiteral("model"), QStringLiteral("rpc")},
                           {QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                           {QStringLiteral("ground_crs"), QStringLiteral("EPSG:4979")},
                           {QStringLiteral("world_frame"), QStringLiteral("EPSG:4978")},
                           {QStringLiteral("height_datum"), QStringLiteral("WGS84_ellipsoidal")},
                           {QStringLiteral("pixel_convention"), QStringLiteral("opencv_zero_based_center")},
                           {QStringLiteral("line_off"), parameters.lineOffset},
                           {QStringLiteral("samp_off"), parameters.sampleOffset},
                           {QStringLiteral("lat_off"), parameters.latitudeOffset},
                           {QStringLiteral("long_off"), parameters.longitudeOffset},
                           {QStringLiteral("height_off"), parameters.heightOffset},
                           {QStringLiteral("line_scale"), parameters.lineScale},
                           {QStringLiteral("samp_scale"), parameters.sampleScale},
                           {QStringLiteral("lat_scale"), parameters.latitudeScale},
                           {QStringLiteral("long_scale"), parameters.longitudeScale},
                           {QStringLiteral("height_scale"), parameters.heightScale},
                           {QStringLiteral("line_num_coeff"), coefficientsToJson(parameters.lineNumerator)},
                           {QStringLiteral("line_den_coeff"), coefficientsToJson(parameters.lineDenominator)},
                           {QStringLiteral("samp_num_coeff"), coefficientsToJson(parameters.sampleNumerator)},
                           {QStringLiteral("samp_den_coeff"), coefficientsToJson(parameters.sampleDenominator)}};
        if (parameters.errorBiasMeters)
        {
            result[QStringLiteral("err_bias_m")] = *parameters.errorBiasMeters;
        }
        if (parameters.errorRandomMeters)
        {
            result[QStringLiteral("err_rand_m")] = *parameters.errorRandomMeters;
        }
        if (hasImageCorrection(camera.imageCorrection()))
        {
            result[QStringLiteral("image_correction")] = imageCorrectionToJson(camera.imageCorrection());
        }
        if (camera.imageSize())
        {
            result[QStringLiteral("image_samples")] = camera.imageSize()->samples;
            result[QStringLiteral("image_lines")] = camera.imageSize()->lines;
        }
        return result;
    }

    bool parseRpcCameraRaster(const QString& raster_path, QJsonObject* camera_metadata, QString* error_message)
    {
        if (!camera_metadata)
        {
            if (error_message)
            {
                *error_message = QStringLiteral("相机元数据输出参数为空");
            }
            return false;
        }
        *camera_metadata = QJsonObject{};
        xjw::RpcCameraModel camera;
        std::string error;
        if (!xjw::loadRpcCameraFromRaster(xjw::common::io::toUtf8Path(raster_path), &camera, &error))
        {
            if (error_message)
            {
                *error_message = QString::fromUtf8(error.c_str());
            }
            return false;
        }
        *camera_metadata = cameraToJson(camera);
        return true;
    }

    bool cameraFromJson(const QJsonObject& camera_object, xjw::RpcCameraModel* camera)
    {
        if (!camera || camera_object.value(QStringLiteral("model"))
                               .toString()
                               .compare(QStringLiteral("rpc"), Qt::CaseInsensitive) != 0)
        {
            return false;
        }
        xjw::RpcCameraModel::Parameters parameters;
        if (!requiredRpcNumber(camera_object, QStringLiteral("line_off"), &parameters.lineOffset) ||
            !requiredRpcNumber(camera_object, QStringLiteral("samp_off"), &parameters.sampleOffset) ||
            !requiredRpcNumber(camera_object, QStringLiteral("lat_off"), &parameters.latitudeOffset) ||
            !requiredRpcNumber(camera_object, QStringLiteral("long_off"), &parameters.longitudeOffset) ||
            !requiredRpcNumber(camera_object, QStringLiteral("height_off"), &parameters.heightOffset) ||
            !requiredRpcNumber(camera_object, QStringLiteral("line_scale"), &parameters.lineScale) ||
            !requiredRpcNumber(camera_object, QStringLiteral("samp_scale"), &parameters.sampleScale) ||
            !requiredRpcNumber(camera_object, QStringLiteral("lat_scale"), &parameters.latitudeScale) ||
            !requiredRpcNumber(camera_object, QStringLiteral("long_scale"), &parameters.longitudeScale) ||
            !requiredRpcNumber(camera_object, QStringLiteral("height_scale"), &parameters.heightScale) ||
            !coefficientsFromJson(camera_object, QStringLiteral("line_num_coeff"), &parameters.lineNumerator) ||
            !coefficientsFromJson(camera_object, QStringLiteral("line_den_coeff"), &parameters.lineDenominator) ||
            !coefficientsFromJson(camera_object, QStringLiteral("samp_num_coeff"), &parameters.sampleNumerator) ||
            !coefficientsFromJson(camera_object, QStringLiteral("samp_den_coeff"), &parameters.sampleDenominator))
        {
            return false;
        }
        double optional_error = 0.0;
        if (requiredRpcNumber(camera_object, QStringLiteral("err_bias_m"), &optional_error))
        {
            parameters.errorBiasMeters = optional_error;
        }
        if (requiredRpcNumber(camera_object, QStringLiteral("err_rand_m"), &optional_error))
        {
            parameters.errorRandomMeters = optional_error;
        }

        xjw::RpcCameraModel parsed;
        if (!parsed.setParameters(parameters))
        {
            return false;
        }
        if (camera_object.contains(QStringLiteral("image_correction")))
        {
            xjw::RpcCameraModel::ImageCorrection correction;
            if (!imageCorrectionFromJson(camera_object.value(QStringLiteral("image_correction")).toObject(),
                                         &correction) ||
                !parsed.setImageCorrection(correction))
            {
                return false;
            }
        }
        const int samples = camera_object.value(QStringLiteral("image_samples")).toInt();
        const int lines = camera_object.value(QStringLiteral("image_lines")).toInt();
        if (samples > 0 && lines > 0)
        {
            parsed.setImageSize(xjw::CameraImageSize{samples, lines});
        }
        *camera = std::move(parsed);
        return true;
    }

    bool imageCameraFromEntry(const QJsonObject& image_object, xjw::RpcCameraModel* camera)
    {
        return cameraFromJson(image_object.value(QStringLiteral("camera")).toObject(), camera);
    }

    std::unique_ptr<xjw::CameraModel> cameraModelFromJson(const QJsonObject& camera_object)
    {
        if (camera_object.value(QStringLiteral("model"))
                .toString()
                .compare(QStringLiteral("rpc"), Qt::CaseInsensitive) == 0)
        {
            auto camera = std::make_unique<xjw::RpcCameraModel>();
            if (!cameraFromJson(camera_object, camera.get()))
            {
                return nullptr;
            }
            return camera;
        }
        auto camera = std::make_unique<xjw::FramePinholeCamera>();
        if (!cameraFromJson(camera_object, camera.get()))
        {
            return nullptr;
        }
        return camera;
    }

    std::unique_ptr<xjw::CameraModel> imageCameraModelFromEntry(const QJsonObject& image_object)
    {
        return cameraModelFromJson(image_object.value(QStringLiteral("camera")).toObject());
    }

} // namespace xjw::common::project

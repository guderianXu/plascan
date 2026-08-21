#include "MvsImagePreprocessor.h"

#include "io/PathIO.h"

#include <QByteArray>
#include <QDir>
#include <QString>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace xjw
{
namespace mvs
{
namespace
{

bool isFiniteDistortion(const FramePinholeCamera::Distortion &distortion)
{
    return std::isfinite(distortion.radialK1)
        && std::isfinite(distortion.radialK2)
        && std::isfinite(distortion.radialK3)
        && std::isfinite(distortion.tangentialP1)
        && std::isfinite(distortion.tangentialP2);
}

bool hasDistortion(const FramePinholeCamera::Distortion &distortion) noexcept
{
    constexpr double epsilon = 1e-15;
    return std::fabs(distortion.radialK1) > epsilon
        || std::fabs(distortion.radialK2) > epsilon
        || std::fabs(distortion.radialK3) > epsilon
        || std::fabs(distortion.tangentialP1) > epsilon
        || std::fabs(distortion.tangentialP2) > epsilon;
}

bool writePngAtomic(const QString &path,
                    const cv::Mat &image,
                    std::string *errorMessage)
{
    std::vector<std::uint8_t> encoded;
    try
    {
        if (!cv::imencode(".png", image, encoded) || encoded.empty())
        {
            if (errorMessage)
            {
                *errorMessage = "无法编码 MVS prepared PNG: " +
                    xjw::common::io::toUtf8Path(path);
            }
            return false;
        }
    }
    catch (const cv::Exception &exception)
    {
        if (errorMessage)
        {
            *errorMessage = std::string("编码 MVS prepared PNG 失败: ") +
                exception.what();
        }
        return false;
    }

    const QByteArray payload(
        reinterpret_cast<const char *>(encoded.data()),
        static_cast<qsizetype>(encoded.size()));
    QString write_error;
    if (!xjw::common::io::writeFileBytesAtomic(path, payload, &write_error))
    {
        if (errorMessage)
        {
            const QByteArray write_error_utf8 = write_error.toUtf8();
            *errorMessage = "无法原子写入 MVS prepared PNG: " +
                xjw::common::io::toUtf8Path(path) + " (" +
                std::string(write_error_utf8.constData(),
                            static_cast<std::size_t>(write_error_utf8.size())) +
                ")";
        }
        return false;
    }
    return true;
}

} // namespace

bool mvsImagePreparationRequiresDistinctPixels(const FramePinholeCamera &camera) noexcept
{
    return hasDistortion(camera.distortion());
}

cv::Mat normalizeMvsPhotometry(const cv::Mat &source,
                               MvsSceneProfile sceneProfile)
{
    if (source.empty() || source.type() != CV_8UC1)
    {
        return source;
    }

    const double image_mean = cv::mean(source)[0];
    cv::Mat normalized;
    if (sceneProfile == MvsSceneProfile::OrbitalObject)
    {
        // Planetary/orbital sequences commonly mix strongly illuminated and
        // shadowed surface views. Apply one moderate transform to the whole
        // batch; the previous mean-threshold branch enhanced only part of a
        // sequence and damaged cross-view photometric comparability.
        cv::createCLAHE(2.0, cv::Size(8, 8))->apply(source, normalized);
        return normalized;
    }

    if (image_mean >= 80.0)
    {
        return source;
    }
    if (image_mean < 30.0)
    {
        cv::Mat float_image;
        source.convertTo(float_image, CV_32F, 1.0 / 255.0);
        cv::pow(float_image, 0.4, float_image);
        float_image.convertTo(normalized, CV_8U, 255.0);
        cv::createCLAHE(8.0, cv::Size(8, 8))->apply(
            normalized, normalized);
        return normalized;
    }

    cv::createCLAHE(4.0, cv::Size(8, 8))->apply(source, normalized);
    return normalized;
}

bool prepareMvsImage(const cv::Mat &source,
                     const FramePinholeCamera &sourceCamera,
                     cv::Mat *prepared,
                     FramePinholeCamera *preparedCamera,
                     std::string *errorMessage)
{
    cv::Mat unused_valid_mask;
    return prepareMvsImageAndMask(source,
                                  cv::Mat(),
                                  sourceCamera,
                                  prepared,
                                  &unused_valid_mask,
                                  preparedCamera,
                                  errorMessage);
}

bool prepareMvsImageAndMask(const cv::Mat &source,
                            const cv::Mat &sourceValidMask,
                            const FramePinholeCamera &sourceCamera,
                            cv::Mat *prepared,
                            cv::Mat *preparedValidMask,
                            FramePinholeCamera *preparedCamera,
                            std::string *errorMessage)
{
    if (prepared == nullptr || preparedValidMask == nullptr || preparedCamera == nullptr)
    {
        if (errorMessage) *errorMessage = "MVS 影像预处理输出指针不能为空";
        return false;
    }
    if (source.empty())
    {
        if (errorMessage) *errorMessage = "MVS 影像预处理输入影像为空";
        return false;
    }
    if (!sourceCamera.isValid())
    {
        if (errorMessage) *errorMessage = "MVS 影像预处理相机无效";
        return false;
    }
    if (!sourceValidMask.empty() &&
        (sourceValidMask.type() != CV_8UC1 || sourceValidMask.size() != source.size()))
    {
        if (errorMessage)
        {
            *errorMessage = "MVS 有效区域蒙版必须为与影像同尺寸的 CV_8UC1";
        }
        return false;
    }

    FramePinholeCamera normalized = sourceCamera.normalizedForPositiveDepth();
    const FramePinholeCamera::Intrinsics intrinsics = normalized.intrinsics();
    const FramePinholeCamera::Distortion distortion = normalized.distortion();
    if (!(intrinsics.focalX > 0.0) || !(intrinsics.focalY > 0.0)
        || !std::isfinite(intrinsics.focalX) || !std::isfinite(intrinsics.focalY)
        || !std::isfinite(intrinsics.principalX) || !std::isfinite(intrinsics.principalY)
        || !isFiniteDistortion(distortion))
    {
        if (errorMessage) *errorMessage = "MVS 影像预处理相机包含非法内参或畸变参数";
        return false;
    }

    try
    {
        if (!hasDistortion(distortion))
        {
            *prepared = source;
            *preparedValidMask = sourceValidMask;
        }
        else
        {
            const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3)
                << intrinsics.focalX, 0.0, intrinsics.principalX,
                   0.0, intrinsics.focalY, intrinsics.principalY,
                   0.0, 0.0, 1.0);
            const cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 5)
                << distortion.radialK1,
                   distortion.radialK2,
                   distortion.tangentialP1,
                   distortion.tangentialP2,
                   distortion.radialK3);
            cv::Mat map_x;
            cv::Mat map_y;
            cv::initUndistortRectifyMap(camera_matrix,
                                        distortion_coefficients,
                                        cv::Mat(),
                                        camera_matrix,
                                        source.size(),
                                        CV_32FC1,
                                        map_x,
                                        map_y);
            cv::remap(source,
                      *prepared,
                      map_x,
                      map_y,
                      cv::INTER_LINEAR,
                      cv::BORDER_CONSTANT);
            if (!sourceValidMask.empty())
            {
                cv::remap(sourceValidMask,
                          *preparedValidMask,
                          map_x,
                          map_y,
                          cv::INTER_NEAREST,
                          cv::BORDER_CONSTANT,
                          cv::Scalar(0));

                // INTER_NEAREST can round a slightly out-of-range coordinate
                // back onto an edge pixel. Keep the geometric contract strict:
                // every destination pixel whose source coordinate is outside
                // the decoded image is invalid regardless of interpolation.
                for (int row = 0; row < map_x.rows; ++row)
                {
                    const float *map_x_row = map_x.ptr<float>(row);
                    const float *map_y_row = map_y.ptr<float>(row);
                    std::uint8_t *mask_row = preparedValidMask->ptr<std::uint8_t>(row);
                    for (int column = 0; column < map_x.cols; ++column)
                    {
                        if (!std::isfinite(map_x_row[column]) ||
                            !std::isfinite(map_y_row[column]) ||
                            map_x_row[column] < 0.0f ||
                            map_x_row[column] > static_cast<float>(source.cols - 1) ||
                            map_y_row[column] < 0.0f ||
                            map_y_row[column] > static_cast<float>(source.rows - 1))
                        {
                            mask_row[column] = 0;
                        }
                    }
                }
            }
            else
            {
                preparedValidMask->release();
            }
        }
    }
    catch (const cv::Exception &exception)
    {
        if (errorMessage)
        {
            *errorMessage = std::string("MVS 影像去畸变失败: ") + exception.what();
        }
        return false;
    }

    normalized.setDistortion(FramePinholeCamera::Distortion{});
    *preparedCamera = normalized;
    if (errorMessage) errorMessage->clear();
    return true;
}

bool saveMvsPreparedRasterArtifact(
    const std::string &inputRasterPath,
    const FramePinholeCamera &inputCamera,
    const cv::Mat &preparedValidMask,
    const std::string &workspaceDirectory,
    int frameIndex,
    MvsPreparedRasterArtifact *artifact,
    std::string *errorMessage)
{
    if (!artifact || frameIndex < 0 || workspaceDirectory.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "MVS prepared raster 输出参数、workspace 或帧下标无效";
        }
        return false;
    }

    const cv::Mat source_color = xjw::common::io::readImage(
        inputRasterPath, cv::IMREAD_COLOR);
    if (source_color.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "无法读取 MVS prepared raster 输入彩色影像: " +
                inputRasterPath;
        }
        return false;
    }

    cv::Mat prepared_color;
    FramePinholeCamera prepared_camera;
    std::string preparation_error;
    if (!prepareMvsImage(source_color,
                         inputCamera,
                         &prepared_color,
                         &prepared_camera,
                         &preparation_error))
    {
        if (errorMessage)
        {
            *errorMessage = "准备 MVS workspace 彩色栅格失败: " +
                preparation_error;
        }
        return false;
    }

    cv::Mat valid_mask;
    if (preparedValidMask.empty())
    {
        valid_mask = cv::Mat(
            prepared_color.size(), CV_8UC1, cv::Scalar(255));
    }
    else
    {
        if (preparedValidMask.type() != CV_8UC1 ||
            preparedValidMask.size() != prepared_color.size())
        {
            if (errorMessage)
            {
                *errorMessage =
                    "MVS prepared valid mask 必须与彩色工作栅格同尺寸且为 CV_8UC1";
            }
            return false;
        }
        cv::threshold(
            preparedValidMask, valid_mask, 0.0, 255.0, cv::THRESH_BINARY);
    }

    QDir workspace_directory(
        xjw::common::io::fromUtf8Path(workspaceDirectory));
    const QString prepared_directory_name = QStringLiteral("prepared_images");
    if (!workspace_directory.mkpath(prepared_directory_name))
    {
        if (errorMessage)
        {
            *errorMessage = "无法创建 MVS prepared raster 目录: " +
                xjw::common::io::toUtf8Path(
                    workspace_directory.filePath(prepared_directory_name));
        }
        return false;
    }
    const QDir prepared_directory(
        workspace_directory.filePath(prepared_directory_name));
    const QString stem = QStringLiteral("frame_%1").arg(
        frameIndex, 6, 10, QLatin1Char('0'));
    const QString image_path = prepared_directory.filePath(
        stem + QStringLiteral(".png"));
    const QString valid_mask_path = prepared_directory.filePath(
        stem + QStringLiteral("_valid.png"));
    if (!writePngAtomic(image_path, prepared_color, errorMessage) ||
        !writePngAtomic(valid_mask_path, valid_mask, errorMessage))
    {
        return false;
    }

    prepared_camera.setImageSize(CameraImageSize{
        prepared_color.cols,
        prepared_color.rows});
    artifact->imagePath = xjw::common::io::toUtf8Path(image_path);
    artifact->validMaskPath = xjw::common::io::toUtf8Path(valid_mask_path);
    artifact->camera = prepared_camera;
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

} // namespace mvs
} // namespace xjw

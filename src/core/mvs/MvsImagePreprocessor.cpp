#include "MvsImagePreprocessor.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>

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

} // namespace mvs
} // namespace xjw

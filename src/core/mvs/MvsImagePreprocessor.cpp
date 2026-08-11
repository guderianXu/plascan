#include "MvsImagePreprocessor.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

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

bool prepareMvsImage(const cv::Mat &source,
                     const FramePinholeCamera &sourceCamera,
                     cv::Mat *prepared,
                     FramePinholeCamera *preparedCamera,
                     std::string *errorMessage)
{
    if (prepared == nullptr || preparedCamera == nullptr)
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

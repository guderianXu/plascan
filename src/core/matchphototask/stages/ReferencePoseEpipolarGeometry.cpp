#include "ReferencePoseEpipolarGeometry.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::matchphotos
{
    namespace
    {

        cv::Matx33d matrix3x3(const std::array<double, 9>& values)
        {
            return cv::Matx33d(
                values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]);
        }

        bool hasNegligibleDistortion(const FramePinholeCamera& camera)
        {
            const FramePinholeCamera::Distortion distortion = camera.distortion();
            const double maximum = std::max({std::abs(distortion.radialK1),
                                             std::abs(distortion.radialK2),
                                             std::abs(distortion.radialK3),
                                             std::abs(distortion.tangentialP1),
                                             std::abs(distortion.tangentialP2)});
            // 原始 SIFT 坐标仍位于畸变像素域。明显畸变时极线是曲线，不能用直线 F
            // 近似约束，否则会在画面边缘系统性漏配。
            return std::isfinite(maximum) && maximum <= 1.0e-10;
        }

        bool validRotation(const cv::Matx33d& rotation)
        {
            const cv::Matx33d identity = rotation.t() * rotation;
            double error = 0.0;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    const double expected = row == column ? 1.0 : 0.0;
                    error += std::abs(identity(row, column) - expected);
                }
            }
            return std::isfinite(error) && error <= 1.0e-4 && std::abs(cv::determinant(rotation) - 1.0) <= 1.0e-4;
        }

    } // namespace

    ReferencePoseEpipolarGeometry fundamentalFromReferenceCameras(const FramePinholeCamera& camera0,
                                                                  const FramePinholeCamera& camera1)
    {
        ReferencePoseEpipolarGeometry result;
        if (!camera0.isValid() || !camera1.isValid() || !hasNegligibleDistortion(camera0) ||
            !hasNegligibleDistortion(camera1))
        {
            return result;
        }

        const FramePinholeCamera normalized0 = camera0.normalizedForPositiveDepth();
        const FramePinholeCamera normalized1 = camera1.normalizedForPositiveDepth();
        const FramePinholeCamera::Intrinsics intrinsics0 = normalized0.intrinsics();
        const FramePinholeCamera::Intrinsics intrinsics1 = normalized1.intrinsics();
        const bool validIntrinsics = std::isfinite(intrinsics0.focalX) && intrinsics0.focalX > 0.0 &&
                                     std::isfinite(intrinsics0.focalY) && intrinsics0.focalY > 0.0 &&
                                     std::isfinite(intrinsics0.principalX) && std::isfinite(intrinsics0.principalY) &&
                                     std::isfinite(intrinsics1.focalX) && intrinsics1.focalX > 0.0 &&
                                     std::isfinite(intrinsics1.focalY) && intrinsics1.focalY > 0.0 &&
                                     std::isfinite(intrinsics1.principalX) && std::isfinite(intrinsics1.principalY);
        if (!validIntrinsics)
        {
            return result;
        }

        const cv::Matx33d rotation0CameraToWorld = matrix3x3(normalized0.cameraToWorldRotation());
        const cv::Matx33d rotation1WorldToCamera = matrix3x3(normalized1.worldToCameraRotation());
        if (!validRotation(rotation0CameraToWorld) || !validRotation(rotation1WorldToCamera))
        {
            return result;
        }

        const std::array<double, 3> center0 = normalized0.cameraCenter();
        const std::array<double, 3> center1 = normalized1.cameraCenter();
        const cv::Vec3d centerDelta(center0[0] - center1[0], center0[1] - center1[1], center0[2] - center1[2]);
        result.baseline = cv::norm(centerDelta);
        if (!std::isfinite(result.baseline) || result.baseline <= 1.0e-12)
        {
            return result;
        }

        const cv::Matx33d relativeRotation = rotation1WorldToCamera * rotation0CameraToWorld;
        const cv::Vec3d relativeTranslation = rotation1WorldToCamera * centerDelta;
        const cv::Matx33d translationCross(0.0,
                                           -relativeTranslation[2],
                                           relativeTranslation[1],
                                           relativeTranslation[2],
                                           0.0,
                                           -relativeTranslation[0],
                                           -relativeTranslation[1],
                                           relativeTranslation[0],
                                           0.0);
        const cv::Matx33d essential = translationCross * relativeRotation;
        const cv::Matx33d intrinsic0(intrinsics0.focalX,
                                     0.0,
                                     intrinsics0.principalX,
                                     0.0,
                                     intrinsics0.focalY,
                                     intrinsics0.principalY,
                                     0.0,
                                     0.0,
                                     1.0);
        const cv::Matx33d intrinsic1(intrinsics1.focalX,
                                     0.0,
                                     intrinsics1.principalX,
                                     0.0,
                                     intrinsics1.focalY,
                                     intrinsics1.principalY,
                                     0.0,
                                     0.0,
                                     1.0);
        cv::Matx33d fundamental = intrinsic1.inv(cv::DECOMP_SVD).t() * essential * intrinsic0.inv(cv::DECOMP_SVD);

        double squaredNorm = 0.0;
        for (const double value : fundamental.val)
        {
            squaredNorm += value * value;
        }
        const double norm = std::sqrt(squaredNorm);
        if (!std::isfinite(norm) || norm <= 1.0e-15)
        {
            return result;
        }

        for (int index = 0; index < 9; ++index)
        {
            result.fundamental[static_cast<std::size_t>(index)] = fundamental.val[index] / norm;
        }
        result.valid = true;
        return result;
    }

    double epipolarSampsonDistance(const std::array<double, 9>& fundamental, double x0, double y0, double x1, double y1)
    {
        const double fx0x = fundamental[0] * x0 + fundamental[1] * y0 + fundamental[2];
        const double fx0y = fundamental[3] * x0 + fundamental[4] * y0 + fundamental[5];
        const double fx0z = fundamental[6] * x0 + fundamental[7] * y0 + fundamental[8];
        const double ftx1x = fundamental[0] * x1 + fundamental[3] * y1 + fundamental[6];
        const double ftx1y = fundamental[1] * x1 + fundamental[4] * y1 + fundamental[7];
        const double numerator = x1 * fx0x + y1 * fx0y + fx0z;
        const double denominator = fx0x * fx0x + fx0y * fx0y + ftx1x * ftx1x + ftx1y * ftx1y;
        if (!std::isfinite(denominator) || denominator <= 1.0e-15)
        {
            return std::numeric_limits<double>::infinity();
        }
        return std::sqrt((numerator * numerator) / denominator);
    }

} // namespace xjw::matchphotos

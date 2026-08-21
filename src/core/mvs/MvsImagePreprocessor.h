#pragma once

#include "camera/FramePinholeCamera.h"
#include "MvsTypes.h"

#include <opencv2/core.hpp>

#include <string>

namespace xjw
{
namespace mvs
{

struct MvsPreparedRasterArtifact
{
    std::string imagePath;
    std::string validMaskPath;
    FramePinholeCamera camera;
};

/// Returns true when MVS preparation must allocate a distinct undistorted image.
/// The result depends only on camera metadata and is safe to use before decoding.
bool mvsImagePreparationRequiresDistinctPixels(const FramePinholeCamera &camera) noexcept;

/// Applies the same local photometric normalization to every view in an
/// orbital capture. Mixing enhanced dark frames with untouched bright frames
/// creates a non-linear appearance discontinuity inside one PatchMatch batch.
cv::Mat normalizeMvsPhotometry(const cv::Mat &source,
                               MvsSceneProfile sceneProfile);

/**
 * @brief 将原始影像转换为 MVS 使用的正深度、无畸变针孔影像。
 * @param source 原始影像，支持 OpenCV remap 接受的任意通道数。
 * @param sourceCamera 与 source 像素严格对应的完整 FramePinholeCamera。
 * @param prepared 输出影像；无畸变时允许与 source 共享像素存储。
 * @param preparedCamera 输出与 prepared 对应的正深度、零畸变 FramePinholeCamera。
 * @param errorMessage 失败原因，可为空。
 * @return 输入有效且影像转换成功时返回 true。
 */
bool prepareMvsImage(const cv::Mat &source,
                     const FramePinholeCamera &sourceCamera,
                     cv::Mat *prepared,
                     FramePinholeCamera *preparedCamera,
                     std::string *errorMessage = nullptr);

/**
 * @brief 使用同一去畸变映射准备 MVS 影像及其有效区域蒙版。
 * @param source 原始影像。
 * @param sourceValidMask 与 source 同尺寸的 CV_8UC1 蒙版；允许为空。
 * @param sourceCamera 与 source 像素严格对应的完整 FramePinholeCamera。
 * @param prepared 输出无畸变影像。
 * @param preparedValidMask 输出无畸变蒙版；越界映射固定为无效。
 * @param preparedCamera 输出与 prepared 对应的正深度、零畸变相机。
 * @param errorMessage 失败原因，可为空。
 */
bool prepareMvsImageAndMask(const cv::Mat &source,
                            const cv::Mat &sourceValidMask,
                            const FramePinholeCamera &sourceCamera,
                            cv::Mat *prepared,
                            cv::Mat *preparedValidMask,
                            FramePinholeCamera *preparedCamera,
                            std::string *errorMessage = nullptr);

/**
 * @brief 将 MVS 实际使用的全分辨率彩色工作栅格和有效蒙版持久化到 workspace。
 *
 * inputRasterPath 与 inputCamera 必须严格对应。preparedValidMask 已位于
 * prepareMvsImage 产出的工作栅格中，非零表示有效；为空时保存全有效蒙版。
 * 输出使用无损 PNG，并返回与输出彩色栅格严格对应的零畸变相机。
 */
bool saveMvsPreparedRasterArtifact(
    const std::string &inputRasterPath,
    const FramePinholeCamera &inputCamera,
    const cv::Mat &preparedValidMask,
    const std::string &workspaceDirectory,
    int frameIndex,
    MvsPreparedRasterArtifact *artifact,
    std::string *errorMessage = nullptr);

} // namespace mvs
} // namespace xjw

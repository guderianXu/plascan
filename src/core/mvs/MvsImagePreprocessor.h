#pragma once

#include "camera/FramePinholeCamera.h"

#include <opencv2/core.hpp>

#include <string>

namespace xjw
{
namespace mvs
{

/// Returns true when MVS preparation must allocate a distinct undistorted image.
/// The result depends only on camera metadata and is safe to use before decoding.
bool mvsImagePreparationRequiresDistinctPixels(const FramePinholeCamera &camera) noexcept;

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

} // namespace mvs
} // namespace xjw

#pragma once

#include "MvsTypes.h"

#include <QString>

#include <vector>

namespace xjw::gui::project
{

// 为深度图生成准备 CPU 侧稀疏点云。该服务不依赖 QWidget 或项目状态。
xjw::mvs::SparseCloud prepareDenseSparseCloud(const QString &sparseCloudPath,
                                               const std::vector<xjw::mvs::CameraView> &views);

} // namespace xjw::gui::project

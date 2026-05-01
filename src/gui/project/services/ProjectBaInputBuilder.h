#pragma once

#include "BaInputBuilder.h"

namespace xjw::gui::project {

using BaInputBuildStatus = xjw::core::project::BaInputBuildStatus;
using BaInputBuildResult = xjw::core::project::BaInputBuildResult;

// 从项目 metadata 中构建 BA 所需输入：
// 1) 解析所选影像相机；2) 读取 sidecar 匹配点；3) 前方交汇初始化三维点。
using xjw::core::project::buildBaInputFromMeta;

} // namespace xjw::gui::project

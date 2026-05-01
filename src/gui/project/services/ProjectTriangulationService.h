// =============================================================================
// 文件名: ProjectTriangulationService.h
// 描述:   初始稀疏点云三角化服务。
//
//         本服务复用项目中的相机参数与匹配 sidecar，构建可导出的初始稀疏点云，
//         供“重建→稀疏重建→三角化生成初始稀疏点云”菜单直接调用。
// =============================================================================
#pragma once

#include "TriangulationService.h"

namespace xjw::gui::project {

using TriangulationServiceOptions = xjw::core::project::TriangulationServiceOptions;
using TriangulationServiceResult = xjw::core::project::TriangulationServiceResult;

class ProjectTriangulationService
{
public:
    static TriangulationServiceResult run(const QJsonObject &meta,
                                          const QStringList &selectedImages,
                                          const TriangulationServiceOptions &options)
    {
        return xjw::core::project::TriangulationService::run(meta,
                                                              selectedImages,
                                                              options);
    }
};

} // namespace xjw::gui::project
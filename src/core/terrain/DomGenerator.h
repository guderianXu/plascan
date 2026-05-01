#pragma once

#include "DemDomTypes.h"

#include <QString>
#include <QStringList>

#include <opencv2/core.hpp>

namespace xjw
{

/**
 * @brief DOM 核心生成器。
 *
 * 负责结合 DEM 有效区与多张输入影像，生成统一尺寸的 DOM 图像。
 */
class DomGenerator
{
public:
    /**
     * @brief 从 DEM 与影像集合生成 DOM 图像。
     *
     * 采用多影像加权叠合（锐度权重 + 曝光补偿），
     * 最终以 DEM 有效区 validMask 裁剪输出。
     */
    static bool generateFromImages(const DemGridData &demGrid,
                                   const QStringList &images,
                                   const DomGenerationOptions &options,
                                   cv::Mat *domImage,
                                   QString *errorMsg = nullptr);

    /**
     * @brief 从带纹理 OBJ 网格生成 DOM 图像。
     *
     * 对每个三角面在 XY 栅格上做扫描线光栅化，
     * 以重心坐标插值 UV，然后双线性采样纹理图像，
     * 输出与 demGrid 等尺寸的 BGR 彩色图像。
     *
     * 若 domImage 已分配内存（非空），则直接绘制到现有图像上（支持多瓦片拼接）；
     * 否则自动分配与 demGrid 等大的图像。
     */
    static bool generateFromTexturedMesh(const TerrainMeshInput &input,
                                         const DemGridData &demGrid,
                                         cv::Mat *domImage,
                                         QString *errorMsg = nullptr);
};

} // namespace xjw

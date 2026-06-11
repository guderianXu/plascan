#pragma once
// =============================================================================
// 文件: SparseCloudPreprocessor.h
// 模块: MVS - 稀疏点云预处理
// 说明:
//   读取稀疏点云文件（.xyz / .ply），过滤离群值，计算包围盒，
//   为 DepthMapGenerator 的 SparseCloud 提供来源。
// =============================================================================

#include "MvsTypes.h"
#include <string>
#include <vector>
#include <array>

#include <plapoint/filters/preprocessing.h>

namespace xjw
{
namespace mvs
{

struct PreprocessOptions
{
    float reprojectThresh = 2.0f;    ///< 重投影误差阈值（像素），超过此值的点被过滤
    float outlierRadius   = 1.0f;    ///< 离群值搜索半径
    int   minNeighbors    = 3;       ///< 半径内最少邻居数（否则视为离群）
    int   minPoints       = 500;     ///< 过滤后最少有效点数
};

struct PreprocessResult
{
    int rawCount      = 0;    ///< 原始点数
    int filteredCount = 0;    ///< 过滤后有效点数
    bool tooFewPoints = false;///< 是否因点数过少导致失败

    std::array<float,3> minPt = {0,0,0}; ///< AABB 最小值
    std::array<float,3> maxPt = {0,0,0}; ///< AABB 最大值

    SparseCloud cloud;        ///< 处理后的稀疏点云（用于 DepthMapGenerator）
};

class SparseCloudPreprocessor
{
public:
    explicit SparseCloudPreprocessor(
        plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto)
        : _processingDevice(processingDevice)
    {
    }

    /// 从文件加载稀疏点云并预处理
    /// @param cloudPath  .xyz 或 .ply 文件路径
    /// @param views      相机视图列表（用于过滤视野外的点）
    /// @param result     输出预处理结果
    /// @param errorMsg   出错时填充
    /// @return 成功返回 true
    bool run(const std::string              &cloudPath,
             const std::vector<CameraView>  &views,
             PreprocessResult               &result,
             std::string                    *errorMsg = nullptr) const;

private:
    /// 尝试读取 .xyz 文件（每行 x y z 或 x y z r g b）
    static bool loadXYZ(const std::string &path,
                        std::vector<std::array<float,3>> &pts,
                        std::string *err);

    /// 统计离群值并过滤
    static void filterOutliers(std::vector<std::array<float,3>> &pts,
                               float radius,
                               int minNeigh,
                               plapoint::ProcessingDevice processingDevice);

    plapoint::ProcessingDevice _processingDevice = plapoint::ProcessingDevice::Auto;
};

} // namespace mvs
} // namespace xjw

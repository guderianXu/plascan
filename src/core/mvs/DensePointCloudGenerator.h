#pragma once
// =============================================================================
// 文件: DensePointCloudGenerator.h
// 模块: MVS - 稠密点云生成（向后兼容 GUI 接口）
// 说明:
//   包装 DenseCloudBuilder / DensePointCloudCUDA，提供 GUI 层使用的
//   旧式接口。简化后移除了 KDTree 复杂后处理。
// =============================================================================

#include "MvsTypes.h"
#include "DenseCloudBuilder.h"
#include <opencv2/core.hpp>
#include <QVector>
#include <QSharedPointer>
#include <string>
#include <array>
#include <vector>

namespace xjw {
namespace mvs {

/// 稠密点云生成配置
struct DenseCloudGenConfig {
    bool  useCudaUnprojection  = true;
    float minDepth             = 0.01f;
    float maxDepth             = 1e6f;
    int   subsample            = 1;

    // AABB 裁剪
    bool  clipToSparseAabb     = false;
    std::array<float,3> sparseMinPt = {0,0,0};
    std::array<float,3> sparseMaxPt = {0,0,0};
    float aabbPaddingRatio     = 0.1f;

    // 体素降采样
    float voxelSize            = 0.0f;  ///< 0 = 不做
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto;

    // 统计过滤
    bool  doStatisticalFilter  = false;
    float statStdMul           = 2.0f;
    int   statKnn              = 20;
};

/// 反投影输入
struct UnprojectInput {
    QSharedPointer<cv::Mat> fusedDepth;
    QSharedPointer<cv::Mat> validMask;
    std::vector<int>        colorFrameIndices;
    bool                    depthFlippedZ = false;
};

/// 稠密点云生成器
class DensePointCloudGenerator
{
public:
    DensePointCloudGenerator(const std::vector<CameraView> &views,
                              const DenseCloudGenConfig     &config = DenseCloudGenConfig{});
    ~DensePointCloudGenerator() = default;

    /// 执行反投影
    bool generate(const UnprojectInput       &input,
                  int                         frameIdx,
                  QVector<DensePoint>        &cloud,
                  void                       *unused,
                  std::string                *errorMsg = nullptr);

    /// 后处理（体素降采样）
    void postProcess(QVector<DensePoint> &cloud);

    /// 保存为 XYZ 文本格式
    static bool saveToXyz(const QVector<DensePoint> &cloud,
                          const std::string         &path,
                          std::string               *errorMsg = nullptr);

    /// 保存为 PLY 二进制格式
    static bool saveToPly(const QVector<DensePoint> &cloud,
                          const std::string         &path,
                          std::string               *errorMsg = nullptr);

private:
    std::vector<CameraView>  m_views;
    DenseCloudGenConfig      m_config;
};

} // namespace mvs
} // namespace xjw

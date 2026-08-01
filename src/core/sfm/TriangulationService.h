#pragma once

/**
 * @file TriangulationService.h
 * @brief 从工程相机和已有匹配轨迹独立生成初始稀疏点云。
 *
 * 这是 project 层的无状态服务，不执行增量相机注册或 BA。它适用于相机位姿已经
 * 可用、仅需重新三角化/过滤连接点的工作流，并负责把结果 PLY 和统计写到输出目录。
 */

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::core::project
{

struct TriangulationServiceOptions
{
    QString outputDir; ///< 稀疏点云和结果记录目录，不能为空。
    double minTriAngleDeg = 2.0; ///< 轨迹允许的最小保守交会角，度。
    double maxReprojErrorPx = 2.0; ///< 多视 RMS 像素上限。
    int minObservations = 2; ///< 进入三角化前的最少有效观测数。
    bool ignoreTwoViewTracks = false; ///< true 时要求至少三视观测。
    int minTrackLength = 2; ///< 对原始轨迹长度的额外下限。
};

struct TriangulationServiceResult
{
    bool success = false; ///< PLY 与 resultJson 都成功生成。
    QString errorMessage; ///< 输入、三角化或文件写出失败原因。
    QString sparseCloudPath; ///< 成功时的绝对 PLY 路径。
    int exportedPointCount = 0; ///< 通过所有门控并写出的点数。
    int candidateTrackCount = 0; ///< 从匹配构建出的候选轨迹数。
    int rejectedByObservationCount = 0; ///< 观测/轨迹长度不足数。
    int rejectedByTriAngleCount = 0; ///< 交会角不足数。
    int rejectedByReprojCount = 0; ///< 重投影 RMS 超限数。
    QJsonObject resultJson; ///< 可直接合并进工程记录的结构化结果。
};

/// 已知相机三角化服务入口。
class TriangulationService
{
public:
    /**
     * @brief 读取选中影像对应相机与匹配，三角化并写出稀疏点云。
     *
     * selectedImages 的顺序定义相机索引映射；工程中无法解析到该集合的历史匹配
     * 会被忽略。函数同步执行，GUI 必须在 worker 中调用。
     */
    static TriangulationServiceResult run(const QJsonObject &meta,
                                          const QStringList &selectedImages,
                                          const TriangulationServiceOptions &options);
};

} // namespace xjw::core::project

#pragma once

/**
 * @file SfmAttemptRunner.h
 * @brief 一次确定内参初始化/初始对配置下的正式 SfM 尝试。
 *
 * Runner 只消费已经落盘的多视连接点图，将其转换为 core/sfm::IncrementalSfm
 * 输入并收集内存重建。焦距候选搜索、结果择优和文件写出由 Pipeline 负责。
 */

#include "common/SfmTypes.h"
#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"

#include <QMap>
#include <QSize>
#include <QString>
#include <QStringList>

#include <memory>
#include <vector>

namespace xjw
{
class SfmReconstruction;
}

namespace xjw::aerial_triangulation
{

/// 当前连接点图中的一个有向语义 pair；matches 的 idx1/idx2 分别属于 A/B。
struct PreparedTiePointMatchPair
{
    ImageId imageA = kInvalidImageId; ///< imagePaths 下标。
    ImageId imageB = kInvalidImageId; ///< imagePaths 下标。
    std::vector<FeatureMatch> matches; ///< 已通过前端几何验证的索引匹配。
};

/// 从 tie point 文件一次性解析出的稳定 SfM 输入图。
struct PreparedTiePointGraph
{
    QStringList imagePaths; ///< 与请求影像集合一致的稳定 ImageId 顺序。
    QMap<ImageId, std::vector<FeatureKeypoint>> keypointsByImage; ///< 每图完整关键点坐标。
    std::vector<PreparedTiePointMatchPair> matchPairs; ///< pairwise 边。
    int trackCount = 0; ///< 前端已整理的多视轨迹总数。
};

/// 单次尝试结果，同时保留后续质量报告/写出所需的内存对象。
struct SfmAttemptExecutionResult
{
    AerialTriangulationReconstructionResult result; ///< 数值结果和诊断，尚未保证已写盘。
    std::shared_ptr<xjw::SfmReconstruction> reconstruction; ///< 成功/部分成功模型。
    std::shared_ptr<const PreparedTiePointGraph> graph; ///< 本次实际消费的共享只读连接点图。
};

// 单次 SfM 尝试只消费 matchphototask 落盘的多视图连接点，不读取描述子，
// 也不具备特征提取或影像匹配能力。
class SfmAttemptRunner
{
public:
    /**
     * @brief 读取连接点、构造相机先验并运行一次 IncrementalSfm。
     *
     * input.estimatedFocalScale 在无可信内参时生成初始像素焦距；是否释放共享焦距
     * 由 input.adaptiveCameraModelFitting 控制。
     */
    SfmAttemptExecutionResult run(const PreparedAerialTriangulationInput &input) const;

    /// 解析并校验多视连接点文件，selectedImages 定义允许集合和 ImageId 顺序。
    static bool readTiePointGraph(const QString &tiePointPath,
                                  const QStringList &selectedImages,
                                  PreparedTiePointGraph *graph,
                                  QString *errorMessage);

    // 影像尺寸只能来自真实文件头，不能由受蒙版裁剪后的关键点包围盒推断；
    // 否则主点和焦距尺度会随蒙版内容漂移。
    static QSize resolveInputImageSize(const QString &imagePath);
};

} // namespace xjw::aerial_triangulation

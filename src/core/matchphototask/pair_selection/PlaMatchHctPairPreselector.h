#pragma once

#include "FramePinholeCamera.h"
#include "MatchPhotosOptions.h"
#include "PairTypes.h"
#include "sift/SiftBackendType.h"

#include <QMap>
#include <QString>
#include <QStringList>

#include <atomic>
#include <array>

namespace xjw::matchphotos
{

    class MatchPhotosFeatureCache;

    struct PlaMatchHctPairPreselectionStats
    {
        int coarseCandidateCount = 0;
        int genericSelectedCount = 0;
        int referenceSelectedCount = 0;
        int finalSelectedCount = 0;
        bool usedReferenceIndexFallback = false;
        bool usedAllPairsFallback = false;
        bool usedDescriptorAdapter = false;
        QString detail;
    };

    /**
     * @brief 按“对齐照片”的通用/参考预选顺序生成 PlaMatch-HCT 候选。
     *
     * PlaMatch 直接使用原生 coarse 特征；其它算法复用正式浮点描述子生成 coarse 视图。
     * 两者都执行 HCTree、局部一致性和多轮森林削减；参考预选随后按
     * Source/Estimated/Sequential 语义做集合并集。
     * 显式手工像对由 MatchPhotosTask 在调用本类之前处理。
     */
    class PlaMatchHctPairPreselector
    {
    public:
        static bool select(const QStringList& images,
                           const MatchPhotosFeatureCache& featureCache,
                           const MatchPhotosOptions& options,
                           const QMap<QString, FramePinholeCamera>& referenceCameras,
                           image_matching::SiftComputeBackend backend,
                           int deviceIndex,
                           PairSelectionResult* output,
                           PlaMatchHctPairPreselectionStats* stats = nullptr,
                           std::atomic_bool* cancelFlag = nullptr,
                           QString* errorMessage = nullptr);

        static bool selectWithPositions(const QStringList& images,
                                        const MatchPhotosFeatureCache& featureCache,
                                        const MatchPhotosOptions& options,
                                        const QMap<QString, FramePinholeCamera>& referenceCameras,
                                        const QMap<QString, std::array<double, 3>>& referencePositions,
                                        image_matching::SiftComputeBackend backend,
                                        int deviceIndex,
                                        PairSelectionResult* output,
                                        PlaMatchHctPairPreselectionStats* stats = nullptr,
                                        std::atomic_bool* cancelFlag = nullptr,
                                        QString* errorMessage = nullptr);
    };

} // namespace xjw::matchphotos

#pragma once

#include "common/SfmTypes.h"
#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"

#include <QMap>
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

struct PreparedTiePointMatchPair
{
    ImageId imageA = kInvalidImageId;
    ImageId imageB = kInvalidImageId;
    std::vector<FeatureMatch> matches;
};

struct PreparedTiePointGraph
{
    QStringList imagePaths;
    QMap<ImageId, std::vector<FeatureKeypoint>> keypointsByImage;
    std::vector<PreparedTiePointMatchPair> matchPairs;
    int trackCount = 0;
};

struct SfmAttemptExecutionResult
{
    AerialTriangulationReconstructionResult result;
    std::shared_ptr<xjw::SfmReconstruction> reconstruction;
    PreparedTiePointGraph graph;
};

// 单次 SfM 尝试只消费 matchphototask 落盘的多视图连接点，不读取描述子，
// 也不具备特征提取或影像匹配能力。
class SfmAttemptRunner
{
public:
    SfmAttemptExecutionResult run(const PreparedAerialTriangulationInput &input) const;

    static bool readTiePointGraph(const QString &tiePointPath,
                                  const QStringList &selectedImages,
                                  PreparedTiePointGraph *graph,
                                  QString *errorMessage);
};

} // namespace xjw::aerial_triangulation

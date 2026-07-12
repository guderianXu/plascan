#pragma once

#include "AerialTriangulationService.h"
#include "matchphototask/task/MatchPhotosTask.h"

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>
#include <memory>

namespace xjw::gui
{

// Metashape 的“对齐照片”对应这里的空中三角测量工作流：
// 输入是影像、相机先验和连接点/匹配设置，输出是已定向相机、BA 质量和正式稀疏观测成果。
// 下游 MVS、网格、DEM/DOM 不属于本模块职责。
struct AerialTriangulationWorkflowOptions
{
    QStringList images;
    QStringList cameraPaths;
    QString projectPath;
    QString outputDir;
    QJsonObject projectMeta;

    QString quality = QStringLiteral("high");
    bool genericPreselection = true;
    bool referencePreselection = false;
    QString referenceMode = QStringLiteral("source_code");
    bool resetAlignment = true;
    bool saveAfterEachStep = false;

    int keypointLimit = 40000;
    int tiepointLimit = 4000;
    QString maskApplyMode = QStringLiteral("none");
    bool excludeFixedTiePoints = true;
    bool guidedImageMatching = false;
    bool adaptiveCameraModelFitting = true;

    QString featureAlgorithm = QStringLiteral("sift");
    QString matchAlgorithm = QStringLiteral("lightglue");
    QString matchPipeline;
    QString device = QStringLiteral("auto");
    int threads = 8;
    bool autoGenerateMissingMatches = false;
    bool restrictPairs = false;
    QStringList allowedPairs;
    QString assetsDir;
    QString featureDir;
    QString matchDir;
    QMap<QString, QString> maskPaths;
    QMap<QString, Camera> referenceCameras;
    float featureGrayscaleMin = 5.0f / 255.0f;
    float featureGrayscaleMax = 1.0f;

    std::shared_ptr<std::atomic<bool>> cancelFlag;
    std::function<void(const QString &stage, int percent)> progressFn;
    std::function<void(const QString &img0, const QString &img1, const QString &matchPath, int numMatches)> pairMatchedFn;
};

struct AerialTriangulationResolvedConfig
{
    // 已解析为可直接传入空三服务的算法级配置。
    AerialTriangulationServiceOptions serviceOptions;
    matchphotos::MatchPhotosOptions tiePointOptions;
    matchphotos::MatchPhotosContext tiePointContext;
    bool prepareTiePoints = false;
    bool forceRebuildTiePoints = false;
    QJsonObject resolvedSettings;
};

struct AerialTriangulationWorkflowResult
{
    AerialTriangulationResolvedConfig config;
    AerialTriangulationServiceResult serviceResult;
    bool tiePointPreparationExecuted = false;
    matchphotos::MatchPhotosResult tiePointResult;
};

class AerialTriangulationWorkflow
{
public:
    using ServiceRunner =
        std::function<AerialTriangulationServiceResult(const AerialTriangulationServiceOptions &options)>;
    using TiePointRunner = std::function<matchphotos::MatchPhotosResult(
        const matchphotos::MatchPhotosOptions &options,
        const matchphotos::MatchPhotosContext &context)>;

    static AerialTriangulationResolvedConfig resolveConfig(const AerialTriangulationWorkflowOptions &options);

    static AerialTriangulationWorkflowResult run(const AerialTriangulationWorkflowOptions &options,
                                                 const ServiceRunner &runner,
                                                 const TiePointRunner &tiePointRunner = {});
};

} // namespace xjw::gui

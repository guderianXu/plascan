#pragma once

#include "SFMService.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>
#include <memory>

namespace xjw::gui
{

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
    bool adaptiveCameraModelFitting = false;

    QString featureAlgorithm = QStringLiteral("disk");
    QString matchAlgorithm = QStringLiteral("lightglue");
    QString matchPipeline;
    QString device = QStringLiteral("auto");
    int threads = 8;
    bool autoGenerateMissingMatches = false;
    bool restrictPairs = false;
    QStringList allowedPairs;
    float featureGrayscaleMin = 5.0f / 255.0f;
    float featureGrayscaleMax = 1.0f;

    std::shared_ptr<std::atomic<bool>> cancelFlag;
    std::function<void(const QString &stage, int percent)> progressFn;
    std::function<void(const QString &img0, const QString &img1, const QString &matchPath, int numMatches)> pairMatchedFn;
};

struct AerialTriangulationResolvedConfig
{
    SFMServiceOptions sfmOptions;
    QJsonObject resolvedSettings;
};

struct AerialTriangulationWorkflowResult
{
    AerialTriangulationResolvedConfig config;
    SFMServiceResult sfmResult;
};

class AerialTriangulationWorkflow
{
public:
    using SfmRunner = std::function<SFMServiceResult(const SFMServiceOptions &options)>;

    static AerialTriangulationResolvedConfig resolveConfig(const AerialTriangulationWorkflowOptions &options);

    static AerialTriangulationWorkflowResult run(const AerialTriangulationWorkflowOptions &options,
                                                const SfmRunner &runner);
};

} // namespace xjw::gui

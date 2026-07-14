#pragma once

#include "ModelGeometryComparator.h"
#include "ModelImageQualityTypes.h"

namespace xjw::qc
{

struct ModelImageQualityResult
{
    bool ok = false;
    QVector<ModelViewQuality> views;
    ModelGeometryQuality geometry;
    ReferenceGeometryQuality referenceGeometry;
    QStringList failureReasons;
    QJsonObject summary;
    QString error;
};

class ModelImageQualityEvaluator
{
public:
    ModelImageQualityResult evaluate(const ModelImageQualityOptions &options) const;

    static QVector<ModelValidationView> validationViewsFromMvsWorkspace(
        const QString &workspacePath,
        QString *error = nullptr);

    static QStringList qualityFailures(ModelSceneType sceneType,
                                       const QVector<ModelViewQuality> &views,
                                       const ModelGeometryQuality &geometry);
};

} // namespace xjw::qc

#pragma once

#include <QPointF>
#include <QString>
#include <QVector>

namespace xjw::gui::views
{

struct FeatureResidualVector
{
    QPointF observed;
    QPointF projected;
    double magnitudePx = 0.0;
};

QVector<FeatureResidualVector> loadFeatureResidualsForImage(const QString &projectPath,
                                                            const QString &imagePath,
                                                            QString *errorMessage = nullptr);

} // namespace xjw::gui::views

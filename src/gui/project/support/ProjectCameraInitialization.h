#pragma once

#include <QJsonObject>
#include <QMap>
#include <QSet>
#include <QSize>
#include <QString>
#include <QStringList>

#include <optional>

class ProjectData;

namespace xjw::gui::project {

std::optional<double> parsePossiblyFractionalNumber(const QString &text);

std::optional<double> focalPixelsFromExif(const QString &imagePath,
                                          const QSize &size,
                                          double sensorWidthMm,
                                          QString *sourceTag);

QStringList resolveInitTargets(ProjectData *projectData,
                               const QJsonObject &settings,
                               QString *errorMsg);

QSet<QString> existingCameraImages(const QJsonObject &meta);

QJsonObject withPreparedCameras(const QJsonObject &baseMeta,
                                const QMap<QString, QJsonObject> &preparedCameraByImage,
                                bool overwriteExisting);

QJsonObject makeInitializedCameraMeta(double fx,
                                      double fy,
                                      double cx,
                                      double cy,
                                      double k1,
                                      double k2,
                                      double p1,
                                      double p2,
                                      const QString &source,
                                      const QString &distortionModel,
                                      const QSize &imageSize);

} // namespace xjw::gui::project
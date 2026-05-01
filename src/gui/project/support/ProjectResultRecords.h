#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::gui::project {

QJsonObject makeDepthResultRecord(const QString &createdAt,
                                  const QString &depthPng,
                                  int gridWidth,
                                  int gridHeight,
                                  const QString &sourceSparseCloud = QString(),
                                  const QString &refImage = QString());

QJsonObject makeDenseResultRecord(const QString &createdAt,
                                  const QString &densePath,
                                  int pointCount,
                                  const QString &sourceSparseCloud = QString());

QJsonObject makeModelResultRecord(const QString &createdAt,
                                  const QString &sourceTag,
                                  const QString &modelPly,
                                  int vertexCount,
                                  int faceCount,
                                  const QString &sourceSparseCloud = QString(),
                                  const QString &sourceDenseCloud = QString(),
                                  const QString &modelDenseCloud = QString());

QJsonObject makeDemResultRecord(const QString &createdAt,
                                const QString &outputDir,
                                const QString &sourceSparseCloud,
                                const QString &demTif,
                                const QString &demType,
                                double demResolution,
                                const QString &tSrs = QString(),
                                const QStringList &images = QStringList());

QJsonObject makeOrthoResultRecord(const QString &createdAt,
                                  const QString &demPath,
                                  const QString &outputPath,
                                  int sourceImageCount,
                                  const QStringList &images,
                                  bool includeResolution = false,
                                  double resolution = 0.0);

QJsonObject makeAtResultRecord(const QString &createdAt,
                               const QString &sparseCloudPath,
                               int sparsePointCount,
                               const QStringList &selectedImages,
                               const QString &outputDir,
                               const QJsonObject &extraRecord = {});

void upsertMetaArrayRecordByPath(QJsonObject *meta,
                                 const QString &arrayKey,
                                 const QString &pathKey,
                                 const QJsonObject &record);

void replaceMetaArrayWithLatest(QJsonObject *meta,
                                const QString &arrayKey,
                                const QJsonObject &record);

void upsertMetaArrayRecordByIndex(QJsonObject *meta,
                                  const QString &arrayKey,
                                  const QJsonObject &record,
                                  int replaceIndex);

} // namespace xjw::gui::project
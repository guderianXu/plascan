#pragma once

#include "task/MatchPhotosResult.h"
#include "ProjectFilesManager.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

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
                                  double resolution = 0.0,
                                  const QJsonObject &payload = {});

QJsonObject makeAtResultRecord(const QString &createdAt,
                               const QString &sparseCloudPath,
                               int sparsePointCount,
                               const QStringList &selectedImages,
                               const QString &outputDir,
                               const QJsonObject &extraRecord = {});

/**
 * @brief 将连接点任务的逐影像分片结果转换为项目元数据记录。
 *
 * 该适配器是 GUI 写回路径的唯一入口。它不会登记任务内的临时 SIFT 特征，
 * 也不会按像对重复登记 `.pimatch`；轨迹路径和统计被附加到每个影像分片设置中。
 */
QVector<ProjectImageMatchResultRecord> makeImageMatchResultRecords(
    const xjw::matchphotos::MatchPhotosResult &result);

void upsertMetaArrayRecordByPath(QJsonObject *meta,
                                 const QString &arrayKey,
                                 const QString &pathKey,
                                 const QJsonObject &record);

} // namespace xjw::gui::project

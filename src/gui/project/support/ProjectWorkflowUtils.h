#pragma once

#include "result/OperationResult.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::gui::project 
{

using OperationResult = xjw::common::OperationResult;

struct SparsePointContext
{
    int sourceResultIndex = -1;
    QString outputDir;
    QString sparseCloudPath;
    QString sidecarPath;
    QStringList selectedImages;
};

struct SparsePointContextResult
{
    OperationResult status;
    SparsePointContext context;
};

struct SparsePointOperationResult
{
    QString outputDir;
    QString sparseCloudPath;
    QString sidecarPath;
    int inputCount = 0;
    int outputCount = 0;
    QJsonObject extraRecord;
};

struct TerrainPipelineResult
{
    bool ok = false;
    QJsonObject payload;
    QString error;
};

QString sparseOperationDisplayName(const QString &operation);

int findLatestAtResultIndex(const QJsonObject &meta,
                            const QString &operation = QString());

int findLatestProductionAtResultIndex(const QJsonObject &meta);

bool writeJsonObjectFile(const QString &path,
                         const QJsonObject &object,
                         QString *errorMessage = nullptr);

OperationResult writeJsonObjectFileResult(const QString &path,
                                          const QJsonObject &object);

TerrainPipelineResult runDemProducts(const QString &sparsePath,
                                     const QString &outputDir,
                                     double demResolution,
                                     const QString &demType,
                                     bool genPointCloud);

TerrainPipelineResult runOrthoProduct(const QStringList &images,
                                      const QString &demPath,
                                      const QString &outputPath,
                                      double resolution,
                                      const QJsonObject &projectMeta = QJsonObject());

bool resolveSparsePointContext(const QJsonObject &meta,
                               int requestedIndex,
                               SparsePointContext *context,
                               QString *errorMessage = nullptr);

SparsePointContextResult resolveSparsePointContextResult(const QJsonObject &meta,
                                                         int requestedIndex);

bool runSparsePointOutlierRemoval(const SparsePointContext &context,
                                  const QJsonObject &settings,
                                  const QString &outputDir,
                                  SparsePointOperationResult *result,
                                  QString *errorMessage);

bool runSparsePointLocalOptim(const SparsePointContext &context,
                              const QJsonObject &settings,
                              const QString &outputDir,
                              SparsePointOperationResult *result,
                              QString *errorMessage);

bool runSparsePointRefine(const SparsePointContext &context,
                          const QJsonObject &settings,
                          const QString &outputDir,
                          SparsePointOperationResult *result,
                          QString *errorMessage);

QJsonArray summarizeAtResults(const QJsonObject &meta);

} // namespace xjw::gui::project

#pragma once

#include "result/OperationResult.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>

namespace xjw::core::project
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
                                      const QJsonObject &settings,
                                      const QJsonObject &projectMeta,
                                      const std::atomic_bool *cancelFlag = nullptr,
                                      const std::function<void(const QString &, int)> &progressCallback = {});

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

} // namespace xjw::core::project

// Transitional aliases keep existing GUI consumers source-compatible while the
// workflow implementation lives behind a core target.
namespace xjw::gui::project
{
using xjw::core::project::OperationResult;
using xjw::core::project::SparsePointContext;
using xjw::core::project::SparsePointContextResult;
using xjw::core::project::SparsePointOperationResult;
using xjw::core::project::TerrainPipelineResult;
using xjw::core::project::findLatestAtResultIndex;
using xjw::core::project::findLatestProductionAtResultIndex;
using xjw::core::project::resolveSparsePointContext;
using xjw::core::project::resolveSparsePointContextResult;
using xjw::core::project::runDemProducts;
using xjw::core::project::runOrthoProduct;
using xjw::core::project::runSparsePointLocalOptim;
using xjw::core::project::runSparsePointOutlierRemoval;
using xjw::core::project::runSparsePointRefine;
using xjw::core::project::sparseOperationDisplayName;
using xjw::core::project::summarizeAtResults;
using xjw::core::project::writeJsonObjectFile;
using xjw::core::project::writeJsonObjectFileResult;
} // namespace xjw::gui::project

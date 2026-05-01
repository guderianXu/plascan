#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

namespace xjw::gui::project {

struct BundleAdjustSparseCloudExport
{
    bool exported = false;
    QString outputDir;
    QString sparseCloudPath;
    int pointCount = 0;
    QJsonObject extraRecord;
    QString errorMessage;
};

QJsonObject buildBundleAdjustReport(const QJsonObject &baResult,
                                    const QMap<QString, QJsonObject> &beforeCameras,
                                    const QMap<QString, QJsonObject> &afterCameras);

QJsonObject buildBundleAdjustWorkflowReport(const QJsonObject &baResult,
                                            const QStringList &images,
                                            const QString &outputDir,
                                            const QString &source,
                                            const QMap<QString, QJsonObject> &beforeCameras,
                                            const QMap<QString, QJsonObject> &afterCameras);

BundleAdjustSparseCloudExport exportBundleAdjustSparseCloud(const QJsonObject &baResult,
                                                            const QStringList &selectedImages,
                                                            const QString &outputDir,
                                                            bool useDedicatedFileName);

bool writeBundleAdjustReport(const QString &reportPath,
                             const QJsonObject &baResult,
                             const QMap<QString, QJsonObject> &beforeCameras,
                             const QMap<QString, QJsonObject> &afterCameras);

bool writeLatestAndAppendHistoryReport(const QString &reportsDir,
                                       const QString &latestFileName,
                                       const QString &historyFileName,
                                       const QJsonObject &report);

} // namespace xjw::gui::project

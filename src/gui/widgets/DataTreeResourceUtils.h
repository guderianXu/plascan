#pragma once

#include "WorkspaceSectionIcons.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QSet>
#include <QString>
#include <QStringList>

#include <initializer_list>

namespace xjw::gui::widgets::data_tree
{
extern const int SectionRole;
extern const int ResourcePathRole;
extern const int AggregateResourcePathsRole;
extern const int WorkspaceSectionRole;
extern const int ChunkIdRole;
extern const int ChunkDirectoryRole;
extern const int WorkspaceRootRole;

QString formattedCount(qint64 count);
QString workspaceSummaryLabel(int chunkCount, int imageCount);
QString chunkSummaryLabel(const QString &name, int imageCount, int tiePointCount);
QString workspaceSectionName(WorkspaceSection section);
QString depthRecordPrimaryPath(const QJsonObject &record);
QString depthQualityLabel(QString profile);
QString depthFilterLabel(QString mode);
bool isDisplayableMeshResult(const QJsonObject &record);
QStringList displayableMeshAssetPaths(const QJsonObject &record);
int displayableMeshResultCount(const QJsonArray &modelResults);
bool isTreeResultKey(const QString &key);
bool hasTreeResultKeys(const QJsonObject &meta);
int compareNaturalText(QString lhs, QString rhs);
QString referenceDatasetPath(const QJsonObject &record);
QString referenceDatasetTypeLabel(QString type);
QString referenceDatasetRoleLabel(QString role);
QString resultPath(const QJsonObject &record, std::initializer_list<const char *> keys);
int countObjectsWithPath(const QJsonArray &records, std::initializer_list<const char *> keys);
QString imagePathFromValue(const QJsonValue &value);
QString imagePathKey(QString path);
bool jsonArrayHasAtLeast(const QJsonValue &value, int size);
bool objectHasTrueFlag(const QJsonObject &object, std::initializer_list<const char *> keys);
bool objectHasAlignedStatus(const QJsonObject &object);
bool cameraHasPose(const QJsonObject &camera);
bool imageObjectHasAlignedPose(const QJsonObject &image);
bool imageIsAligned(const QJsonValue &image, const QSet<QString> &alignedImageKeys);
} // namespace xjw::gui::widgets::data_tree

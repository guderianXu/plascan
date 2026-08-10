#pragma once

#include "ProjectResourceCleanupPlan.h"

#include <QByteArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

class ProjectData;

namespace xjw::core::project::detail
{

enum class CleanupTransactionState
{
    Staging,
    MetadataCommitted
};

enum class CleanupMoveState
{
    Planned,
    Staged
};

struct CleanupTransactionMove
{
    QString source;
    QString destination;
    bool directory = false;
    CleanupMoveState state = CleanupMoveState::Planned;
};

struct CleanupTransactionSession
{
    QString projectPath;
    QString chunkId;
    int chunkDirectory = 0;
    QString projectRoot;
    QString managedRoot;
};

struct CleanupTransactionManifest
{
    QString transactionId;
    QString projectPath;
    QString chunkId;
    int chunkDirectory = 0;
    QString projectRoot;
    QString managedRoot;
    QJsonObject originalMetadata;
    QJsonObject updatedMetadata;
    QByteArray originalMetadataHash;
    QByteArray updatedMetadataHash;
    CleanupTransactionState state = CleanupTransactionState::Staging;
    QVector<CleanupTransactionMove> moves;
};

QString cleanupTrashBase(const QString &managedRoot);

bool createCleanupTransaction(
    const ResourceCleanupPlan &plan,
    QString *transactionRoot,
    CleanupTransactionManifest *manifest,
    QString *errorMessage);

bool appendPlannedCleanupMove(
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    const CleanupTransactionMove &move,
    QString *errorMessage);

bool markCleanupMoveStaged(
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    int moveIndex,
    QString *errorMessage);

bool markCleanupMetadataCommitted(
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    QString *errorMessage);

bool loadAndValidateCleanupTransaction(
    const CleanupTransactionSession &session,
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    QString *errorMessage);

bool restoreStagedCleanupTransaction(
    const QString &transactionRoot,
    const CleanupTransactionManifest &manifest,
    QStringList *failedPaths,
    QString *errorMessage);

bool purgeCommittedCleanupTransaction(
    const QString &transactionRoot,
    const CleanupTransactionManifest &manifest,
    QStringList *failedPaths,
    QString *errorMessage);

bool purgePendingCleanupDirectories(
    const QString &managedRoot,
    QStringList *failedPaths,
    QString *errorMessage);

void removeEmptyCleanupTrashBase(const QString &managedRoot);

} // namespace xjw::core::project::detail

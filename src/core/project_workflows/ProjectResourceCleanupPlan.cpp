#include "ProjectResourceCleanupPlan.h"

#include "ProjectResourceCleanupArtifacts.h"

#include "project/ProjectIO.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectSessionModel.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QList>
#include <QPair>
#include <QSet>

#include <algorithm>

namespace xjw::core::project::detail
{
namespace
{

struct SelectedArtifacts
{
    QStringList files;
    QStringList ownedDirectories;
    QStringList unverifiedDirectories;
};

void mergeArtifacts(const CleanupRecordArtifacts &source,
                    SelectedArtifacts *target)
{
    if (!target)
    {
        return;
    }
    for (const QString &path : source.files)
    {
        appendUniqueCleanupPath(&target->files, path);
    }
    for (const QString &path : source.ownedDirectories)
    {
        appendUniqueCleanupPath(&target->ownedDirectories, path);
    }
    for (const QString &path : source.unverifiedDirectories)
    {
        appendUniqueCleanupPath(&target->unverifiedDirectories, path);
    }
}

QList<QPair<QString, QString>> supportedResultSections()
{
    return {
        {QStringLiteral("连接点"),
         QStringLiteral("aerial_triangulation_results")},
        {QStringLiteral("深度图"), QStringLiteral("depth_map_results")},
        {QStringLiteral("稠密点云"), QStringLiteral("dense_cloud_results")},
        {QStringLiteral("3D模型"), QStringLiteral("model_results")},
        {QStringLiteral("DEM"), QStringLiteral("dem_results")},
        {QStringLiteral("正射影像"), QStringLiteral("ortho_results")}
    };
}

void collectRetainedReferences(const QJsonObject &metadata,
                               const QString &projectRoot,
                               QStringList *paths,
                               QStringList *ownedDirectories)
{
    collectCleanupStringPaths(metadata, projectRoot, paths);
    for (const auto &section : supportedResultSections())
    {
        const QJsonArray records = metadata.value(section.second).toArray();
        for (const QJsonValue &value : records)
        {
            const CleanupRecordArtifacts artifacts =
                collectCleanupRecordArtifacts(section.first,
                                              value.toObject(),
                                              projectRoot);
            for (const QString &path : artifacts.ownedDirectories)
            {
                appendUniqueCleanupPath(ownedDirectories, path);
            }
            for (const QString &path : artifacts.unverifiedDirectories)
            {
                appendUniqueCleanupPath(ownedDirectories, path);
            }
        }
    }
}

QSet<QString> pathIdentitySet(const QStringList &paths)
{
    QSet<QString> identities;
    for (const QString &path : paths)
    {
        const QString identity = cleanupPathIdentity(path);
        if (!identity.isEmpty())
        {
            identities.insert(identity);
        }
    }
    return identities;
}

bool pathCoveredByDirectory(const QString &path,
                            const QStringList &directories)
{
    return std::any_of(directories.cbegin(),
                       directories.cend(),
                       [&](const QString &directory)
                       {
                           return cleanupPathIsInside(directory, path, true);
                       });
}

bool directoriesOverlap(const QString &lhs, const QString &rhs)
{
    return cleanupPathIsInside(lhs, rhs, true)
        || cleanupPathIsInside(rhs, lhs, true);
}

bool retainedReferenceOverlapsDirectory(
    const QString &directory,
    const QStringList &retainedPaths,
    const QStringList &retainedDirectories)
{
    const bool containsReference = std::any_of(
        retainedPaths.cbegin(),
        retainedPaths.cend(),
        [&](const QString &path)
        {
            return cleanupPathIsInside(directory, path, true);
        });
    if (containsReference)
    {
        return true;
    }

    return std::any_of(retainedDirectories.cbegin(),
                       retainedDirectories.cend(),
                       [&](const QString &retainedDirectory)
                       {
                           return directoriesOverlap(directory,
                                                     retainedDirectory);
                       });
}

bool selectedArtifactIsInsideDirectory(
    const QString &directory,
    const SelectedArtifacts &selectedArtifacts)
{
    return std::any_of(selectedArtifacts.files.cbegin(),
                       selectedArtifacts.files.cend(),
                       [&](const QString &path)
                       {
                           return cleanupPathIsInside(directory, path, true);
                       });
}

bool pathIsManaged(const ResourceCleanupPlan &plan, const QString &path)
{
    return cleanupPathIsInside(plan.managedRoot, path, false);
}

void classifyUnverifiedDirectories(
    const SelectedArtifacts &selectedArtifacts,
    ResourceCleanupPlan *plan)
{
    const QSet<QString> verifiedIdentities = pathIdentitySet(
        selectedArtifacts.ownedDirectories);
    for (const QString &directory : selectedArtifacts.unverifiedDirectories)
    {
        if (verifiedIdentities.contains(cleanupPathIdentity(directory)))
        {
            continue;
        }
        appendUniqueCleanupPath(
            pathIsManaged(*plan, directory)
                ? &plan->preservedUnsafePaths
                : &plan->preservedExternalPaths,
            directory);
    }
}

void classifyOwnedDirectories(const SelectedArtifacts &selectedArtifacts,
                              const QStringList &retainedPaths,
                              const QStringList &retainedDirectories,
                              ResourceCleanupPlan *plan)
{
    QStringList approved;
    QStringList candidates = selectedArtifacts.ownedDirectories;
    std::sort(candidates.begin(), candidates.end(),
              [](const QString &lhs, const QString &rhs)
              {
                  return lhs.size() < rhs.size();
              });

    for (const QString &directory : candidates)
    {
        if (!pathIsManaged(*plan, directory))
        {
            appendUniqueCleanupPath(&plan->preservedExternalPaths, directory);
            continue;
        }
        if (cleanupPathIsProtected(plan->managedRoot, directory)
            || cleanupPathTraversesLink(plan->managedRoot, directory)
            || cleanupDirectoryContainsLink(directory)
            || !selectedArtifactIsInsideDirectory(directory,
                                                  selectedArtifacts))
        {
            appendUniqueCleanupPath(&plan->preservedUnsafePaths, directory);
            continue;
        }
        if (retainedReferenceOverlapsDirectory(directory,
                                               retainedPaths,
                                               retainedDirectories))
        {
            appendUniqueCleanupPath(&plan->preservedSharedPaths, directory);
            continue;
        }
        if (pathCoveredByDirectory(directory, approved))
        {
            continue;
        }
        approved.append(directory);
    }
    plan->managedDirectories = approved;
}

void classifyFiles(const SelectedArtifacts &selectedArtifacts,
                   const QStringList &retainedDirectories,
                   const QSet<QString> &retainedIdentities,
                   ResourceCleanupPlan *plan)
{
    QSet<QString> acceptedIdentities;
    for (const QString &path : selectedArtifacts.files)
    {
        if (!pathIsManaged(*plan, path))
        {
            appendUniqueCleanupPath(&plan->preservedExternalPaths, path);
            continue;
        }
        const QFileInfo info(path);
        if (cleanupPathIsProtected(plan->managedRoot, path)
            || cleanupPathTraversesLink(plan->managedRoot, path)
            || (info.exists() && info.isDir()))
        {
            appendUniqueCleanupPath(&plan->preservedUnsafePaths, path);
            continue;
        }

        const QString identity = cleanupPathIdentity(path);
        if (retainedIdentities.contains(identity)
            || pathCoveredByDirectory(path, retainedDirectories))
        {
            appendUniqueCleanupPath(&plan->preservedSharedPaths, path);
            continue;
        }
        if (pathCoveredByDirectory(path, plan->managedDirectories))
        {
            continue;
        }
        if (!acceptedIdentities.contains(identity))
        {
            acceptedIdentities.insert(identity);
            plan->managedFiles.append(path);
        }
    }
}

bool initializeRoots(ProjectData *projectData,
                     ResourceCleanupPlan *plan,
                     QString *errorMessage)
{
    const QString projectPath = projectData->currentProjectPath();
    plan->projectPath = QDir::cleanPath(
        QFileInfo(projectPath).absoluteFilePath());
    plan->chunkId = projectData->activeChunkId();
    plan->chunkDirectory = projectData->activeChunkDirectory();
    plan->projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(
        projectPath);
    plan->managedRoot = xjw::common::project::ProjectPackageLayout::chunkDirectory(
        projectPath,
        projectData->activeChunkDirectory());
    if (plan->projectRoot.isEmpty() || plan->managedRoot.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法确定当前项目的受管 Chunk 目录");
        }
        return false;
    }
    if (cleanupPathIdentity(plan->projectRoot)
        != cleanupPathIdentity(plan->managedRoot))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "项目运行目录与当前 Chunk 目录不一致，已拒绝清理：%1 / %2")
                .arg(plan->projectRoot, plan->managedRoot);
        }
        return false;
    }
    return true;
}

} // namespace

QString cleanupSectionArrayKey(const QString &section)
{
    if (section == QStringLiteral("观测网络"))
    {
        return QStringLiteral("observation_network_results");
    }
    for (const auto &entry : supportedResultSections())
    {
        if (entry.first == section)
        {
            return entry.second;
        }
    }
    return {};
}

bool buildResourceCleanupPlan(ProjectData *projectData,
                              const QString &section,
                              const QStringList &resourcePaths,
                              ResourceCleanupPlan *plan,
                              QString *errorMessage)
{
    if (!projectData || !plan)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("资源清理计划参数无效");
        }
        return false;
    }
    if (!initializeRoots(projectData, plan, errorMessage))
    {
        return false;
    }

    const QString arrayKey = cleanupSectionArrayKey(section);
    plan->originalMetadata = projectData->metadataIncludingResults();
    const QJsonArray sourceArray = plan->originalMetadata.value(arrayKey).toArray();
    QJsonArray keptArray;
    SelectedArtifacts selectedArtifacts;

    QSet<QString> normalizedTargets;
    QSet<int> indexTargets;
    for (const QString &path : resourcePaths)
    {
        if (section == QStringLiteral("观测网络"))
        {
            bool ok = false;
            const int index = path.toInt(&ok);
            if (ok)
            {
                indexTargets.insert(index);
            }
        }
        else
        {
            const QString target = normalizedCleanupPath(plan->projectRoot, path);
            const QString identity = cleanupPathIdentity(target);
            if (!identity.isEmpty())
            {
                normalizedTargets.insert(identity);
            }
        }
    }

    for (int index = 0; index < sourceArray.size(); ++index)
    {
        const QJsonObject record = sourceArray.at(index).toObject();
        const bool selected = section == QStringLiteral("观测网络")
            ? indexTargets.contains(index)
            : normalizedTargets.contains(cleanupPathIdentity(
                  cleanupPrimaryPath(section, record, plan->projectRoot)));
        if (!selected)
        {
            keptArray.append(record);
            continue;
        }

        ++plan->removedCount;
        mergeArtifacts(collectCleanupRecordArtifacts(section,
                                                     record,
                                                     plan->projectRoot),
                       &selectedArtifacts);
    }

    plan->updatedMetadata = plan->originalMetadata;
    plan->updatedMetadata[arrayKey] = keptArray;
    if (plan->removedCount <= 0)
    {
        return true;
    }

    QStringList retainedPaths;
    QStringList retainedDirectories;
    collectRetainedReferences(plan->updatedMetadata,
                              plan->projectRoot,
                              &retainedPaths,
                              &retainedDirectories);
    classifyUnverifiedDirectories(selectedArtifacts, plan);
    classifyOwnedDirectories(selectedArtifacts,
                             retainedPaths,
                             retainedDirectories,
                             plan);
    classifyFiles(selectedArtifacts,
                  retainedDirectories,
                  pathIdentitySet(retainedPaths),
                  plan);
    return true;
}

} // namespace xjw::core::project::detail

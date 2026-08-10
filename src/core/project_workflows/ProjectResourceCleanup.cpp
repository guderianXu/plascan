#include "ProjectResourceCleanup.h"

#include "ProjectResourceCleanupPlan.h"

#include "Logger.h"
#include "project/ProjectSessionModel.h"

namespace xjw::core::project
{

struct PreparedResourceCleanup::Impl
{
    detail::ResourceCleanupPlan plan;
    ProjectResourceCleanupPersistence persistence;
};

bool PreparedResourceCleanup::requiresExecution() const
{
    return static_cast<bool>(_impl);
}

const ResourceCleanupResult &
PreparedResourceCleanup::preparationResult() const
{
    return _preparationResult;
}

void ProjectResourceCleanupService::installAutomaticRecovery(
    ProjectData *projectData)
{
    ProjectData::installProjectOpenPreflight(
        [](const QString &projectPath, QString *errorMessage)
        {
            return detail::recoverPendingResourceCleanupTransactionsBeforeOpen(
                projectPath, errorMessage);
        });
    if (!projectData
        || projectData->property(
            "plascan_resource_cleanup_recovery_installed").toBool())
    {
        return;
    }
    projectData->setProperty(
        "plascan_resource_cleanup_recovery_installed", true);
    const auto recover = [projectData]()
    {
        ResourceCleanupResult recovery;
        if (!detail::recoverPendingResourceCleanupTransactions(
                projectData, &recovery))
        {
            LOG_WARN(QStringLiteral(
                "项目清理事务自动恢复失败：%1")
                         .arg(recovery.errorMessage));
        }
    };
    QObject::connect(projectData,
                     &ProjectData::projectOpened,
                     projectData,
                     [recover](const QString &)
                     {
                         recover();
                     });
    QObject::connect(projectData,
                     &ProjectData::activeChunkChanged,
                     projectData,
                     [recover](const QString &, const QString &, int)
                     {
                         recover();
                     });
    if (projectData->hasProject())
    {
        recover();
    }
}

ResourceCleanupResult ProjectResourceCleanupService::cleanupGeneratedData(
    ProjectData *projectData,
    const QString &section,
    const QStringList &resourcePaths)
{
    ResourceCleanupResult result;

    if (!projectData)
    {
        result.errorMessage = QStringLiteral("ProjectData 未初始化");
        return result;
    }
    if (!detail::recoverPendingResourceCleanupTransactions(projectData,
                                                           &result))
    {
        return result;
    }
    if (resourcePaths.isEmpty())
    {
        result.errorMessage = QStringLiteral("待删除资源为空");
        return result;
    }

    result.sectionArrayKey = detail::cleanupSectionArrayKey(section);
    if (result.sectionArrayKey.isEmpty())
    {
        result.unsupportedSection = true;
        return result;
    }

    detail::ResourceCleanupPlan plan;
    if (!detail::buildResourceCleanupPlan(projectData,
                                          section,
                                          resourcePaths,
                                          &plan,
                                          &result.errorMessage))
    {
        return result;
    }

    result.removedCount = plan.removedCount;
    result.preservedExternalPaths = plan.preservedExternalPaths;
    result.preservedSharedPaths = plan.preservedSharedPaths;
    result.preservedUnsafePaths = plan.preservedUnsafePaths;
    if (plan.removedCount <= 0)
    {
        result.noMatchedRecords = true;
        return result;
    }

    detail::executeResourceCleanupPlan(projectData, plan, &result);
    return result;
}

PreparedResourceCleanup
ProjectResourceCleanupService::prepareGeneratedDataCleanup(
    ProjectData *projectData,
    const QString &section,
    const QStringList &resourcePaths)
{
    PreparedResourceCleanup prepared;
    ResourceCleanupResult &result = prepared._preparationResult;
    if (!projectData)
    {
        result.errorMessage = QStringLiteral("ProjectData 未初始化");
        return prepared;
    }
    if (!detail::recoverPendingResourceCleanupTransactions(projectData,
                                                           &result))
    {
        return prepared;
    }
    if (resourcePaths.isEmpty())
    {
        result.errorMessage = QStringLiteral("待删除资源为空");
        return prepared;
    }

    result.sectionArrayKey = detail::cleanupSectionArrayKey(section);
    if (result.sectionArrayKey.isEmpty())
    {
        result.unsupportedSection = true;
        return prepared;
    }

    auto impl = std::make_shared<PreparedResourceCleanup::Impl>();
    if (!detail::buildResourceCleanupPlan(projectData,
                                          section,
                                          resourcePaths,
                                          &impl->plan,
                                          &result.errorMessage))
    {
        return prepared;
    }
    result.removedCount = impl->plan.removedCount;
    result.preservedExternalPaths = impl->plan.preservedExternalPaths;
    result.preservedSharedPaths = impl->plan.preservedSharedPaths;
    result.preservedUnsafePaths = impl->plan.preservedUnsafePaths;
    if (impl->plan.removedCount <= 0)
    {
        result.noMatchedRecords = true;
        return prepared;
    }

    impl->persistence = projectData->prepareResourceCleanupPersistence(
        impl->plan.updatedMetadata);
    if (!impl->persistence.isValid())
    {
        result.errorMessage = QStringLiteral(
            "无法准备资源清理持久化任务");
        return prepared;
    }
    prepared._impl = std::move(impl);
    return prepared;
}

ResourceCleanupResult
ProjectResourceCleanupService::executePreparedCleanup(
    const PreparedResourceCleanup &prepared)
{
    ResourceCleanupResult result = prepared._preparationResult;
    if (!prepared._impl)
    {
        return result;
    }
    detail::executePreparedResourceCleanupPlan(
        prepared._impl->persistence,
        prepared._impl->plan,
        &result);
    return result;
}

bool ProjectResourceCleanupService::finalizePreparedCleanup(
    ProjectData *projectData,
    const PreparedResourceCleanup &prepared,
    const ResourceCleanupResult &result)
{
    return projectData
        && prepared._impl
        && projectData->finalizeResourceCleanupPersistence(
            prepared._impl->persistence,
            result.success,
            result.metadataStateCommitted);
}

} // namespace xjw::core::project

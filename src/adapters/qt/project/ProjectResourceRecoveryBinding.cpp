#include "ProjectResourceRecoveryBinding.h"

#include "ProjectResourceCleanup.h"
#include "Logger.h"
#include "project/ProjectSessionModel.h"

namespace xjw::app::project
{
    using core::project::ProjectResourceCleanupService;
    using core::project::ResourceCleanupResult;

    void ProjectResourceRecoveryBinding::install(ProjectData* projectData)
    {
        ProjectData::installProjectOpenPreflight(
            [](const QString& projectPath, QString* errorMessage)
            { return ProjectResourceCleanupService::recoverPendingTransactionsBeforeOpen(projectPath, errorMessage); });
        if (!projectData || projectData->property("plascan_resource_cleanup_recovery_installed").toBool())
        {
            return;
        }
        projectData->setProperty("plascan_resource_cleanup_recovery_installed", true);
        const auto recover = [projectData]()
        {
            ResourceCleanupResult recovery;
            if (!ProjectResourceCleanupService::recoverPendingTransactions(projectData, &recovery))
            {
                LOG_WARN(QStringLiteral("项目清理事务自动恢复失败：%1").arg(recovery.errorMessage));
            }
        };
        QObject::connect(
            projectData, &ProjectData::projectOpened, projectData, [recover](const QString&) { recover(); });
        QObject::connect(projectData,
                         &ProjectData::activeChunkChanged,
                         projectData,
                         [recover](const QString&, const QString&, int) { recover(); });
        if (projectData->hasProject())
        {
            recover();
        }
    }

} // namespace xjw::app::project

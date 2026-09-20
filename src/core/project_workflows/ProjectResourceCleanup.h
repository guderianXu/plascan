#pragma once

#include <QString>
#include <QStringList>

#include <memory>

class ProjectData;

namespace xjw::core::project
{

    struct ResourceCleanupResult
    {
        bool success = false;
        bool unsupportedSection = false;
        bool noMatchedRecords = false;
        // Internal lifecycle result: either updated metadata (success) or original
        // metadata (rollback) is durably committed for the active generation.
        bool metadataStateCommitted = false;
        int removedCount = 0;
        QString sectionArrayKey;
        QString errorMessage;
        QStringList failedPaths;
        // 项目外、越界别名和指向项目外的符号链接仅移除元数据引用，绝不删除目标。
        QStringList preservedExternalPaths;
        // 仍被其它保留记录引用的受管产物不会进入删除事务。
        QStringList preservedSharedPaths;
        // 项目归档、恢复目录或无法证明为记录专属的目录等高风险路径保持不变。
        QStringList preservedUnsafePaths;
    };

    class PreparedResourceCleanup final
    {
    public:
        PreparedResourceCleanup() = default;

        bool requiresExecution() const;
        const ResourceCleanupResult& preparationResult() const;

    private:
        struct Impl;
        std::shared_ptr<Impl> _impl;
        ResourceCleanupResult _preparationResult;

        friend class ProjectResourceCleanupService;
    };

    class ProjectResourceCleanupService
    {
    public:
        // Synchronous recovery operations; the application owns session event binding.
        static bool recoverPendingTransactionsBeforeOpen(const QString& projectPath, QString* errorMessage);
        static bool recoverPendingTransactions(ProjectData* projectData, ResourceCleanupResult* result);

        static ResourceCleanupResult
        cleanupGeneratedData(ProjectData* projectData, const QString& section, const QStringList& resourcePaths);

        static PreparedResourceCleanup
        prepareGeneratedDataCleanup(ProjectData* projectData, const QString& section, const QStringList& resourcePaths);
        static ResourceCleanupResult executePreparedCleanup(const PreparedResourceCleanup& prepared);
        static bool finalizePreparedCleanup(ProjectData* projectData,
                                            const PreparedResourceCleanup& prepared,
                                            const ResourceCleanupResult& result);
    };

} // namespace xjw::core::project

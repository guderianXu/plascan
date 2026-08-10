#pragma once

#include <QString>
#include <QStringList>

namespace xjw::common::project
{

class ProjectSharedImageStore
{
public:
    explicit ProjectSharedImageStore(const QString &projectPath);

    bool importImage(const QString &sourcePath,
                     QString *resourceUri,
                     QString *materializedPath = nullptr,
                     QString *errorMessage = nullptr) const;
    // 发布已经复制到共享库的引用。发布后 reservation 会一直保留，直到
    // 包含对应 URI 的 Chunk 归档代次成功提交。
    bool publishReferences(const QStringList &references,
                           QString *errorMessage = nullptr) const;
    // 放弃尚未提交的导入 reservation。已经提交的归档引用不受影响。
    void releaseReservations(const QStringList &references) const;
    QString materialize(const QString &resourceUri,
                        QString *errorMessage = nullptr) const;
    // 在一个已成功提交的归档代次之后执行保守 GC。未引用实体必须连续
    // 两个已记录代次无引用且无 active reservation 才会被删除。
    bool pruneUnreferenced(QString *errorMessage = nullptr) const;

private:
    QString _projectPath;
};

} // namespace xjw::common::project

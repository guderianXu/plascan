#pragma once

#include <memory>

#include <QString>

class QLockFile;

namespace xjw::common::project
{

class ProjectLock
{
public:
    ProjectLock();
    ~ProjectLock();

    ProjectLock(const ProjectLock &) = delete;
    ProjectLock &operator=(const ProjectLock &) = delete;

    bool acquire(const QString &projectPath,
                 QString *errorMessage = nullptr);
    void release();
    bool isLocked() const;

private:
    std::unique_ptr<QLockFile> _lockFile;
};

} // namespace xjw::common::project

#pragma once

#include <QString>

namespace xjw::common::project
{

    // Creates a self-contained ZIP without changing the currently opened project.
    class ProjectExporter
    {
    public:
        // Creates an isolated copy beside the requested output. The caller owns
        // the returned staging directory and must remove it after use.
        static bool createStagingProject(const QString& sourceProjectPath,
                                         const QString& outputZipPath,
                                         QString* stagingProjectPath,
                                         QString* errorMessage = nullptr);

        static bool exportPortableProject(const QString& sourceProjectPath,
                                          const QString& outputZipPath,
                                          QString* errorMessage = nullptr);
    };

} // namespace xjw::common::project

#pragma once

#include "file/FileIO.h"
#include "plafs/PlaChunkLayout.h"
#include "plafs/PlaProjectLayout.h"

#include <QDir>
#include <QString>

#include <filesystem>
#include <optional>

namespace xjw::common::project::path_bridge
{

    /// Converts a Qt project-root string to the Qt-free Chunk path layout.
    inline std::optional<xjw::common::plafs::PlaChunkLayout> chunkLayout(const QString& root)
    {
        if (root.trimmed().isEmpty())
        {
            return std::nullopt;
        }
        return xjw::common::plafs::PlaChunkLayout::open(xjw::common::file::pathFromUtf8(root.toUtf8().toStdString()));
    }

    /// Converts a Qt project descriptor string to the Qt-free project layout.
    inline std::optional<xjw::common::plafs::PlaProjectLayout> projectLayout(const QString& projectPath)
    {
        return xjw::common::plafs::PlaProjectLayout::open(
            xjw::common::file::pathFromUtf8(projectPath.toUtf8().toStdString()));
    }

    /// Converts a standard path back to a QString without locale-dependent conversion.
    inline QString toQtPath(const std::filesystem::path& path)
    {
        const std::string utf8Path = xjw::common::file::pathToUtf8(path);
        return QDir::fromNativeSeparators(QString::fromUtf8(utf8Path.c_str()));
    }

} // namespace xjw::common::project::path_bridge

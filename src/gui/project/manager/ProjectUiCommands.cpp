#include "ProjectUiCommands.h"

#include "ProjectData.h"
#include "ProjectOpenGuard.h"

#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>

namespace {

void configureDialog(QFileDialog &dialog)
{
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    dialog.setFilter(QDir::AllEntries | QDir::Hidden | QDir::AllDirs);
}
} // namespace

ProjectUiCommands::ProjectUiCommands(ProjectData *projectData, QWidget *parentWidget)
    : _projectData(projectData)
    , _parentWidget(parentWidget)
{
}

void ProjectUiCommands::setDirectoryAccessors(std::function<QString(const QString &key)> getLastDir,
                                              std::function<void(const QString &key, const QString &dir)> saveLastDir)
{
    _getLastDir = std::move(getLastDir);
    _saveLastDir = std::move(saveLastDir);
}

bool ProjectUiCommands::createNewProject(QString *createdPath) const
{
    QFileDialog dialog(_parentWidget,
                       QStringLiteral("创建新项目"),
                       readLastDir(QStringLiteral("project")),
                       QStringLiteral("PlaScan项目 (*.plascan)"));
    configureDialog(dialog);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setDefaultSuffix(QStringLiteral("plascan"));
    if (dialog.exec() != QDialog::Accepted)
    {
        return false;
    }
    QString plascanPath = dialog.selectedFiles().first();
    if (plascanPath.isEmpty())
    {
        return false;
    }

    if (!plascanPath.endsWith(QStringLiteral(".plascan"), Qt::CaseInsensitive))
    {
        plascanPath += QStringLiteral(".plascan");
    }

    writeLastDir(QStringLiteral("project"), QFileInfo(plascanPath).absolutePath());

    const QString projectName = QFileInfo(plascanPath).baseName();
    if (!_projectData || !_projectData->createProject(plascanPath, projectName))
    {
        QMessageBox::critical(_parentWidget,
                              QStringLiteral("错误"),
                              QStringLiteral("创建项目失败"));
        return false;
    }

    if (createdPath)
    {
        *createdPath = plascanPath;
    }
    return true;
}

bool ProjectUiCommands::openProjectByDialog(QString *openedPath) const
{
    QString plascanPath;
    if (!selectProjectByDialog(&plascanPath))
    {
        return false;
    }

    if (!openProjectFromPath(plascanPath))
    {
        return false;
    }

    if (openedPath)
    {
        *openedPath = plascanPath;
    }
    return true;
}

bool ProjectUiCommands::selectProjectByDialog(QString *selectedPath) const
{
    QFileDialog dialog(_parentWidget,
                       QStringLiteral("打开项目"),
                       readLastDir(QStringLiteral("project")),
                       QStringLiteral("PlaScan项目 (*.plascan)"));
    configureDialog(dialog);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    if (dialog.exec() != QDialog::Accepted)
    {
        return false;
    }
    const QString plascanPath = dialog.selectedFiles().first();
    if (plascanPath.isEmpty())
    {
        return false;
    }

    writeLastDir(QStringLiteral("project"), QFileInfo(plascanPath).absolutePath());
    if (selectedPath)
    {
        *selectedPath = plascanPath;
    }
    return true;
}

bool ProjectUiCommands::openProjectFromPath(const QString &plascanPath) const
{
    QString error;
    if (_projectData && _projectData->openProject(plascanPath, &error))
    {
        return true;
    }

    QMessageBox::critical(_parentWidget,
                          QStringLiteral("错误"),
                          QStringLiteral("打开项目失败: %1").arg(error));
    return false;
}

bool ProjectUiCommands::saveProject() const
{
    if (!_projectData)
    {
        return false;
    }

    QString error;
    if (_projectData->saveProject(&error))
    {
        return true;
    }

    QMessageBox::critical(_parentWidget,
                          QStringLiteral("错误"),
                          QStringLiteral("保存项目失败: %1").arg(error));
    return false;
}

void ProjectUiCommands::closeProject() const
{
    if (_projectData)
    {
        _projectData->closeProject();
    }
}

bool ProjectUiCommands::selectPhotos(QStringList *selectedFiles) const
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开或创建项目")))
    {
        return false;
    }

    QFileDialog dialog(_parentWidget,
                       QStringLiteral("添加图片"),
                       readLastDir(QStringLiteral("images")),
                       QStringLiteral("图片文件 (*.tif *.tiff *.TIF *.TIFF *.png *.PNG *.jpg *.jpeg *.JPG *.JPEG)"));
    configureDialog(dialog);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFiles);
    if (dialog.exec() != QDialog::Accepted)
    {
        return false;
    }
    const QStringList files = dialog.selectedFiles();
    if (files.isEmpty())
    {
        return false;
    }

    writeLastDir(QStringLiteral("images"), QFileInfo(files.first()).absolutePath());
    if (selectedFiles)
    {
        *selectedFiles = files;
    }
    return true;
}

bool ProjectUiCommands::selectImageFolder(QString *selectedFolder) const
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, QStringLiteral("请先打开或创建项目")))
    {
        return false;
    }

    QFileDialog dialog(_parentWidget,
                       QStringLiteral("选择文件夹"),
                       readLastDir(QStringLiteral("images")));
    configureDialog(dialog);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::Directory);
    if (dialog.exec() != QDialog::Accepted)
    {
        return false;
    }
    const QString folder = dialog.selectedFiles().first();
    if (folder.isEmpty())
    {
        return false;
    }

    writeLastDir(QStringLiteral("images"), folder);
    if (selectedFolder)
    {
        *selectedFolder = folder;
    }
    return true;
}

QString ProjectUiCommands::readLastDir(const QString &key) const
{
    if (_getLastDir)
    {
        const QString value = _getLastDir(key);
        if (!value.isEmpty())
        {
            return value;
        }
    }
    return QDir::homePath();
}

void ProjectUiCommands::writeLastDir(const QString &key, const QString &dir) const
{
    if (_saveLastDir)
    {
        _saveLastDir(key, dir);
    }
}

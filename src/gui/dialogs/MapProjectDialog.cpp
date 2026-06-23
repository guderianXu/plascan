#include "MapProjectDialog.h"

#include "ui_MapProjectDialog.h"

#include <QListWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>

MapProjectDialog::MapProjectDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::MapProjectDialog ui;
    ui.setupUi(this);

    _imageList = ui.m_imageList;
    _demEdit = ui.m_demEdit;
    _outputEdit = ui.m_outputEdit;
    _resolutionSpin = ui.m_resolutionSpin;

    connect(ui.demBtn, &QPushButton::clicked, this, &MapProjectDialog::onChooseDem);
    connect(ui.outBtn, &QPushButton::clicked, this, &MapProjectDialog::onChooseOutput);
    connect(ui.runBtn, &QPushButton::clicked, this, &MapProjectDialog::onRun);

    connect(_demEdit, &QLineEdit::textChanged, this, &MapProjectDialog::onSettingsModified);
    connect(_outputEdit, &QLineEdit::textChanged, this, &MapProjectDialog::onSettingsModified);
    connect(_resolutionSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this,
            &MapProjectDialog::onSettingsModified);
}

void MapProjectDialog::setAvailableImages(const QStringList &images)
{
    _imageList->clear();
    for (const QString &path : images)
    {
        auto *it = new QListWidgetItem(path, _imageList);
        it->setFlags(it->flags() | Qt::ItemIsUserCheckable);
        it->setCheckState(Qt::Checked);
    }
}

void MapProjectDialog::setProjectRoot(const QString &projectRoot)
{
    _projectRoot = projectRoot;
    if (_outputEdit->text().trimmed().isEmpty() && !projectRoot.isEmpty())
    {
        const QString defaultOut = QDir(projectRoot).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
        _outputEdit->setText(defaultOut);
    }
}

void MapProjectDialog::setDefaultDemPath(const QString &demPath)
{
    if (_demEdit && _demEdit->text().trimmed().isEmpty() && !demPath.trimmed().isEmpty())
    {
        _demEdit->setText(demPath.trimmed());
    }
}

void MapProjectDialog::applySettings(const QJsonObject &settings)
{
    if (settings.contains(QStringLiteral("dem_path")))
    {
        const QString savedDemPath = settings.value(QStringLiteral("dem_path")).toString().trimmed();
        if (QFileInfo::exists(savedDemPath))
        {
            _demEdit->setText(savedDemPath);
        }
    }
    if (settings.contains(QStringLiteral("output_path")))
    {
        _outputEdit->setText(settings.value(QStringLiteral("output_path")).toString());
    }
    if (settings.contains(QStringLiteral("resolution")))
    {
        _resolutionSpin->setValue(settings.value(QStringLiteral("resolution")).toDouble(_resolutionSpin->value()));
    }
}

void MapProjectDialog::onChooseDem()
{
    QString path = QFileDialog::getOpenFileName(this,
                                                QStringLiteral("选择 DEM"),
                                                _projectRoot,
                                                QStringLiteral("Raster (*.tif *.tiff *.img *.vrt);;All Files (*)"));
    if (!path.isEmpty())
    {
        _demEdit->setText(path);
    }
}

void MapProjectDialog::onChooseOutput()
{
    QString path = QFileDialog::getSaveFileName(this,
                                                QStringLiteral("选择正射影像输出路径"),
                                                _outputEdit->text(),
                                                QStringLiteral("GeoTIFF (*.tif);;PNG (*.png);;All Files (*)"));
    if (!path.isEmpty())
    {
        _outputEdit->setText(path);
    }
}

void MapProjectDialog::onRun()
{
    QStringList images;
    for (int i = 0; i < _imageList->count(); ++i)
    {
        auto *it = _imageList->item(i);
        if (it && it->checkState() == Qt::Checked)
        {
            images << it->text();
        }
    }

    if (images.isEmpty())
    {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请至少勾选一张输入影像。"));
        return;
    }
    if (_demEdit->text().trimmed().isEmpty())
    {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请指定 DEM 文件。"));
        return;
    }
    const QFileInfo demInfo(_demEdit->text().trimmed());
    if (!demInfo.exists() || !demInfo.isFile())
    {
        QMessageBox::warning(this,
                             QStringLiteral("参数错误"),
                             QStringLiteral("DEM 路径不是有效的 DEM 文件。"));
        return;
    }
    if (_outputEdit->text().trimmed().isEmpty())
    {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请指定正射影像输出路径。"));
        return;
    }

    emit requestRunMapProject(images,
                              _demEdit->text().trimmed(),
                              _outputEdit->text().trimmed(),
                              _resolutionSpin->value());
}

void MapProjectDialog::onSettingsModified()
{
    emit settingsChanged(currentSettings());
}

QJsonObject MapProjectDialog::currentSettings() const
{
    QJsonObject settings;
    settings[QStringLiteral("dem_path")] = _demEdit ? _demEdit->text().trimmed() : QString();
    settings[QStringLiteral("output_path")] = _outputEdit ? _outputEdit->text().trimmed() : QString();
    settings[QStringLiteral("resolution")] = _resolutionSpin ? _resolutionSpin->value() : 0.0;
    return settings;
}

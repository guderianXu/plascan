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

    m_imageList = ui.m_imageList;
    m_demEdit = ui.m_demEdit;
    m_outputEdit = ui.m_outputEdit;
    m_resolutionSpin = ui.m_resolutionSpin;

    connect(ui.demBtn, &QPushButton::clicked, this, &MapProjectDialog::onChooseDem);
    connect(ui.outBtn, &QPushButton::clicked, this, &MapProjectDialog::onChooseOutput);
    connect(ui.runBtn, &QPushButton::clicked, this, &MapProjectDialog::onRun);

    connect(m_demEdit, &QLineEdit::textChanged, this, &MapProjectDialog::onSettingsModified);
    connect(m_outputEdit, &QLineEdit::textChanged, this, &MapProjectDialog::onSettingsModified);
    connect(m_resolutionSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this,
            &MapProjectDialog::onSettingsModified);
}

void MapProjectDialog::setAvailableImages(const QStringList &images)
{
    m_imageList->clear();
    for (const QString &path : images) {
        auto *it = new QListWidgetItem(path, m_imageList);
        it->setFlags(it->flags() | Qt::ItemIsUserCheckable);
        it->setCheckState(Qt::Checked);
    }
}

void MapProjectDialog::setProjectRoot(const QString &projectRoot)
{
    m_projectRoot = projectRoot;
    if (m_outputEdit->text().trimmed().isEmpty() && !projectRoot.isEmpty()) {
        const QString defaultOut = QDir(projectRoot).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
        m_outputEdit->setText(defaultOut);
    }
}

void MapProjectDialog::setDefaultDemPath(const QString &demPath)
{
    if (m_demEdit && m_demEdit->text().trimmed().isEmpty() && !demPath.trimmed().isEmpty())
    {
        m_demEdit->setText(demPath.trimmed());
    }
}

void MapProjectDialog::applySettings(const QJsonObject &settings)
{
    if (settings.contains(QStringLiteral("dem_path"))) {
        const QString savedDemPath = settings.value(QStringLiteral("dem_path")).toString().trimmed();
        if (QFileInfo::exists(savedDemPath)) {
            m_demEdit->setText(savedDemPath);
        }
    }
    if (settings.contains(QStringLiteral("output_path"))) {
        m_outputEdit->setText(settings.value(QStringLiteral("output_path")).toString());
    }
    if (settings.contains(QStringLiteral("resolution"))) {
        m_resolutionSpin->setValue(settings.value(QStringLiteral("resolution")).toDouble(m_resolutionSpin->value()));
    }
}

void MapProjectDialog::onChooseDem()
{
    QString path = QFileDialog::getOpenFileName(this,
                                                QStringLiteral("选择 DEM"),
                                                m_projectRoot,
                                                QStringLiteral("Raster (*.tif *.tiff *.img *.vrt);;All Files (*)"));
    if (!path.isEmpty()) m_demEdit->setText(path);
}

void MapProjectDialog::onChooseOutput()
{
    QString path = QFileDialog::getSaveFileName(this,
                                                QStringLiteral("选择正射影像输出路径"),
                                                m_outputEdit->text(),
                                                QStringLiteral("GeoTIFF (*.tif);;PNG (*.png);;All Files (*)"));
    if (!path.isEmpty()) m_outputEdit->setText(path);
}

void MapProjectDialog::onRun()
{
    QStringList images;
    for (int i = 0; i < m_imageList->count(); ++i) {
        auto *it = m_imageList->item(i);
        if (it && it->checkState() == Qt::Checked) images << it->text();
    }

    if (images.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请至少勾选一张输入影像。"));
        return;
    }
    if (m_demEdit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请指定 DEM 文件。"));
        return;
    }
    const QFileInfo demInfo(m_demEdit->text().trimmed());
    if (!demInfo.exists() || !demInfo.isFile()) {
        QMessageBox::warning(this,
                             QStringLiteral("参数错误"),
                             QStringLiteral("DEM 路径不是有效的 DEM 文件。"));
        return;
    }
    if (m_outputEdit->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("参数错误"), QStringLiteral("请指定正射影像输出路径。"));
        return;
    }

    emit requestRunMapProject(images,
                              m_demEdit->text().trimmed(),
                              m_outputEdit->text().trimmed(),
                              m_resolutionSpin->value());
}

void MapProjectDialog::onSettingsModified()
{
    emit settingsChanged(currentSettings());
}

QJsonObject MapProjectDialog::currentSettings() const
{
    QJsonObject settings;
    settings[QStringLiteral("dem_path")] = m_demEdit ? m_demEdit->text().trimmed() : QString();
    settings[QStringLiteral("output_path")] = m_outputEdit ? m_outputEdit->text().trimmed() : QString();
    settings[QStringLiteral("resolution")] = m_resolutionSpin ? m_resolutionSpin->value() : 0.0;
    return settings;
}

#include "MapProjectDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
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
    setWindowTitle(QStringLiteral("生成正射影像"));
    resize(620, 440);

    auto *mainLay = new QHBoxLayout(this);

    auto *leftLay = new QVBoxLayout();
    auto *hint = new QLabel(
        QStringLiteral("建议输出 GeoTIFF（.tif）以保持与 DEM 一致的投影与像元网格；"
                       "分辨率设为“自动”时将直接沿用 DEM 分辨率。"),
        this);
    hint->setWordWrap(true);
    hint->setStyleSheet(QStringLiteral("color:#666;"));
    leftLay->addWidget(hint);
    leftLay->addWidget(new QLabel(QStringLiteral("输入影像（可多选）"), this));
    m_imageList = new QListWidget(this);
    m_imageList->setSelectionMode(QAbstractItemView::NoSelection);
    leftLay->addWidget(m_imageList, 1);
    mainLay->addLayout(leftLay, 1);

    auto *rightLay = new QVBoxLayout();
    auto *form = new QFormLayout();

    m_demEdit = new QLineEdit(this);
    auto *demBtn = new QPushButton(QStringLiteral("选择 DEM..."), this);
    connect(demBtn, &QPushButton::clicked, this, &MapProjectDialog::onChooseDem);
    auto *demLay = new QHBoxLayout();
    demLay->addWidget(m_demEdit, 1);
    demLay->addWidget(demBtn);
    form->addRow(QStringLiteral("DEM："), demLay);

    m_outputEdit = new QLineEdit(this);
    auto *outBtn = new QPushButton(QStringLiteral("选择输出..."), this);
    connect(outBtn, &QPushButton::clicked, this, &MapProjectDialog::onChooseOutput);
    auto *outLay = new QHBoxLayout();
    outLay->addWidget(m_outputEdit, 1);
    outLay->addWidget(outBtn);
    form->addRow(QStringLiteral("输出影像："), outLay);

    m_resolutionSpin = new QDoubleSpinBox(this);
    m_resolutionSpin->setRange(0.0, 10000.0);
    m_resolutionSpin->setDecimals(3);
    m_resolutionSpin->setValue(0.0);
    m_resolutionSpin->setSpecialValueText(QStringLiteral("自动（与 DEM 一致）"));
    m_resolutionSpin->setSuffix(QStringLiteral(" m/px"));
    form->addRow(QStringLiteral("分辨率："), m_resolutionSpin);

    rightLay->addLayout(form);

    auto *runBtn = new QPushButton(QStringLiteral("运行正射投影"), this);
    connect(runBtn, &QPushButton::clicked, this, &MapProjectDialog::onRun);
    rightLay->addWidget(runBtn);
    rightLay->addStretch(1);
    mainLay->addLayout(rightLay, 1);

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
        m_demEdit->setText(settings.value(QStringLiteral("dem_path")).toString());
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

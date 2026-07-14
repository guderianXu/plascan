#include "PrintMarkersDialog.h"

#include <QtConcurrent>

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QVBoxLayout>

namespace xjw::gui::markers
{

namespace
{

using control_points::MarkerTargetFamily;

bool isCircularCodedFamily(MarkerTargetFamily family)
{
    return family >= MarkerTargetFamily::Circular12Bit
        && family <= MarkerTargetFamily::Circular20Bit;
}

} // namespace

PrintMarkersDialog::PrintMarkersDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    populateFamilies();
    connect(&_watcher, &QFutureWatcher<control_points::MarkerPdfWriteResult>::finished,
            this, &PrintMarkersDialog::handleFinished);
}

void PrintMarkersDialog::setDefaultOutputDirectory(const QString &directory)
{
    _defaultOutputDirectory = QDir::cleanPath(directory);
    if (_outputEdit->text().isEmpty() && !_defaultOutputDirectory.isEmpty())
    {
        _outputEdit->setText(QDir(_defaultOutputDirectory).filePath(QStringLiteral("marker_sheet.pdf")));
    }
}

void PrintMarkersDialog::closeEvent(QCloseEvent *event)
{
    if (_running)
    {
        _closeAfterRun = true;
        _statusLabel->setText(QStringLiteral("PDF 正在写入，完成后关闭窗口..."));
        event->ignore();
        return;
    }
    QDialog::closeEvent(event);
}

void PrintMarkersDialog::setupUi()
{
    setWindowTitle(QStringLiteral("打印标靶"));
    setMinimumWidth(560);
    setModal(true);
    auto *root = new QVBoxLayout(this);
    auto *form = new QFormLayout();

    _familyCombo = new QComboBox(this);
    _familyCombo->setObjectName(QStringLiteral("printMarkerFamilyCombo"));
    form->addRow(QStringLiteral("标靶类型:"), _familyCombo);

    _startIdSpin = new QSpinBox(this);
    _startIdSpin->setObjectName(QStringLiteral("printMarkerStartId"));
    _startIdSpin->setRange(0, 1000000);
    _startIdSpin->setValue(0);
    form->addRow(QStringLiteral("起始 ID:"), _startIdSpin);

    _countSpin = new QSpinBox(this);
    _countSpin->setObjectName(QStringLiteral("printMarkerCount"));
    _countSpin->setRange(1, 10000);
    _countSpin->setValue(12);
    form->addRow(QStringLiteral("数量:"), _countSpin);

    _diameterSpin = new QDoubleSpinBox(this);
    _diameterSpin->setObjectName(QStringLiteral("printMarkerDiameterMm"));
    _diameterSpin->setRange(2.0, 500.0);
    _diameterSpin->setDecimals(1);
    _diameterSpin->setValue(30.0);
    _diameterSpin->setSuffix(QStringLiteral(" mm"));
    form->addRow(QStringLiteral("目标直径:"), _diameterSpin);

    _pageSizeCombo = new QComboBox(this);
    _pageSizeCombo->setObjectName(QStringLiteral("printMarkerPageSize"));
    _pageSizeCombo->addItem(QStringLiteral("A4 (210 x 297 mm)"), QSizeF(210.0, 297.0));
    _pageSizeCombo->addItem(QStringLiteral("Letter (215.9 x 279.4 mm)"), QSizeF(215.9, 279.4));
    form->addRow(QStringLiteral("页面:"), _pageSizeCombo);

    _marginSpin = new QDoubleSpinBox(this);
    _marginSpin->setRange(0.0, 100.0);
    _marginSpin->setValue(12.0);
    _marginSpin->setSuffix(QStringLiteral(" mm"));
    form->addRow(QStringLiteral("页边距:"), _marginSpin);

    _spacingSpin = new QDoubleSpinBox(this);
    _spacingSpin->setRange(0.0, 100.0);
    _spacingSpin->setValue(8.0);
    _spacingSpin->setSuffix(QStringLiteral(" mm"));
    form->addRow(QStringLiteral("标靶间距:"), _spacingSpin);

    _showLabelsCheck = new QCheckBox(QStringLiteral("显示族名与 ID"), this);
    _showLabelsCheck->setChecked(true);
    form->addRow(QString(), _showLabelsCheck);

    auto *output_row = new QHBoxLayout();
    _outputEdit = new QLineEdit(this);
    _outputEdit->setObjectName(QStringLiteral("printMarkerOutputPath"));
    _browseButton = new QPushButton(QStringLiteral("浏览..."), this);
    output_row->addWidget(_outputEdit, 1);
    output_row->addWidget(_browseButton);
    form->addRow(QStringLiteral("输出 PDF:"), output_row);
    root->addLayout(form);

    _statusLabel = new QLabel(QStringLiteral("打印尺寸按 PDF 物理页面保存"), this);
    _statusLabel->setWordWrap(true);
    root->addWidget(_statusLabel);

    auto *buttons = new QHBoxLayout();
    buttons->addStretch();
    _generateButton = new QPushButton(QStringLiteral("生成 PDF"), this);
    _generateButton->setObjectName(QStringLiteral("generateMarkerPdfButton"));
    _generateButton->setDefault(true);
    _closeButton = new QPushButton(QStringLiteral("关闭"), this);
    buttons->addWidget(_generateButton);
    buttons->addWidget(_closeButton);
    root->addLayout(buttons);

    connect(_browseButton, &QPushButton::clicked, this, &PrintMarkersDialog::selectOutputPath);
    connect(_generateButton, &QPushButton::clicked, this, &PrintMarkersDialog::generatePdf);
    connect(_closeButton, &QPushButton::clicked, this, &QDialog::close);
}

void PrintMarkersDialog::populateFamilies()
{
    const QVector<MarkerTargetFamily> families = {
        MarkerTargetFamily::AprilTag16h5,
        MarkerTargetFamily::AprilTag25h9,
        MarkerTargetFamily::AprilTag36h10,
        MarkerTargetFamily::AprilTag36h11,
        MarkerTargetFamily::AprilTagCircle21h7,
        MarkerTargetFamily::AprilTagStandard41h12,
        MarkerTargetFamily::AprilTagStandard52h13,
        MarkerTargetFamily::Circular12Bit,
        MarkerTargetFamily::Circular14Bit,
        MarkerTargetFamily::Circular16Bit,
        MarkerTargetFamily::Circular20Bit,
        MarkerTargetFamily::NonCodedCircle,
        MarkerTargetFamily::NonCodedFourQuadrant,
    };
    for (const MarkerTargetFamily family : families)
    {
        _familyCombo->addItem(control_points::markerTargetFamilyName(family),
                              static_cast<int>(family));
        const int index = _familyCombo->count() - 1;
        if (isCircularCodedFamily(family))
        {
            _familyCombo->setItemData(
                index,
                QStringLiteral("需要先导入经许可导出的 Metashape 官方圆形标靶语料"),
                Qt::ToolTipRole);
            if (auto *model = qobject_cast<QStandardItemModel *>(_familyCombo->model()))
            {
                model->item(index)->setEnabled(false);
            }
        }
    }
    _familyCombo->setCurrentIndex(_familyCombo->findData(
        static_cast<int>(MarkerTargetFamily::AprilTag36h11)));
}

void PrintMarkersDialog::selectOutputPath()
{
    const QString initial = _outputEdit->text().isEmpty()
        ? QDir(_defaultOutputDirectory).filePath(QStringLiteral("marker_sheet.pdf"))
        : _outputEdit->text();
    const QString selected = QFileDialog::getSaveFileName(
        this, QStringLiteral("保存标靶 PDF"), initial, QStringLiteral("PDF 文件 (*.pdf)"));
    if (!selected.isEmpty())
    {
        _outputEdit->setText(selected);
    }
}

control_points::MarkerPrintRequest PrintMarkersDialog::printRequest() const
{
    control_points::MarkerPrintRequest request;
    request.family = static_cast<MarkerTargetFamily>(_familyCombo->currentData().toInt());
    for (int offset = 0; offset < _countSpin->value(); ++offset)
    {
        request.ids.push_back(_startIdSpin->value() + offset);
    }
    request.targetDiameterMm = _diameterSpin->value();
    request.pageSizeMm = _pageSizeCombo->currentData().toSizeF();
    request.marginMm = _marginSpin->value();
    request.spacingMm = _spacingSpin->value();
    request.showLabels = _showLabelsCheck->isChecked();
    return request;
}

void PrintMarkersDialog::generatePdf()
{
    if (_running)
    {
        return;
    }
    if (_outputEdit->text().trimmed().isEmpty())
    {
        selectOutputPath();
        if (_outputEdit->text().trimmed().isEmpty()) return;
    }
    const control_points::MarkerPrintRequest request = printRequest();
    const QString output_path = QDir::cleanPath(_outputEdit->text().trimmed());
    setRunning(true);
    _statusLabel->setText(QStringLiteral("正在生成标靶 PDF..."));
    _watcher.setFuture(QtConcurrent::run([request, output_path]()
    {
        return control_points::MarkerPdfWriter::write(request, output_path, 300);
    }));
}

void PrintMarkersDialog::handleFinished()
{
    const control_points::MarkerPdfWriteResult result = _watcher.result();
    setRunning(false);
    _statusLabel->setText(result.ok
                              ? QStringLiteral("已生成 %1 页：%2")
                                    .arg(result.pageCount)
                                    .arg(result.outputPath)
                              : QStringLiteral("生成失败：%1").arg(result.error));
    if (_closeAfterRun)
    {
        _closeAfterRun = false;
        accept();
    }
}

void PrintMarkersDialog::setRunning(bool running)
{
    _running = running;
    _familyCombo->setEnabled(!running);
    _startIdSpin->setEnabled(!running);
    _countSpin->setEnabled(!running);
    _diameterSpin->setEnabled(!running);
    _pageSizeCombo->setEnabled(!running);
    _marginSpin->setEnabled(!running);
    _spacingSpin->setEnabled(!running);
    _showLabelsCheck->setEnabled(!running);
    _outputEdit->setEnabled(!running);
    _browseButton->setEnabled(!running);
    _generateButton->setEnabled(!running);
    _closeButton->setEnabled(!running);
}

} // namespace xjw::gui::markers

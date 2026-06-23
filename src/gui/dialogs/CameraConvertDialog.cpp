#include "CameraConvertDialog.h"

#include "CameraFormatConverter.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QStringList>
#include <QTextEdit>
#include <QVBoxLayout>

CameraConvertDialog::CameraConvertDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("相机格式转换"));
    setMinimumWidth(680);
    buildUi();
}

void CameraConvertDialog::buildUi()
{
    auto *mainLayout = new QVBoxLayout(this);

    auto *hint = new QLabel(tr("将外部相机文件转换为 PlaScan 可直接使用的 tsai 文件和 image_camera.lis。"));
    hint->setWordWrap(true);
    mainLayout->addWidget(hint);

    auto *formLayout = new QFormLayout();
    formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    _formatCombo = new QComboBox(this);
    _formatCombo->addItem(tr("自动识别"), QStringLiteral("auto"));
    _formatCombo->addItem(tr("Middlebury *_par.txt"), QStringLiteral("middlebury-par"));
    _formatCombo->addItem(tr("EPFL/Strecha *.camera"), QStringLiteral("epfl-camera"));
    _formatCombo->addItem(tr("COLMAP text sparse"), QStringLiteral("colmap-text"));
    _formatCombo->addItem(tr("Metashape doc.xml / chunk.zip"), QStringLiteral("metashape-xml"));
    formLayout->addRow(tr("输入格式:"), _formatCombo);

    auto *inputRow = new QWidget(this);
    auto *inputLayout = new QHBoxLayout(inputRow);
    inputLayout->setContentsMargins(0, 0, 0, 0);
    _inputEdit = new QLineEdit(inputRow);
    _inputEdit->setPlaceholderText(tr("输入相机文件或目录"));
    auto *inputDirButton = new QPushButton(tr("选目录"), inputRow);
    auto *inputFileButton = new QPushButton(tr("选文件"), inputRow);
    inputLayout->addWidget(_inputEdit, 1);
    inputLayout->addWidget(inputDirButton);
    inputLayout->addWidget(inputFileButton);
    formLayout->addRow(tr("输入路径:"), inputRow);

    auto *outputRow = new QWidget(this);
    auto *outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    _outputEdit = new QLineEdit(outputRow);
    _outputEdit->setPlaceholderText(tr("输出目录"));
    auto *outputButton = new QPushButton(tr("浏览..."), outputRow);
    outputLayout->addWidget(_outputEdit, 1);
    outputLayout->addWidget(outputButton);
    formLayout->addRow(tr("输出目录:"), outputRow);

    _overwriteCheck = new QCheckBox(tr("覆盖非空输出目录"), this);
    formLayout->addRow(QString(), _overwriteCheck);

    mainLayout->addLayout(formLayout);

    _statusLabel = new QLabel(tr("未开始"), this);
    mainLayout->addWidget(_statusLabel);

    _resultEdit = new QTextEdit(this);
    _resultEdit->setReadOnly(true);
    _resultEdit->setMinimumHeight(120);
    mainLayout->addWidget(_resultEdit);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close, this);
    _runButton = new QPushButton(tr("开始转换"), buttonBox);
    buttonBox->addButton(_runButton, QDialogButtonBox::AcceptRole);
    mainLayout->addWidget(buttonBox);

    connect(inputDirButton, &QPushButton::clicked, this, &CameraConvertDialog::browseInputDirectory);
    connect(inputFileButton, &QPushButton::clicked, this, &CameraConvertDialog::browseInputFile);
    connect(outputButton, &QPushButton::clicked, this, &CameraConvertDialog::browseOutput);
    connect(_runButton, &QPushButton::clicked, this, &CameraConvertDialog::runConversion);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void CameraConvertDialog::browseInputDirectory()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输入相机目录"),
        _inputEdit ? _inputEdit->text() : QString());
    if (!dir.isEmpty() && _inputEdit)
    {
        _inputEdit->setText(dir);
    }
}

void CameraConvertDialog::browseInputFile()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        tr("选择输入相机文件"),
        _inputEdit ? _inputEdit->text() : QString(),
        tr("相机文件 (*.txt *.camera);;所有文件 (*)"));
    if (!path.isEmpty() && _inputEdit)
    {
        _inputEdit->setText(path);
    }
}

void CameraConvertDialog::browseOutput()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输出目录"),
        _outputEdit ? _outputEdit->text() : QString());
    if (!dir.isEmpty() && _outputEdit)
    {
        _outputEdit->setText(dir);
    }
}

void CameraConvertDialog::runConversion()
{
    const QString inputPath = _inputEdit ? _inputEdit->text().trimmed() : QString();
    const QString outputDir = _outputEdit ? _outputEdit->text().trimmed() : QString();
    if (inputPath.isEmpty())
    {
        setResultText(tr("请输入相机文件或目录。"), true);
        return;
    }
    if (outputDir.isEmpty())
    {
        setResultText(tr("请输入输出目录。"), true);
        return;
    }

    const QString formatName = _formatCombo
        ? _formatCombo->currentData().toString()
        : QStringLiteral("auto");
    const auto parsedFormat = xjw::camera::parseCameraFormat(formatName.toStdString());
    if (!parsedFormat)
    {
        setResultText(tr("不支持的相机格式：%1").arg(formatName), true);
        return;
    }

    xjw::camera::CameraConversionOptions options;
    options.format = *parsedFormat;
    options.inputPath = inputPath.toStdString();
    options.outputDir = outputDir.toStdString();
    options.overwrite = _overwriteCheck && _overwriteCheck->isChecked();

    setResultText(tr("正在转换..."), false);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    const auto result = xjw::camera::convertCameraDataset(options);
    QApplication::restoreOverrideCursor();

    if (!result.success)
    {
        setResultText(tr("转换失败：%1").arg(QString::fromStdString(result.errorMessage)), true);
        return;
    }

    QStringList lines;
    lines << tr("转换完成：%1 个相机").arg(result.cameraCount);
    lines << tr("输入格式：%1").arg(QString::fromStdString(xjw::camera::cameraFormatName(result.inputFormat)));
    lines << tr("image_camera.lis：%1").arg(QString::fromStdString(result.imageCameraList.string()));
    lines << tr("summary.json：%1").arg(QString::fromStdString(result.summaryPath.string()));
    for (const std::string &warning : result.warnings)
    {
        lines << tr("警告：%1").arg(QString::fromStdString(warning));
    }
    setResultText(lines.join(QLatin1Char('\n')), false);
    QMessageBox::information(this, tr("相机格式转换"), tr("转换完成。"));
}

void CameraConvertDialog::setResultText(const QString &text, bool error)
{
    if (_statusLabel)
    {
        _statusLabel->setText(error ? tr("失败") : tr("状态更新"));
    }
    if (_resultEdit)
    {
        _resultEdit->setPlainText(text);
    }
}

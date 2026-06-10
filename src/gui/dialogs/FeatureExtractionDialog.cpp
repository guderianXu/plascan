// FeatureExtractionDialog.cpp
// 多算法特征点提取配置对话框实现

#include "FeatureExtractionDialog.h"
#include "ui_FeatureExtractionDialog.h"

#include <QAbstractItemView>
#include <QListWidget>
#include <QPushButton>
#include <QJsonObject>
#include <QLineEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QCoreApplication>
#include <QDir>
#include <QGroupBox>
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QLabel>
#include <QToolButton>
#include <QMessageBox>
#include <QSignalBlocker>

namespace
{

QString findModelFile(const QString &modelName)
{
    QStringList candidates;

    const QString envModelDir = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!envModelDir.isEmpty())
    {
        candidates.append(QDir(envModelDir).filePath(modelName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("resources/models/%1").arg(modelName)));
#endif

    const QString exePath = QCoreApplication::applicationDirPath();
    candidates.append(QDir(exePath).filePath(QStringLiteral("../models/%1").arg(modelName)));
    candidates.append(QDir(exePath).filePath(QStringLiteral("../resources/models/%1").arg(modelName)));
    candidates.append(QDir(exePath).filePath(QStringLiteral("../../resources/models/%1").arg(modelName)));
    candidates.append(QDir::current().filePath(QStringLiteral("resources/models/%1").arg(modelName)));
    candidates.append(QDir::current().filePath(QStringLiteral("models/%1").arg(modelName)));

    for (const QString &candidate : candidates)
    {
        if (QFile::exists(candidate))
        {
            return QDir::cleanPath(QFileInfo(candidate).absoluteFilePath());
        }
    }

    return QString();
}

QStringList modelCandidates(const QString &algorithm, bool useCuda)
{
    if (algorithm == QStringLiteral("superpoint"))
    {
        QStringList candidates;
        if (useCuda)
        {
            candidates << QStringLiteral("superpoint_extractor_cuda.pt");
        }
        candidates << QStringLiteral("superpoint_extractor_cpu.pt")
                   << QStringLiteral("superpoint_extractor.pt");
        return candidates;
    }

    if (algorithm == QStringLiteral("disk"))
    {
        QStringList candidates;
        if (useCuda)
        {
            candidates << QStringLiteral("disk_extractor_cuda_1200.pt");
        }
        candidates << QStringLiteral("disk_extractor_cpu_1200.pt")
                   << QStringLiteral("disk_extractor.pt");
        return candidates;
    }

    if (algorithm == QStringLiteral("aliked"))
    {
        QStringList candidates;
        if (useCuda)
        {
            candidates << QStringLiteral("aliked_extractor_cuda_480.pt");
        }
        candidates << QStringLiteral("aliked_extractor_cpu_480.pt")
                   << QStringLiteral("aliked_extractor.pt");
        return candidates;
    }

    return {};
}

QString defaultModelPath(const QString &algorithm, bool useCuda)
{
    const QStringList candidates = modelCandidates(algorithm, useCuda);
    for (const QString &candidate : candidates)
    {
        const QString path = findModelFile(candidate);
        if (!path.isEmpty())
        {
            return path;
        }
    }
    return QString();
}

bool isManagedModelPath(const QString &path)
{
    const QString fileName = QFileInfo(path).fileName().toLower();
    return fileName.startsWith(QStringLiteral("superpoint_extractor"))
        || fileName.startsWith(QStringLiteral("disk_extractor"))
        || fileName.startsWith(QStringLiteral("aliked_extractor"));
}

void setFormRowVisible(QFormLayout *form, QWidget *field, bool visible)
{
    if (!form || !field)
    {
        return;
    }

    if (QWidget *label = form->labelForField(field))
    {
        label->setVisible(visible);
    }
    field->setVisible(visible);
}

} // namespace

FeatureExtractionDialog::FeatureExtractionDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    setupConnections();
}

FeatureExtractionDialog::~FeatureExtractionDialog() = default;

void FeatureExtractionDialog::setupUi() 
{
    Ui::FeatureExtractionDialog ui;
    ui.setupUi(this);

    m_fileList = ui.m_fileList;
    m_addFilesBtn = ui.m_addFilesBtn;
    m_addFolderBtn = ui.m_addFolderBtn;
    m_removeBtn = ui.m_removeBtn;
    m_clearBtn = ui.m_clearBtn;
    m_outputLine = ui.m_outputLine;
    m_browseOutBtn = ui.m_browseOutBtn;
    m_basicForm = ui.m_basicForm;
    m_algorithmCombo = ui.m_algorithmCombo;
    m_modelPathEdit = ui.m_modelPathEdit;
    m_cudaRowWidget = ui.m_cudaRowWidget;
    m_useCudaChk = ui.m_useCudaChk;
    m_cudaDeviceSpin = ui.m_cudaDeviceSpin;
    m_nmsRadiusSpin = ui.m_nmsRadiusSpin;
    m_detectionThresholdSpin = ui.m_detectionThresholdSpin;
    m_maxKeypointsSpin = ui.m_maxKeypointsSpin;
    m_removeBordersSpin = ui.m_removeBordersSpin;
    m_grayRangeWidget = ui.m_grayRangeWidget;
    m_grayscaleMinSpin = ui.m_grayscaleMinSpin;
    m_grayscaleMaxSpin = ui.m_grayscaleMaxSpin;
    m_advancedGroup = ui.m_advancedGroup;
    m_advancedForm = ui.m_advancedForm;
    m_advancedHintLabel = ui.m_advancedHintLabel;
    m_normalizeInputChk = ui.m_normalizeInputChk;
    m_descriptorDimSpin = ui.m_descriptorDimSpin;
    m_gridSizeSpin = ui.m_gridSizeSpin;
    m_batchSizeSpin = ui.m_batchSizeSpin;
    m_neighborhoodRadiusSpin = ui.m_neighborhoodRadiusSpin;
    m_neighborhoodThresholdSpin = ui.m_neighborhoodThresholdSpin;
    m_systemGroup = ui.m_systemGroup;
    m_deviceCombo = ui.m_deviceCombo;
    m_allowFallbackChk = ui.m_allowFallbackChk;
    m_pythonPathEdit = ui.m_pythonPathEdit;
    m_debugGroup = ui.m_debugGroup;
    m_saveCsvChk = ui.m_saveCsvChk;
    m_saveOverlayChk = ui.m_saveOverlayChk;
    m_resetBtn = ui.m_resetBtn;
    m_runBtn = ui.m_runBtn;
    m_cancelBtn = ui.m_cancelBtn;

    if (ui.topSplit)
    {
        ui.topSplit->setStretchFactor(0, 3);
        ui.topSplit->setStretchFactor(1, 2);
    }

    m_fileList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    m_algorithmCombo->clear();
    m_algorithmCombo->addItem("SuperPoint", "superpoint");
    m_algorithmCombo->addItem("DISK", "disk");
    m_algorithmCombo->addItem("ALIKED", "aliked");
    m_algorithmCombo->addItem("ORB", "orb");
    m_algorithmCombo->addItem("SIFT", "sift");
    m_algorithmCombo->setCurrentIndex(0);

    m_deviceCombo->clear();
    m_deviceCombo->addItems(QStringList() << "CPU" << "CUDA");
    m_deviceCombo->setCurrentText("CUDA");

    connect(ui.advancedToggle, &QToolButton::toggled, this, [this, advancedToggle = ui.advancedToggle](bool checked)
    {
        m_advancedGroup->setVisible(checked);
        advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.systemToggle, &QToolButton::toggled, this, [this, systemToggle = ui.systemToggle](bool checked)
    {
        m_systemGroup->setVisible(checked);
        systemToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.debugToggle, &QToolButton::toggled, this, [this, debugToggle = ui.debugToggle](bool checked)
    {
        m_debugGroup->setVisible(checked);
        debugToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });
}

void FeatureExtractionDialog::setupConnections() 
{
    // 文件操作
    connect(m_addFilesBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onAddFiles);
    connect(m_addFolderBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onAddFolder);
    connect(m_removeBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onRemoveSelected);
    connect(m_clearBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onClearFiles);
    connect(m_browseOutBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onBrowseOutput);

    // 底部按钮
    connect(m_runBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onCancel);
    connect(m_resetBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onResetDefaults);

    // 参数变更信号（实时同步到项目配置）
    // 基础参数
        connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureExtractionDialog::onAlgorithmChanged);
        connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_nmsRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_detectionThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_removeBordersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_grayscaleMinSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_grayscaleMaxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_useCudaChk, &QCheckBox::toggled, this, [this]()
    {
        updateModelPathForCurrentAlgorithm();
        emitSettingsNow();
    });
    connect(m_cudaDeviceSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);

    // 高级参数
    connect(m_normalizeInputChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_descriptorDimSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_gridSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_batchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_neighborhoodRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_neighborhoodThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &FeatureExtractionDialog::emitSettingsNow);

    // 系统参数
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_allowFallbackChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_pythonPathEdit, &QLineEdit::textChanged, this, &FeatureExtractionDialog::emitSettingsNow);

    // 调试参数
    connect(m_saveCsvChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);
    connect(m_saveOverlayChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);

    // 初始状态
    onAlgorithmChanged(m_algorithmCombo->currentIndex());
}

void FeatureExtractionDialog::onAddFiles() 
{
    QStringList files = QFileDialog::getOpenFileNames(
        this, 
        tr("选择输入影像"),
        QString(),
        tr("图像文件 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;所有文件 (*)")
    );
    
    for (const QString& file : files) {
        if (!file.isEmpty()) {
            QListWidgetItem *item = new QListWidgetItem(file, m_fileList);
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            item->setCheckState(Qt::Checked);
        }
    }
}

void FeatureExtractionDialog::onAddFolder() 
{
    QString dir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输入文件夹"),
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (!dir.isEmpty()) {
        // 扫描文件夹中的图像文件
        QDir folder(dir);
        QStringList filters;
        filters << "*.jpg" << "*.jpeg" << "*.png" << "*.tif" << "*.tiff" << "*.bmp";
        QStringList files = folder.entryList(filters, QDir::Files);
        
        for (const QString &file : files) {
            QString fullPath = folder.filePath(file);
            // 创建带checkbox的item
            QListWidgetItem *item = new QListWidgetItem(fullPath, m_fileList);
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            item->setCheckState(Qt::Checked); // 默认选中
        }
    }
}

void FeatureExtractionDialog::onRemoveSelected() 
{
    QList<QListWidgetItem*> selected = m_fileList->selectedItems();
    for (QListWidgetItem* item : selected) {
        delete m_fileList->takeItem(m_fileList->row(item));
    }
}

void FeatureExtractionDialog::onClearFiles() 
{
    m_fileList->clear();
}

void FeatureExtractionDialog::onBrowseOutput() 
{
    QString dir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输出目录"),
        m_outputLine->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (!dir.isEmpty()) {
        m_outputLine->setText(dir);
    }
}

void FeatureExtractionDialog::onRun() 
{
    // 收集选中的输入文件列表（只处理checkbox选中的）
    QStringList inputs;
    for (int i = 0; i < m_fileList->count(); ++i) {
        QListWidgetItem *item = m_fileList->item(i);
        if (item->checkState() == Qt::Checked) {
            inputs.append(item->text());
        }
    }

    if (inputs.isEmpty()) {
        QMessageBox::warning(this, tr("警告"), tr("请至少选中一个输入影像！"));
        return;
    }

    // 收集配置参数
    QJsonObject config = collectSettings();

    // 发送运行请求
    emit runRequested(config, inputs);
    
    // 接受对话框
    accept();
}

void FeatureExtractionDialog::onCancel() 
{
    reject();
}

void FeatureExtractionDialog::onResetDefaults()
{
    // 恢复所有参数到默认值
    m_algorithmCombo->setCurrentIndex(0);
    m_nmsRadiusSpin->setValue(3);
    m_detectionThresholdSpin->setValue(0.003);
    m_maxKeypointsSpin->setValue(-1);
    m_removeBordersSpin->setValue(4);
    m_grayscaleMinSpin->setValue(0.0);
    m_grayscaleMaxSpin->setValue(1.0);

    m_normalizeInputChk->setChecked(true);
    m_descriptorDimSpin->setValue(256);
    m_gridSizeSpin->setValue(8);
    m_batchSizeSpin->setValue(8);
    m_neighborhoodRadiusSpin->setValue(3);
    m_neighborhoodThresholdSpin->setValue(0.05);

    m_deviceCombo->setCurrentText("CUDA");
    m_allowFallbackChk->setChecked(true);

    m_saveCsvChk->setChecked(false);
    m_saveOverlayChk->setChecked(false);
    m_pythonPathEdit->clear();
    updateModelPathForCurrentAlgorithm();
}

void FeatureExtractionDialog::onAlgorithmChanged(int)
{
    const QString algo = m_algorithmCombo->currentData().toString();
    const bool isDL = (algo == "superpoint" || algo == "disk" || algo == "aliked");
    const bool isSP = (algo == "superpoint");
    const bool isPythonDL = (algo == "disk" || algo == "aliked");
    const bool isTraditional = (algo == "orb" || algo == "sift");

    // 模型路径: 仅 DL 算法需要
    setFormRowVisible(m_basicForm, m_modelPathEdit, isDL);
    setFormRowVisible(m_basicForm, m_cudaRowWidget, isDL);

    // SuperPoint 专有参数
    setFormRowVisible(m_basicForm, m_nmsRadiusSpin, isSP);
    setFormRowVisible(m_basicForm, m_detectionThresholdSpin, isDL);
    setFormRowVisible(m_basicForm, m_removeBordersSpin, isSP);
    setFormRowVisible(m_basicForm, m_grayRangeWidget, isSP);

    if (m_advancedHintLabel)
    {
        const bool showHint = isPythonDL || isTraditional;
        if (isPythonDL)
        {
            m_advancedHintLabel->setText(tr("DISK/ALIKED 的可调提取参数集中在基础参数：检测阈值、最大关键点数；"
                                            "模型路径、设备和 Python 环境在系统参数中设置。"));
        }
        else if (isTraditional)
        {
            m_advancedHintLabel->setText(tr("ORB/SIFT 使用 OpenCV 提取器，基础参数中的最大关键点数、边界移除和灰度过滤会生效。"));
        }
        setFormRowVisible(m_advancedForm, m_advancedHintLabel, showHint);
    }

    setFormRowVisible(m_advancedForm, m_normalizeInputChk, isSP);
    setFormRowVisible(m_advancedForm, m_descriptorDimSpin, isSP);
    setFormRowVisible(m_advancedForm, m_gridSizeSpin, isSP);
    setFormRowVisible(m_advancedForm, m_batchSizeSpin, isSP);
    setFormRowVisible(m_advancedForm, m_neighborhoodRadiusSpin, isSP);
    setFormRowVisible(m_advancedForm, m_neighborhoodThresholdSpin, isSP);

    // 描述子维度自动设置
    if (algo == "disk" || algo == "aliked")
        m_descriptorDimSpin->setValue(128);
    else if (isSP)
        m_descriptorDimSpin->setValue(256);

    updateModelPathForCurrentAlgorithm();
    emitSettingsNow();
}

void FeatureExtractionDialog::updateModelPathForCurrentAlgorithm()
{
    const QString algo = m_algorithmCombo->currentData().toString();
    const bool isDL = (algo == "superpoint" || algo == "disk" || algo == "aliked");
    if (!isDL)
    {
        QSignalBlocker blocker(m_modelPathEdit);
        m_modelPathEdit->clear();
        return;
    }

    const QString resolvedPath = defaultModelPath(algo, m_useCudaChk->isChecked());
    const QString currentPath = m_modelPathEdit->text().trimmed();
    const bool shouldReplace = currentPath.isEmpty() || isManagedModelPath(currentPath);

    if (shouldReplace)
    {
        QSignalBlocker blocker(m_modelPathEdit);
        m_modelPathEdit->setText(resolvedPath);
    }

    if (!resolvedPath.isEmpty())
    {
        m_modelPathEdit->setToolTip(tr("当前自动解析模型路径：\n%1").arg(resolvedPath));
    }
    else
    {
        m_modelPathEdit->setToolTip(tr("未找到当前算法的默认模型文件"));
    }
}

void FeatureExtractionDialog::applySettings(const QJsonObject &settings) 
{
    if (settings.contains("feature_algorithm"))
    {
        const QString featureAlgorithm = settings["feature_algorithm"].toString("superpoint").trimmed().toLower();
        const int index = m_algorithmCombo->findData(featureAlgorithm);
        m_algorithmCombo->setCurrentIndex(index >= 0 ? index : 0);
    }

    if (settings.contains("model_path"))
    {
        const QString modelPath = settings["model_path"].toString().trimmed();
        if (!modelPath.isEmpty())
        {
            m_modelPathEdit->setText(modelPath);
        }
    }
    if (settings.contains("use_cuda"))
        m_useCudaChk->setChecked(settings["use_cuda"].toBool());
    if (settings.contains("cuda_device"))
        m_cudaDeviceSpin->setValue(settings["cuda_device"].toInt());
    if (settings.contains("python_executable"))
        m_pythonPathEdit->setText(settings["python_executable"].toString().trimmed());

    // 基础参数
    if (settings.contains("nms_radius"))
        m_nmsRadiusSpin->setValue(settings["nms_radius"].toInt(3));
    if (settings.contains("detection_threshold"))
        m_detectionThresholdSpin->setValue(settings["detection_threshold"].toDouble(0.003));
    if (settings.contains("max_num_keypoints"))
        m_maxKeypointsSpin->setValue(settings["max_num_keypoints"].toInt(-1));
    if (settings.contains("remove_borders"))
        m_removeBordersSpin->setValue(settings["remove_borders"].toInt(4));
    if (settings.contains("grayscale_min"))
        m_grayscaleMinSpin->setValue(settings["grayscale_min"].toDouble(0.0));
    if (settings.contains("grayscale_max"))
        m_grayscaleMaxSpin->setValue(settings["grayscale_max"].toDouble(1.0));

    // 高级参数
    if (settings.contains("normalize_input"))
        m_normalizeInputChk->setChecked(settings["normalize_input"].toBool(true));
    if (settings.contains("descriptor_dim"))
        m_descriptorDimSpin->setValue(settings["descriptor_dim"].toInt(256));
    if (settings.contains("grid_size"))
        m_gridSizeSpin->setValue(settings["grid_size"].toInt(8));
    if (settings.contains("batch_size"))
        m_batchSizeSpin->setValue(settings["batch_size"].toInt(8));
    if (settings.contains("neighborhood_check_radius"))
        m_neighborhoodRadiusSpin->setValue(settings["neighborhood_check_radius"].toInt(3));
    if (settings.contains("neighborhood_threshold"))
        m_neighborhoodThresholdSpin->setValue(settings["neighborhood_threshold"].toDouble(0.05));

    // 系统参数
    if (settings.contains("device"))
        m_deviceCombo->setCurrentText(settings["device"].toString("CUDA"));  // 默认CUDA
    if (settings.contains("allow_device_fallback"))
        m_allowFallbackChk->setChecked(settings["allow_device_fallback"].toBool(true));

    // 调试参数
    if (settings.contains("save_keypoints_csv"))
        m_saveCsvChk->setChecked(settings["save_keypoints_csv"].toBool(false));
    if (settings.contains("save_overlay_image"))
        m_saveOverlayChk->setChecked(settings["save_overlay_image"].toBool(false));

    // 输出目录
    if (settings.contains("output_dir")) {
        QString outputDir = settings["output_dir"].toString();
        if (!outputDir.isEmpty()) {
            m_outputLine->setText(outputDir);
        }
    }

    updateModelPathForCurrentAlgorithm();
}

void FeatureExtractionDialog::setProjectImages(const QStringList &paths) 
{
    m_fileList->clear();
    for (const QString &path : paths) 
    {
        // 创建带checkbox的item
        QListWidgetItem *item = new QListWidgetItem(path, m_fileList);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked); // 默认全选
    }
}

void FeatureExtractionDialog::updatePreview() 
{
    // TODO: 可选实现，提供参数预览或影响说明
}

void FeatureExtractionDialog::emitSettingsNow() 
{
    emit settingsChanged(collectSettings());
}

QJsonObject FeatureExtractionDialog::collectSettings() const 
{
    QJsonObject settings;

    settings["feature_algorithm"] = m_algorithmCombo->currentData().toString();
    settings["model_path"]  = m_modelPathEdit->text();
    settings["use_cuda"]    = m_useCudaChk->isChecked();
    settings["cuda_device"] = m_cudaDeviceSpin->value();

    // 基础参数
    settings["nms_radius"] = m_nmsRadiusSpin->value();
    settings["detection_threshold"] = m_detectionThresholdSpin->value();
    settings["max_num_keypoints"] = m_maxKeypointsSpin->value();
    settings["remove_borders"] = m_removeBordersSpin->value();
    settings["grayscale_min"] = m_grayscaleMinSpin->value();
    settings["grayscale_max"] = m_grayscaleMaxSpin->value();

    // 高级参数
    settings["normalize_input"] = m_normalizeInputChk->isChecked();
    settings["descriptor_dim"] = m_descriptorDimSpin->value();
    settings["grid_size"] = m_gridSizeSpin->value();
    settings["batch_size"] = m_batchSizeSpin->value();
    settings["neighborhood_check_radius"] = m_neighborhoodRadiusSpin->value();
    settings["neighborhood_threshold"] = m_neighborhoodThresholdSpin->value();

    // 系统参数
    settings["device"] = m_deviceCombo->currentText();
    settings["allow_device_fallback"] = m_allowFallbackChk->isChecked();
    settings["python_executable"] = m_pythonPathEdit->text().trimmed();

    // 调试参数
    settings["save_keypoints_csv"] = m_saveCsvChk->isChecked();
    settings["save_overlay_image"] = m_saveOverlayChk->isChecked();

    // 输出目录
    settings["output_dir"] = m_outputLine->text();

    return settings;
}

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
#include <QScopedValueRollback>
#include <QSignalBlocker>
#include <QtGlobal>

namespace
{

constexpr int kDefaultGrayscaleMinPx = 5;
constexpr int kDefaultGrayscaleMaxPx = 255;

double grayscalePixelToNormalized(int value)
{
    return static_cast<double>(qBound(0, value, 255)) / 255.0;
}

int normalizedGrayscaleToPixel(double value)
{
    if (value > 1.0)
    {
        return qBound(0, qRound(value), 255);
    }
    return qBound(0, qRound(value * 255.0), 255);
}

int grayscalePixelSetting(const QJsonObject &settings,
                          const QString &pixelKey,
                          const QString &normalizedKey,
                          int fallback)
{
    if (settings.contains(pixelKey))
    {
        return qBound(0, settings.value(pixelKey).toInt(fallback), 255);
    }
    if (settings.contains(normalizedKey))
    {
        return normalizedGrayscaleToPixel(
            settings.value(normalizedKey).toDouble(grayscalePixelToNormalized(fallback)));
    }
    return fallback;
}

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
            candidates << QStringLiteral("superpoint_extractor_cuda.torchscript")
                       << QStringLiteral("superpoint_extractor_cuda.pt");
        }
        candidates << QStringLiteral("superpoint_extractor_cpu.torchscript")
                   << QStringLiteral("superpoint_extractor_cpu.pt")
                   << QStringLiteral("superpoint_extractor.torchscript")
                   << QStringLiteral("superpoint_extractor.pt");
        return candidates;
    }

    if (algorithm == QStringLiteral("disk"))
    {
        QStringList candidates;
        if (useCuda)
        {
            candidates << QStringLiteral("disk_extractor_cuda_8192.torchscript")
                       << QStringLiteral("disk_extractor_cuda_8192.pt")
                       << QStringLiteral("disk_extractor_cuda_1200.torchscript")
                       << QStringLiteral("disk_extractor_cuda_1200.pt");
        }
        candidates << QStringLiteral("disk_extractor_cpu_8192.torchscript")
                   << QStringLiteral("disk_extractor_cpu_8192.pt")
                   << QStringLiteral("disk_extractor_cpu_1200.torchscript")
                   << QStringLiteral("disk_extractor_cpu_1200.pt")
                   << QStringLiteral("disk_extractor.torchscript")
                   << QStringLiteral("disk_extractor.pt");
        return candidates;
    }

    if (algorithm == QStringLiteral("aliked"))
    {
        QStringList candidates;
        if (useCuda)
        {
            candidates << QStringLiteral("aliked_extractor_cuda_480.torchscript")
                       << QStringLiteral("aliked_extractor_cuda_480.pt");
        }
        candidates << QStringLiteral("aliked_extractor_cpu_480.torchscript")
                   << QStringLiteral("aliked_extractor_cpu_480.pt")
                   << QStringLiteral("aliked_extractor.torchscript")
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

bool deviceTextRequestsCuda(const QString &deviceText)
{
    return deviceText.compare(QStringLiteral("CUDA"), Qt::CaseInsensitive) == 0;
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

void setComboDataOrFirst(QComboBox *combo, const QString &data)
{
    if (!combo)
    {
        return;
    }

    const int index = combo->findData(data);
    combo->setCurrentIndex(index >= 0 ? index : 0);
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

    _fileList = ui.m_fileList;
    _addFilesBtn = ui.m_addFilesBtn;
    _addFolderBtn = ui.m_addFolderBtn;
    _removeBtn = ui.m_removeBtn;
    _clearBtn = ui.m_clearBtn;
    _outputLine = ui.m_outputLine;
    _browseOutBtn = ui.m_browseOutBtn;
    _basicForm = ui.m_basicForm;
    _algorithmCombo = ui.m_algorithmCombo;
    _modelPathEdit = ui.m_modelPathEdit;
    _cudaRowWidget = ui.m_cudaRowWidget;
    _useCudaChk = ui.m_useCudaChk;
    _cudaDeviceSpin = ui.m_cudaDeviceSpin;
    _nmsRadiusSpin = ui.m_nmsRadiusSpin;
    _detectionThresholdSpin = ui.m_detectionThresholdSpin;
    _maxKeypointsSpin = ui.m_maxKeypointsSpin;
    _removeBordersSpin = ui.m_removeBordersSpin;
    _grayRangeWidget = ui.m_grayRangeWidget;
    _grayscaleMinSpin = ui.m_grayscaleMinSpin;
    _grayscaleMaxSpin = ui.m_grayscaleMaxSpin;
    _advancedGroup = ui.m_advancedGroup;
    _advancedForm = ui.m_advancedForm;
    _advancedHintLabel = ui.m_advancedHintLabel;
    _normalizeInputChk = ui.m_normalizeInputChk;
    _descriptorDimSpin = ui.m_descriptorDimSpin;
    _gridSizeSpin = ui.m_gridSizeSpin;
    _batchSizeSpin = ui.m_batchSizeSpin;
    _neighborhoodRadiusSpin = ui.m_neighborhoodRadiusSpin;
    _neighborhoodThresholdSpin = ui.m_neighborhoodThresholdSpin;
    _systemGroup = ui.m_systemGroup;
    _deviceCombo = ui.m_deviceCombo;
    _allowFallbackChk = ui.m_allowFallbackChk;
    _pythonPathEdit = ui.m_pythonPathEdit;
    _debugGroup = ui.m_debugGroup;
    _saveCsvChk = ui.m_saveCsvChk;
    _saveOverlayChk = ui.m_saveOverlayChk;
    _resetBtn = ui.m_resetBtn;
    _runBtn = ui.m_runBtn;
    _cancelBtn = ui.m_cancelBtn;

    if (ui.topSplit)
    {
        ui.topSplit->setStretchFactor(0, 3);
        ui.topSplit->setStretchFactor(1, 2);
    }

    _fileList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    _algorithmCombo->clear();
    _algorithmCombo->addItem("SuperPoint", "superpoint");
    _algorithmCombo->addItem("DISK", "disk");
    _algorithmCombo->addItem("ALIKED", "aliked");
    _algorithmCombo->addItem("ORB", "orb");
    _algorithmCombo->addItem("SIFT", "sift");
    setComboDataOrFirst(_algorithmCombo, QStringLiteral("disk"));

    _deviceCombo->clear();
    _deviceCombo->addItems(QStringList() << "CPU" << "CUDA");
    _deviceCombo->setCurrentText("CUDA");
    setFormRowVisible(ui.systemForm, _pythonPathEdit, false);

    connect(ui.advancedToggle, &QToolButton::toggled, this, [this, advancedToggle = ui.advancedToggle](bool checked)
    {
        _advancedGroup->setVisible(checked);
        advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.systemToggle, &QToolButton::toggled, this, [this, systemToggle = ui.systemToggle](bool checked)
    {
        _systemGroup->setVisible(checked);
        systemToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.debugToggle, &QToolButton::toggled, this, [this, debugToggle = ui.debugToggle](bool checked)
    {
        _debugGroup->setVisible(checked);
        debugToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });
}

void FeatureExtractionDialog::setupConnections() 
{
    // 文件操作
    connect(_addFilesBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onAddFiles);
    connect(_addFolderBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onAddFolder);
    connect(_removeBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onRemoveSelected);
    connect(_clearBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onClearFiles);
    connect(_browseOutBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onBrowseOutput);

    // 底部按钮
    connect(_runBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onRun);
    connect(_cancelBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onCancel);
    connect(_resetBtn, &QPushButton::clicked, this, &FeatureExtractionDialog::onResetDefaults);

    // 参数变更信号（实时同步到项目配置）
    // 基础参数
    connect(_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureExtractionDialog::onAlgorithmChanged);
    connect(_nmsRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_detectionThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_removeBordersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_grayscaleMinSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_grayscaleMaxSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_useCudaChk, &QCheckBox::toggled, this, [this]()
    {
        {
            QSignalBlocker blocker(_deviceCombo);
            _deviceCombo->setCurrentText(_useCudaChk->isChecked() ? QStringLiteral("CUDA") : QStringLiteral("CPU"));
        }
        updateModelPathForCurrentAlgorithm();
        emitSettingsNow();
    });
    connect(_cudaDeviceSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);

    // 高级参数
    connect(_normalizeInputChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_descriptorDimSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_gridSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_batchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_neighborhoodRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_neighborhoodThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureExtractionDialog::emitSettingsNow);

    // 系统参数
    connect(_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]()
    {
        {
            QSignalBlocker blocker(_useCudaChk);
            _useCudaChk->setChecked(deviceTextRequestsCuda(_deviceCombo->currentText()));
        }
        updateModelPathForCurrentAlgorithm();
        emitSettingsNow();
    });
    connect(_allowFallbackChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_modelPathEdit, &QLineEdit::textChanged, this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_pythonPathEdit, &QLineEdit::textChanged, this, &FeatureExtractionDialog::emitSettingsNow);

    // 调试参数
    connect(_saveCsvChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);
    connect(_saveOverlayChk, &QCheckBox::toggled, this, &FeatureExtractionDialog::emitSettingsNow);

    // 初始状态
    onAlgorithmChanged(_algorithmCombo->currentIndex());
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
            QListWidgetItem *item = new QListWidgetItem(file, _fileList);
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
            QListWidgetItem *item = new QListWidgetItem(fullPath, _fileList);
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            item->setCheckState(Qt::Checked); // 默认选中
        }
    }
}

void FeatureExtractionDialog::onRemoveSelected() 
{
    QList<QListWidgetItem*> selected = _fileList->selectedItems();
    for (QListWidgetItem* item : selected) {
        delete _fileList->takeItem(_fileList->row(item));
    }
}

void FeatureExtractionDialog::onClearFiles() 
{
    _fileList->clear();
}

void FeatureExtractionDialog::onBrowseOutput() 
{
    QString dir = QFileDialog::getExistingDirectory(
        this,
        tr("选择输出目录"),
        _outputLine->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (!dir.isEmpty()) {
        _outputLine->setText(dir);
    }
}

void FeatureExtractionDialog::onRun() 
{
    // 收集选中的输入文件列表（只处理checkbox选中的）
    QStringList inputs;
    for (int i = 0; i < _fileList->count(); ++i) {
        QListWidgetItem *item = _fileList->item(i);
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
    {
        QScopedValueRollback<bool> applying_settings(_applyingSettings, true);

        // 恢复所有参数到默认值
        setComboDataOrFirst(_algorithmCombo, QStringLiteral("disk"));
        _nmsRadiusSpin->setValue(3);
        _detectionThresholdSpin->setValue(0.003);
        _maxKeypointsSpin->setValue(-1);
        _removeBordersSpin->setValue(4);
        _grayscaleMinSpin->setValue(kDefaultGrayscaleMinPx);
        _grayscaleMaxSpin->setValue(kDefaultGrayscaleMaxPx);

        _normalizeInputChk->setChecked(true);
        _descriptorDimSpin->setValue(256);
        _gridSizeSpin->setValue(8);
        _batchSizeSpin->setValue(8);
        _neighborhoodRadiusSpin->setValue(3);
        _neighborhoodThresholdSpin->setValue(0.05);

        _deviceCombo->setCurrentText(QStringLiteral("CUDA"));
        _cudaDeviceSpin->setValue(0);
        _allowFallbackChk->setChecked(true);

        _saveCsvChk->setChecked(false);
        _saveOverlayChk->setChecked(false);
        _modelPathEdit->clear();
        _outputLine->clear();
        _pythonPathEdit->clear();
        updateModelPathForCurrentAlgorithm();
    }
    emitSettingsNow();
}

void FeatureExtractionDialog::onAlgorithmChanged(int)
{
    const QString algo = _algorithmCombo->currentData().toString();
    const bool isDL = (algo == "superpoint" || algo == "disk" || algo == "aliked");
    const bool isSP = (algo == "superpoint");
    const bool isPythonDL = (algo == "disk" || algo == "aliked");
    const bool isTraditional = (algo == "orb" || algo == "sift");
    const bool canUseCuda = isDL || algo == "sift";

    // 模型路径: 仅 DL 算法需要
    setFormRowVisible(_basicForm, _modelPathEdit, isDL);
    setFormRowVisible(_basicForm, _cudaRowWidget, canUseCuda);

    // SuperPoint 专有参数
    setFormRowVisible(_basicForm, _nmsRadiusSpin, isSP);
    setFormRowVisible(_basicForm, _detectionThresholdSpin, isDL);
    setFormRowVisible(_basicForm, _removeBordersSpin, isSP);

    if (_advancedHintLabel)
    {
        const bool showHint = isPythonDL || isTraditional;
        if (isPythonDL)
        {
            _advancedHintLabel->setText(tr("DISK/ALIKED 的检测阈值、最大关键点数在基础参数中设置；"
                                            "灰度阈值在高级参数中设置；"
                                            "模型路径和设备会传给 C++ TorchScript 提取器。"));
        }
        else if (isTraditional)
        {
            _advancedHintLabel->setText(tr("ORB 使用 OpenCV 提取器；SIFT 可优先使用 CUDA SIFT，"
                                            "不可用时回退 OpenCV SIFT。最大关键点数和灰度阈值会生效。"));
        }
        setFormRowVisible(_advancedForm, _advancedHintLabel, showHint);
    }

    setFormRowVisible(_advancedForm, _grayRangeWidget, isDL || isTraditional);
    setFormRowVisible(_advancedForm, _normalizeInputChk, isSP);
    setFormRowVisible(_advancedForm, _descriptorDimSpin, isSP);
    setFormRowVisible(_advancedForm, _gridSizeSpin, isSP);
    setFormRowVisible(_advancedForm, _batchSizeSpin, isSP);
    setFormRowVisible(_advancedForm, _neighborhoodRadiusSpin, isSP);
    setFormRowVisible(_advancedForm, _neighborhoodThresholdSpin, isSP);

    // 描述子维度自动设置
    if (algo == "disk" || algo == "aliked")
        _descriptorDimSpin->setValue(128);
    else if (isSP)
        _descriptorDimSpin->setValue(256);

    updateModelPathForCurrentAlgorithm();
    emitSettingsNow();
}

void FeatureExtractionDialog::updateModelPathForCurrentAlgorithm()
{
    const QString algo = _algorithmCombo->currentData().toString();
    const bool isDL = (algo == "superpoint" || algo == "disk" || algo == "aliked");
    if (!isDL)
    {
        QSignalBlocker blocker(_modelPathEdit);
        _modelPathEdit->clear();
        return;
    }

    const QString resolvedPath = defaultModelPath(algo, deviceTextRequestsCuda(_deviceCombo->currentText()));
    const QString currentPath = _modelPathEdit->text().trimmed();
    const bool shouldReplace = currentPath.isEmpty() || isManagedModelPath(currentPath);

    if (shouldReplace)
    {
        QSignalBlocker blocker(_modelPathEdit);
        _modelPathEdit->setText(resolvedPath);
    }

    if (!resolvedPath.isEmpty())
    {
        _modelPathEdit->setToolTip(tr("当前自动解析模型路径：\n%1").arg(resolvedPath));
    }
    else
    {
        _modelPathEdit->setToolTip(tr("未找到当前算法的默认模型文件"));
    }
}

void FeatureExtractionDialog::applySettings(const QJsonObject &settings) 
{
    QScopedValueRollback<bool> applying_settings(_applyingSettings, true);
    const bool has_model_path = settings.contains(QStringLiteral("model_path"));
    const QString restored_model_path = settings.value(QStringLiteral("model_path")).toString().trimmed();

    if (settings.contains("feature_algorithm"))
    {
        const QString feature_algorithm = settings["feature_algorithm"].toString("disk").trimmed().toLower();
        setComboDataOrFirst(_algorithmCombo, feature_algorithm);
    }

    if (has_model_path)
    {
        QSignalBlocker blocker(_modelPathEdit);
        _modelPathEdit->setText(restored_model_path);
    }
    if (settings.contains("cuda_device"))
        _cudaDeviceSpin->setValue(settings["cuda_device"].toInt());
    if (settings.contains("python_executable"))
        _pythonPathEdit->setText(settings["python_executable"].toString().trimmed());

    // 基础参数
    if (settings.contains("nms_radius"))
        _nmsRadiusSpin->setValue(settings["nms_radius"].toInt(3));
    if (settings.contains("detection_threshold"))
        _detectionThresholdSpin->setValue(settings["detection_threshold"].toDouble(0.003));
    if (settings.contains("max_num_keypoints"))
        _maxKeypointsSpin->setValue(settings["max_num_keypoints"].toInt(-1));
    if (settings.contains("remove_borders"))
        _removeBordersSpin->setValue(settings["remove_borders"].toInt(4));
    if (settings.contains(QStringLiteral("grayscale_min_px")) || settings.contains(QStringLiteral("grayscale_min")))
    {
        _grayscaleMinSpin->setValue(grayscalePixelSetting(settings,
                                                           QStringLiteral("grayscale_min_px"),
                                                           QStringLiteral("grayscale_min"),
                                                           kDefaultGrayscaleMinPx));
    }
    if (settings.contains(QStringLiteral("grayscale_max_px")) || settings.contains(QStringLiteral("grayscale_max")))
    {
        _grayscaleMaxSpin->setValue(grayscalePixelSetting(settings,
                                                           QStringLiteral("grayscale_max_px"),
                                                           QStringLiteral("grayscale_max"),
                                                           kDefaultGrayscaleMaxPx));
    }

    // 高级参数
    if (settings.contains("normalize_input"))
        _normalizeInputChk->setChecked(settings["normalize_input"].toBool(true));
    if (settings.contains("descriptor_dim"))
        _descriptorDimSpin->setValue(settings["descriptor_dim"].toInt(256));
    if (settings.contains("grid_size"))
        _gridSizeSpin->setValue(settings["grid_size"].toInt(8));
    if (settings.contains("batch_size"))
        _batchSizeSpin->setValue(settings["batch_size"].toInt(8));
    if (settings.contains("neighborhood_check_radius"))
        _neighborhoodRadiusSpin->setValue(settings["neighborhood_check_radius"].toInt(3));
    if (settings.contains("neighborhood_threshold"))
        _neighborhoodThresholdSpin->setValue(settings["neighborhood_threshold"].toDouble(0.05));

    // 系统参数
    if (settings.contains("device"))
    {
        const bool use_cuda = deviceTextRequestsCuda(settings["device"].toString("CUDA").trimmed());
        _deviceCombo->setCurrentText(use_cuda ? QStringLiteral("CUDA") : QStringLiteral("CPU"));
        _useCudaChk->setChecked(use_cuda);
    }
    else if (settings.contains("use_cuda"))
    {
        const bool use_cuda = settings["use_cuda"].toBool();
        _useCudaChk->setChecked(use_cuda);
        _deviceCombo->setCurrentText(use_cuda ? QStringLiteral("CUDA") : QStringLiteral("CPU"));
    }
    if (settings.contains("allow_device_fallback"))
        _allowFallbackChk->setChecked(settings["allow_device_fallback"].toBool(true));

    // 调试参数
    if (settings.contains("save_keypoints_csv"))
        _saveCsvChk->setChecked(settings["save_keypoints_csv"].toBool(false));
    if (settings.contains("save_overlay_image"))
        _saveOverlayChk->setChecked(settings["save_overlay_image"].toBool(false));

    // 输出目录
    if (settings.contains("output_dir"))
    {
        const QString output_dir = settings["output_dir"].toString();
        if (!output_dir.isEmpty())
        {
            _outputLine->setText(output_dir);
        }
    }

    updateModelPathForCurrentAlgorithm();
    if (has_model_path && !restored_model_path.isEmpty())
    {
        QSignalBlocker blocker(_modelPathEdit);
        _modelPathEdit->setText(restored_model_path);
    }
    updatePreview();
}

void FeatureExtractionDialog::setProjectImages(const QStringList &paths) 
{
    _fileList->clear();
    for (const QString &path : paths) 
    {
        // 创建带checkbox的item
        QListWidgetItem *item = new QListWidgetItem(path, _fileList);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked); // 默认全选
    }
}

void FeatureExtractionDialog::updatePreview() 
{
    if (!_advancedHintLabel)
    {
        return;
    }

    const QString algo = _algorithmCombo->currentText().trimmed();
    const QString device = _deviceCombo->currentText().trimmed();
    const int grayscale_min_px = qBound(0, qMin(_grayscaleMinSpin->value(), _grayscaleMaxSpin->value()), 255);
    const int grayscale_max_px = qBound(0, qMax(_grayscaleMinSpin->value(), _grayscaleMaxSpin->value()), 255);
    const QString max_keypoints = _maxKeypointsSpin->value() < 0
        ? tr("不限")
        : QString::number(_maxKeypointsSpin->value());
    const QString model_state = _modelPathEdit->text().trimmed().isEmpty()
        ? tr("无模型")
        : QFileInfo(_modelPathEdit->text()).fileName();

    const QString base_hint = _advancedHintLabel->text().section(QStringLiteral("\n\n有效配置："), 0, 0);
    _advancedHintLabel->setText(
        tr("%1\n\n有效配置：%2 / %3，最大关键点 %4，灰度 %5-%6，模型 %7")
            .arg(base_hint,
                 algo,
                 device,
                 max_keypoints,
                 QString::number(grayscale_min_px),
                 QString::number(grayscale_max_px),
                 model_state));
}

void FeatureExtractionDialog::emitSettingsNow() 
{
    if (_applyingSettings)
    {
        return;
    }
    updatePreview();
    emit settingsChanged(collectSettings());
}

QJsonObject FeatureExtractionDialog::collectSettings() const 
{
    QJsonObject settings;

    settings["feature_algorithm"] = _algorithmCombo->currentData().toString();
    settings["model_path"]  = _modelPathEdit->text();
    settings["use_cuda"]    = deviceTextRequestsCuda(_deviceCombo->currentText());
    settings["cuda_device"] = _cudaDeviceSpin->value();

    // 基础参数
    settings["nms_radius"] = _nmsRadiusSpin->value();
    settings["detection_threshold"] = _detectionThresholdSpin->value();
    settings["max_num_keypoints"] = _maxKeypointsSpin->value();
    settings["remove_borders"] = _removeBordersSpin->value();
    const int grayscale_min_px = qBound(0, qMin(_grayscaleMinSpin->value(), _grayscaleMaxSpin->value()), 255);
    const int grayscale_max_px = qBound(0, qMax(_grayscaleMinSpin->value(), _grayscaleMaxSpin->value()), 255);
    settings["grayscale_min_px"] = grayscale_min_px;
    settings["grayscale_max_px"] = grayscale_max_px;
    settings["grayscale_min"] = grayscalePixelToNormalized(grayscale_min_px);
    settings["grayscale_max"] = grayscalePixelToNormalized(grayscale_max_px);

    // 高级参数
    settings["normalize_input"] = _normalizeInputChk->isChecked();
    settings["descriptor_dim"] = _descriptorDimSpin->value();
    settings["grid_size"] = _gridSizeSpin->value();
    settings["batch_size"] = _batchSizeSpin->value();
    settings["neighborhood_check_radius"] = _neighborhoodRadiusSpin->value();
    settings["neighborhood_threshold"] = _neighborhoodThresholdSpin->value();

    // 系统参数
    settings["device"] = _deviceCombo->currentText();
    settings["allow_device_fallback"] = _allowFallbackChk->isChecked();
    settings["python_executable"] = _pythonPathEdit->text().trimmed();

    // 调试参数
    settings["save_keypoints_csv"] = _saveCsvChk->isChecked();
    settings["save_overlay_image"] = _saveOverlayChk->isChecked();

    // 输出目录
    settings["output_dir"] = _outputLine->text();

    return settings;
}

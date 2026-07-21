#include "InitCameraPoseDialog.h"
#include "ui_InitCameraPoseDialog.h"

#include "AlgorithmCompat.h"

#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QStackedWidget>
#include <QLabel>
#include <QDialogButtonBox>
#include <QFileInfo>

namespace
{

void setComboDataOrFirst(QComboBox *combo, const QString &data)
{
    if (!combo)
    {
        return;
    }
    const int idx = combo->findData(data);
    combo->setCurrentIndex(idx >= 0 ? idx : 0);
}

} // namespace

InitCameraPoseDialog::InitCameraPoseDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("初始化相机位姿"));
    setMinimumSize(720, 560);

    Ui::InitCameraPoseDialog ui;
    ui.setupUi(this);

    _modeCombo = ui.m_modeCombo;
    _statusLabel = ui.m_statusLabel;
    _applyBox = ui.m_applyBox;
    _applyForm = ui.m_applyForm;
    _modeStack = ui.m_modeStack;
    _applyScopeCombo = ui.m_applyScopeCombo;
    _applyTargetImageCombo = ui.m_applyTargetImageCombo;
    _overwriteExistingCheck = ui.m_overwriteExistingCheck;
    _applyHintLabel = ui.m_applyHintLabel;
    _qualityCombo = ui.m_qualityCombo;
    _threadsSpin = ui.m_threadsSpin;
    _matchAlgorithmCombo = new QComboBox(this);
    _matchAlgorithmCombo->setObjectName(QStringLiteral("m_matchAlgorithmCombo"));
    _matchAlgorithmCombo->setToolTip(tr("选择初始化位姿时复用哪一种已生成匹配。"));
    _featureSuffixCombo = new QComboBox(this);
    _featureSuffixCombo->setObjectName(QStringLiteral("m_featureSuffixCombo"));
    _featureSuffixCombo->setToolTip(tr("选择与匹配结果对应的特征文件类型。"));
    _exifAutoCheck = ui.m_exifAutoCheck;
    _defaultFocalSpin = ui.m_defaultFocalSpin;
    _sensorWidthSpin = ui.m_sensorWidthSpin;
    _intrinsicsForm = ui.m_intrinsicsForm;
    _fxSpin = ui.m_fxSpin;
    _fySpin = ui.m_fySpin;
    _cxSpin = ui.m_cxSpin;
    _cySpin = ui.m_cySpin;
    _distModelCombo = ui.m_distModelCombo;
    _k1Spin = ui.m_k1Spin;
    _k2Spin = ui.m_k2Spin;
    _p1Spin = ui.m_p1Spin;
    _p2Spin = ui.m_p2Spin;
    _cameraImportForm = ui.m_cameraImportForm;
    _cameraImportModeCombo = ui.m_cameraImportModeCombo;
    _targetImageCombo = ui.m_targetImageCombo;
    _cameraFormatCombo = ui.m_cameraFormatCombo;
    _cameraImportHintLabel = ui.m_cameraImportHintLabel;

    _modeCombo->clear();
    _modeCombo->addItems({
        tr("无相机文件 (从 EXIF 推断)"),
        tr("仅有内参矩阵"),
        tr("完整相机文件 (.tsai/.yaml/.xml)")
    });

    _applyScopeCombo->clear();
    _applyScopeCombo->addItems({
        tr("回写全部参与影像"),
        tr("仅回写目标影像")
    });

    _qualityCombo->clear();
    _qualityCombo->addItem(tr("快速"), 0);
    _qualityCombo->addItem(tr("标准"), 1);
    _qualityCombo->addItem(tr("高质量"), 2);
    _qualityCombo->addItem(tr("最高质量"), 3);
    _qualityCombo->setCurrentIndex(1);

    _matchAlgorithmCombo->clear();
    _matchAlgorithmCombo->addItem(tr("LightGlue"), QStringLiteral("lightglue"));
    _matchAlgorithmCombo->addItem(tr("SuperGlue"), QStringLiteral("superglue"));
    _matchAlgorithmCombo->addItem(tr("BF-Hamming (ORB)"), QStringLiteral("orb_bf_hamming"));
    _matchAlgorithmCombo->addItem(tr("BF-L2 (SIFT)"), QStringLiteral("sift_bf_l2"));
    _matchAlgorithmCombo->addItem(tr("FLANN (SIFT)"), QStringLiteral("sift_flann"));
    setComboDataOrFirst(_matchAlgorithmCombo, QStringLiteral("lightglue"));
    if (ui.solveForm)
    {
        ui.solveForm->addRow(tr("匹配算法:"), _matchAlgorithmCombo);
        ui.solveForm->addRow(tr("特征类型:"), _featureSuffixCombo);
    }
    refreshFeatureSuffixChoices();

    _distModelCombo->clear();
    _distModelCombo->addItems({
        tr("无畸变"),
        tr("径向 (k1, k2)"),
        tr("Brown (k1, k2, p1, p2)")
    });
    _distModelCombo->setCurrentIndex(2);

    _cameraImportModeCombo->clear();
    _cameraImportModeCombo->addItems({
        tr("为单张影像选择对应相机文件"),
        tr("选择目录并按文件名自动匹配")
    });

    _cameraFormatCombo->clear();
    _cameraFormatCombo->addItems({tr("Tsai (.tsai)"), tr("自动检测")});
    _cameraFormatCombo->setCurrentIndex(1);

    ui.buttonBox->button(QDialogButtonBox::Ok)->setText(tr("写入相机初值"));

    connect(_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onModeChanged);
    connect(_applyScopeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onInitTargetModeChanged);
    connect(_cameraImportModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onCameraImportModeChanged);
    connect(_distModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onDistortionModelChanged);

    auto changed = [this]()
    {
        emitSettingsNow();
    };
    connect(_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_applyScopeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_applyTargetImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_overwriteExistingCheck, &QCheckBox::toggled, this, changed);
    connect(_qualityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onMatchPipelineChanged);
    connect(_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_cameraImportModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_targetImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_cameraFormatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_exifAutoCheck, &QCheckBox::toggled, this, changed);
    connect(_defaultFocalSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_sensorWidthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_fxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_fySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_cxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_cySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_distModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_k1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_k2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_p1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_p2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);

    connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &InitCameraPoseDialog::onRun);
    connect(ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onModeChanged(0);
    onMatchPipelineChanged();
    onInitTargetModeChanged(0);
    onCameraImportModeChanged(0);
    onDistortionModelChanged(_distModelCombo->currentIndex());
}

void InitCameraPoseDialog::onModeChanged(int index)
{
    _modeStack->setCurrentIndex(index);
    if (_applyBox)
    {
        _applyBox->setVisible(index != 2);
    }
    updateStatusText();
    updateTargetUi();
}

void InitCameraPoseDialog::onInitTargetModeChanged(int index)
{
    Q_UNUSED(index);
    updateTargetUi();
}

void InitCameraPoseDialog::onDistortionModelChanged(int index)
{
    const bool showK = (index >= 1);
    const bool showP = (index >= 2);

    auto toggleField = [this](QWidget *field, bool visible)
    {
        if (!field || !_intrinsicsForm)
        {
            return;
        }
        field->setVisible(visible);
        if (QWidget *label = _intrinsicsForm->labelForField(field))
        {
            label->setVisible(visible);
        }
    };

    toggleField(_k1Spin, showK);
    toggleField(_k2Spin, showK);
    toggleField(_p1Spin, showP);
    toggleField(_p2Spin, showP);
}

void InitCameraPoseDialog::onCameraImportModeChanged(int index)
{
    const bool exactImport = (index == 0);
    if (_targetImageCombo)
    {
        _targetImageCombo->setVisible(exactImport);
    }
    if (_cameraImportForm && _targetImageCombo)
    {
        if (QWidget *label = _cameraImportForm->labelForField(_targetImageCombo))
        {
            label->setVisible(exactImport);
        }
    }

    if (_cameraImportHintLabel)
    {
        if (exactImport)
        {
            _cameraImportHintLabel->setText(
                tr("点击[初始化位姿]后，将为所选影像打开现有的单文件导入流程。"
                   "适合逐张检查和精确绑定相机文件。"));
        }
        else
        {
            _cameraImportHintLabel->setText(
                tr("点击[初始化位姿]后，将打开现有的目录批量导入流程。"
                   "程序会按文件名自动匹配相机文件与项目影像。"));
        }
    }

    updateStatusText();
}

void InitCameraPoseDialog::setAvailableImages(const QStringList &imagePaths)
{
    auto refillCombo = [&imagePaths](QComboBox *combo)
    {
        if (!combo)
        {
            return;
        }
        const QString currentPath = combo->currentData().toString();
        combo->blockSignals(true);
        combo->clear();

        for (const QString &path : imagePaths)
        {
            const QFileInfo fi(path);
            const QString label = fi.fileName().isEmpty() ? path : fi.fileName();
            combo->addItem(label, path);
        }

        if (!currentPath.isEmpty())
        {
            const int idx = combo->findData(currentPath);
            if (idx >= 0)
            {
                combo->setCurrentIndex(idx);
            }
        }

        combo->setEnabled(combo->count() > 0);
        if (combo->count() == 0)
        {
            combo->addItem(tr("当前项目无可选影像"), QString());
            combo->setEnabled(false);
        }
        combo->blockSignals(false);
    };

    refillCombo(_applyTargetImageCombo);
    refillCombo(_targetImageCombo);
    updateTargetUi();
}

void InitCameraPoseDialog::setAvailableFeatureSuffixes(const QStringList &suffixes)
{
    _projectFeatureSuffixes = xjw::feature_match::normalizedFeatureSuffixes(suffixes);
    refreshFeatureSuffixChoices();
}

QJsonObject InitCameraPoseDialog::collectSettings() const
{
    QJsonObject o;
    o["mode"] = _modeCombo->currentIndex();
    o["applyScope"] = _applyScopeCombo->currentIndex();
    o["applyTargetImagePath"] = _applyTargetImageCombo->currentData().toString();
    o["applyTargetImageName"] = _applyTargetImageCombo->currentText();
    o["overwriteExisting"] = _overwriteExistingCheck->isChecked();
    o["quality"] = _qualityCombo->currentData().toInt();
    o["threads"] = _threadsSpin->value();
    o["match_algorithm"] = _matchAlgorithmCombo->currentData().toString();
    o["feature_suffix"] = selectedFeatureSuffix();
    o["feature_algorithm"] = selectedFeatureAlgorithm();
    // 模式 0
    o["exifAuto"]     = _exifAutoCheck->isChecked();
    o["defaultFocal"]  = _defaultFocalSpin->value();
    o["sensorWidth"]   = _sensorWidthSpin->value();
    // 模式 1
    o["fx"] = _fxSpin->value();
    o["fy"] = _fySpin->value();
    o["cx"] = _cxSpin->value();
    o["cy"] = _cySpin->value();
    o["distortionModel"] = _distModelCombo->currentText();
    o["k1"] = _k1Spin->value();
    o["k2"] = _k2Spin->value();
    o["p1"] = _p1Spin->value();
    o["p2"] = _p2Spin->value();
    // 模式 2
    o["cameraImportMode"] = _cameraImportModeCombo->currentIndex();
    o["targetImagePath"]  = _targetImageCombo->currentData().toString();
    o["targetImageName"]  = _targetImageCombo->currentText();
    o["cameraFormat"] = _cameraFormatCombo->currentText();
    return o;
}

void InitCameraPoseDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("mode"))
    {
        _modeCombo->setCurrentIndex(s["mode"].toInt());
    }
    if (s.contains("applyScope"))
    {
        _applyScopeCombo->setCurrentIndex(s["applyScope"].toInt());
    }
    if (s.contains("applyTargetImagePath"))
    {
        int i = _applyTargetImageCombo->findData(s["applyTargetImagePath"].toString());
        if (i >= 0)
        {
            _applyTargetImageCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("overwriteExisting"))
    {
        _overwriteExistingCheck->setChecked(s["overwriteExisting"].toBool());
    }
    if (s.contains("quality"))
    {
        const int idx = _qualityCombo->findData(s["quality"].toInt());
        if (idx >= 0)
        {
            _qualityCombo->setCurrentIndex(idx);
        }
    }
    if (s.contains("threads")) _threadsSpin->setValue(s["threads"].toInt());
    if (s.contains("match_algorithm"))
    {
        setComboDataOrFirst(_matchAlgorithmCombo,
                            s.value(QStringLiteral("match_algorithm")).toString(QStringLiteral("lightglue")));
        refreshFeatureSuffixChoices();
    }
    if (s.contains("feature_suffix"))
    {
        const QString suffix = xjw::feature_match::normalizedFeatureSuffix(
            s.value(QStringLiteral("feature_suffix")).toString());
        const int idx = _featureSuffixCombo->findData(suffix);
        if (idx >= 0)
        {
            _featureSuffixCombo->setCurrentIndex(idx);
        }
    }
    if (s.contains("exifAuto"))      _exifAutoCheck->setChecked(s["exifAuto"].toBool());
    if (s.contains("defaultFocal"))  _defaultFocalSpin->setValue(s["defaultFocal"].toDouble());
    if (s.contains("sensorWidth"))   _sensorWidthSpin->setValue(s["sensorWidth"].toDouble());
    if (s.contains("fx")) _fxSpin->setValue(s["fx"].toDouble());
    if (s.contains("fy")) _fySpin->setValue(s["fy"].toDouble());
    if (s.contains("cx")) _cxSpin->setValue(s["cx"].toDouble());
    if (s.contains("cy")) _cySpin->setValue(s["cy"].toDouble());
    if (s.contains("distortionModel"))
    {
        int i = _distModelCombo->findText(s["distortionModel"].toString());
        if (i >= 0) _distModelCombo->setCurrentIndex(i);
    }
    if (s.contains("k1")) _k1Spin->setValue(s["k1"].toDouble());
    if (s.contains("k2")) _k2Spin->setValue(s["k2"].toDouble());
    if (s.contains("p1")) _p1Spin->setValue(s["p1"].toDouble());
    if (s.contains("p2")) _p2Spin->setValue(s["p2"].toDouble());
    if (s.contains("cameraImportMode"))
        _cameraImportModeCombo->setCurrentIndex(s["cameraImportMode"].toInt());
    if (s.contains("targetImagePath"))
    {
        int i = _targetImageCombo->findData(s["targetImagePath"].toString());
        if (i >= 0)
        {
            _targetImageCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("cameraFormat"))
    {
        int i = _cameraFormatCombo->findText(s["cameraFormat"].toString());
        if (i >= 0) _cameraFormatCombo->setCurrentIndex(i);
    }
    updateTargetUi();
    updateStatusText();
}

QString InitCameraPoseDialog::selectedFeatureSuffix() const
{
    if (!_featureSuffixCombo)
    {
        return QStringLiteral(".dsk");
    }
    const QVariant data = _featureSuffixCombo->currentData();
    const QString suffix = data.isValid() ? data.toString() : _featureSuffixCombo->currentText();
    const QString normalized = xjw::feature_match::normalizedFeatureSuffix(suffix);
    return normalized.isEmpty() ? QStringLiteral(".dsk") : normalized;
}

QString InitCameraPoseDialog::selectedFeatureAlgorithm() const
{
    const QString algorithm = xjw::feature_match::featureAlgorithmForSuffix(selectedFeatureSuffix());
    return algorithm.isEmpty() ? QStringLiteral("disk") : algorithm;
}

void InitCameraPoseDialog::refreshFeatureSuffixChoices()
{
    if (!_matchAlgorithmCombo || !_featureSuffixCombo)
    {
        return;
    }

    const QString previousSuffix = selectedFeatureSuffix();
    const QString algo = _matchAlgorithmCombo->currentData().toString();
    const QStringList compatibleSuffixes = xjw::feature_match::compatibleFeatureSuffixes(algo);

    QStringList suffixes;
    for (const QString &suffix : compatibleSuffixes)
    {
        const QString normalized = xjw::feature_match::normalizedFeatureSuffix(suffix);
        if (normalized.isEmpty())
        {
            continue;
        }
        if (_projectFeatureSuffixes.isEmpty() || _projectFeatureSuffixes.contains(normalized))
        {
            suffixes.append(normalized);
        }
    }

    if (suffixes.isEmpty())
    {
        suffixes = compatibleSuffixes.isEmpty()
            ? QStringList{QStringLiteral(".dsk")}
            : xjw::feature_match::normalizedFeatureSuffixes(compatibleSuffixes);
    }

    _featureSuffixCombo->blockSignals(true);
    _featureSuffixCombo->clear();
    for (const QString &suffix : suffixes)
    {
        _featureSuffixCombo->addItem(suffix, suffix);
    }

    int restoreIndex = _featureSuffixCombo->findData(previousSuffix);
    if (restoreIndex < 0 && _featureSuffixCombo->count() > 0)
    {
        restoreIndex = 0;
    }
    if (restoreIndex >= 0)
    {
        _featureSuffixCombo->setCurrentIndex(restoreIndex);
    }
    _featureSuffixCombo->blockSignals(false);
}

void InitCameraPoseDialog::onMatchPipelineChanged()
{
    refreshFeatureSuffixChoices();
    emitSettingsNow();
}

void InitCameraPoseDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void InitCameraPoseDialog::updateStatusText()
{
    const int mode = _modeCombo ? _modeCombo->currentIndex() : 0;
    QString text;
    if (mode == 0)
    {
        text = tr("当前将先为缺少相机文件的影像准备内参初值（EXIF 优先，默认焦距回退），"
              "再运行相对定向 / 增量 SFM 恢复真实位姿，并将结果回写到项目。需要至少 2 张存在匹配关系的影像。");
    }
    else if (mode == 1)
    {
        text = tr("当前将使用手工内参作为求解初值，运行相对定向 / 增量 SFM 恢复外参。"
                  "这适合已有标定结果但暂无标准相机文件的情况；cx/cy 设为自动时将使用图像中心。 ");
    }
    else
    {
        const bool exactImport = (_cameraImportModeCombo && _cameraImportModeCombo->currentIndex() == 0);
        text = exactImport
            ? tr("当前将复用现有“单影像 → 单相机文件”导入流程；成功后写入该影像的相机参数。")
            : tr("当前将复用现有“目录批量导入”流程；程序会按文件名自动匹配相机文件与项目影像。");
    }
    if (_statusLabel)
    {
        _statusLabel->setText(text);
    }
}

void InitCameraPoseDialog::updateTargetUi()
{
    const bool singleImage = (_applyScopeCombo && _applyScopeCombo->currentIndex() == 1);
    if (_applyTargetImageCombo)
    {
        _applyTargetImageCombo->setVisible(singleImage);
    }
    if (_applyForm && _applyTargetImageCombo)
    {
        if (QWidget *label = _applyForm->labelForField(_applyTargetImageCombo))
        {
            label->setVisible(singleImage);
        }
    }
    if (_applyHintLabel)
    {
        _applyHintLabel->setText(singleImage
            ? tr("求解时仍会联合当前项目中可用影像进行相对定向，但最终只回写这张目标影像的相机位姿。")
            : tr("会对当前项目中的全部参与影像运行相对定向 / 增量 SFM，并将成功恢复的相机位姿批量回写；未勾选覆盖时会跳过已有相机参数。"));
    }
}

void InitCameraPoseDialog::onRun()
{
    emit runRequested(collectSettings());
}

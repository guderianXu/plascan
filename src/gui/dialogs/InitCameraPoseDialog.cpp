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

QString normalizeFeatureSuffix(QString suffix)
{
    suffix = suffix.trimmed().toLower();
    if (suffix.isEmpty())
    {
        return QString();
    }
    if (!suffix.startsWith(QLatin1Char('.')))
    {
        suffix.prepend(QLatin1Char('.'));
    }
    return suffix;
}

QStringList normalizeFeatureSuffixes(const QStringList &suffixes)
{
    QStringList normalized;
    for (const QString &suffix : suffixes)
    {
        const QString value = normalizeFeatureSuffix(suffix);
        if (!value.isEmpty() && !normalized.contains(value))
        {
            normalized.append(value);
        }
    }
    return normalized;
}

QString featureAlgorithmForSuffix(const QString &suffix)
{
    const QString normalized = normalizeFeatureSuffix(suffix);
    if (normalized == QStringLiteral(".dsk"))
    {
        return QStringLiteral("disk");
    }
    if (normalized == QStringLiteral(".alk"))
    {
        return QStringLiteral("aliked");
    }
    if (normalized == QStringLiteral(".sp"))
    {
        return QStringLiteral("superpoint");
    }
    if (normalized == QStringLiteral(".sift"))
    {
        return QStringLiteral("sift");
    }
    if (normalized == QStringLiteral(".orb"))
    {
        return QStringLiteral("orb");
    }
    if (normalized == QStringLiteral(".akz"))
    {
        return QStringLiteral("akaze");
    }
    if (normalized == QStringLiteral(".dedode"))
    {
        return QStringLiteral("dedode");
    }
    return QString();
}

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

    m_modeCombo = ui.m_modeCombo;
    m_statusLabel = ui.m_statusLabel;
    m_applyBox = ui.m_applyBox;
    m_applyForm = ui.m_applyForm;
    m_modeStack = ui.m_modeStack;
    m_applyScopeCombo = ui.m_applyScopeCombo;
    m_applyTargetImageCombo = ui.m_applyTargetImageCombo;
    m_overwriteExistingCheck = ui.m_overwriteExistingCheck;
    m_applyHintLabel = ui.m_applyHintLabel;
    m_qualityCombo = ui.m_qualityCombo;
    m_threadsSpin = ui.m_threadsSpin;
    m_matchAlgorithmCombo = new QComboBox(this);
    m_matchAlgorithmCombo->setObjectName(QStringLiteral("m_matchAlgorithmCombo"));
    m_matchAlgorithmCombo->setToolTip(tr("选择初始化位姿时复用哪一种已生成匹配。"));
    m_featureSuffixCombo = new QComboBox(this);
    m_featureSuffixCombo->setObjectName(QStringLiteral("m_featureSuffixCombo"));
    m_featureSuffixCombo->setToolTip(tr("选择与匹配结果对应的特征文件类型。"));
    m_exifAutoCheck = ui.m_exifAutoCheck;
    m_defaultFocalSpin = ui.m_defaultFocalSpin;
    m_sensorWidthSpin = ui.m_sensorWidthSpin;
    m_intrinsicsForm = ui.m_intrinsicsForm;
    m_fxSpin = ui.m_fxSpin;
    m_fySpin = ui.m_fySpin;
    m_cxSpin = ui.m_cxSpin;
    m_cySpin = ui.m_cySpin;
    m_distModelCombo = ui.m_distModelCombo;
    m_k1Spin = ui.m_k1Spin;
    m_k2Spin = ui.m_k2Spin;
    m_p1Spin = ui.m_p1Spin;
    m_p2Spin = ui.m_p2Spin;
    m_cameraImportForm = ui.m_cameraImportForm;
    m_cameraImportModeCombo = ui.m_cameraImportModeCombo;
    m_targetImageCombo = ui.m_targetImageCombo;
    m_cameraFormatCombo = ui.m_cameraFormatCombo;
    m_cameraImportHintLabel = ui.m_cameraImportHintLabel;

    m_modeCombo->clear();
    m_modeCombo->addItems({
        tr("无相机文件 (从 EXIF 推断)"),
        tr("仅有内参矩阵"),
        tr("完整相机文件 (.tsai/.yaml/.xml)")
    });

    m_applyScopeCombo->clear();
    m_applyScopeCombo->addItems({
        tr("回写全部参与影像"),
        tr("仅回写目标影像")
    });

    m_qualityCombo->clear();
    m_qualityCombo->addItem(tr("快速"), 0);
    m_qualityCombo->addItem(tr("标准"), 1);
    m_qualityCombo->addItem(tr("高质量"), 2);
    m_qualityCombo->addItem(tr("最高质量"), 3);
    m_qualityCombo->setCurrentIndex(1);

    m_matchAlgorithmCombo->clear();
    m_matchAlgorithmCombo->addItem(tr("LightGlue"), QStringLiteral("lightglue"));
    m_matchAlgorithmCombo->addItem(tr("SuperGlue"), QStringLiteral("superglue"));
    m_matchAlgorithmCombo->addItem(tr("BF-Hamming (ORB)"), QStringLiteral("orb_bf_hamming"));
    m_matchAlgorithmCombo->addItem(tr("BF-L2 (SIFT)"), QStringLiteral("sift_bf_l2"));
    m_matchAlgorithmCombo->addItem(tr("FLANN (SIFT)"), QStringLiteral("sift_flann"));
    setComboDataOrFirst(m_matchAlgorithmCombo, QStringLiteral("lightglue"));
    if (ui.solveForm)
    {
        ui.solveForm->addRow(tr("匹配算法:"), m_matchAlgorithmCombo);
        ui.solveForm->addRow(tr("特征类型:"), m_featureSuffixCombo);
    }
    refreshFeatureSuffixChoices();

    m_distModelCombo->clear();
    m_distModelCombo->addItems({
        tr("无畸变"),
        tr("径向 (k1, k2)"),
        tr("Brown (k1, k2, p1, p2)")
    });
    m_distModelCombo->setCurrentIndex(2);

    m_cameraImportModeCombo->clear();
    m_cameraImportModeCombo->addItems({
        tr("为单张影像选择对应相机文件"),
        tr("选择目录并按文件名自动匹配")
    });

    m_cameraFormatCombo->clear();
    m_cameraFormatCombo->addItems({tr("Tsai (.tsai)"), tr("自动检测")});
    m_cameraFormatCombo->setCurrentIndex(1);

    ui.buttonBox->button(QDialogButtonBox::Ok)->setText(tr("写入相机初值"));

    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onModeChanged);
    connect(m_applyScopeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onInitTargetModeChanged);
    connect(m_cameraImportModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onCameraImportModeChanged);
    connect(m_distModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onDistortionModelChanged);

    auto changed = [this]()
    {
        emitSettingsNow();
    };
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_applyScopeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_applyTargetImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_overwriteExistingCheck, &QCheckBox::toggled, this, changed);
    connect(m_qualityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InitCameraPoseDialog::onMatchPipelineChanged);
    connect(m_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_cameraImportModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_targetImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_cameraFormatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_exifAutoCheck, &QCheckBox::toggled, this, changed);
    connect(m_defaultFocalSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_sensorWidthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_fxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_fySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_cxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_cySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_distModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_k1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_k2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_p1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_p2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);

    connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &InitCameraPoseDialog::onRun);
    connect(ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onModeChanged(0);
    onMatchPipelineChanged();
    onInitTargetModeChanged(0);
    onCameraImportModeChanged(0);
    onDistortionModelChanged(m_distModelCombo->currentIndex());
}

void InitCameraPoseDialog::onModeChanged(int index)
{
    m_modeStack->setCurrentIndex(index);
    if (m_applyBox)
    {
        m_applyBox->setVisible(index != 2);
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
        if (!field || !m_intrinsicsForm)
        {
            return;
        }
        field->setVisible(visible);
        if (QWidget *label = m_intrinsicsForm->labelForField(field))
        {
            label->setVisible(visible);
        }
    };

    toggleField(m_k1Spin, showK);
    toggleField(m_k2Spin, showK);
    toggleField(m_p1Spin, showP);
    toggleField(m_p2Spin, showP);
}

void InitCameraPoseDialog::onCameraImportModeChanged(int index)
{
    const bool exactImport = (index == 0);
    if (m_targetImageCombo)
    {
        m_targetImageCombo->setVisible(exactImport);
    }
    if (m_cameraImportForm && m_targetImageCombo)
    {
        if (QWidget *label = m_cameraImportForm->labelForField(m_targetImageCombo))
        {
            label->setVisible(exactImport);
        }
    }

    if (m_cameraImportHintLabel)
    {
        if (exactImport)
        {
            m_cameraImportHintLabel->setText(
                tr("点击[初始化位姿]后，将为所选影像打开现有的单文件导入流程。"
                   "适合逐张检查和精确绑定相机文件。"));
        }
        else
        {
            m_cameraImportHintLabel->setText(
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

    refillCombo(m_applyTargetImageCombo);
    refillCombo(m_targetImageCombo);
    updateTargetUi();
}

void InitCameraPoseDialog::setAvailableFeatureSuffixes(const QStringList &suffixes)
{
    m_projectFeatureSuffixes = normalizeFeatureSuffixes(suffixes);
    refreshFeatureSuffixChoices();
}

QJsonObject InitCameraPoseDialog::collectSettings() const
{
    QJsonObject o;
    o["mode"] = m_modeCombo->currentIndex();
    o["applyScope"] = m_applyScopeCombo->currentIndex();
    o["applyTargetImagePath"] = m_applyTargetImageCombo->currentData().toString();
    o["applyTargetImageName"] = m_applyTargetImageCombo->currentText();
    o["overwriteExisting"] = m_overwriteExistingCheck->isChecked();
    o["quality"] = m_qualityCombo->currentData().toInt();
    o["threads"] = m_threadsSpin->value();
    o["match_algorithm"] = m_matchAlgorithmCombo->currentData().toString();
    o["feature_suffix"] = selectedFeatureSuffix();
    o["feature_algorithm"] = selectedFeatureAlgorithm();
    // 模式 0
    o["exifAuto"]     = m_exifAutoCheck->isChecked();
    o["defaultFocal"]  = m_defaultFocalSpin->value();
    o["sensorWidth"]   = m_sensorWidthSpin->value();
    // 模式 1
    o["fx"] = m_fxSpin->value();
    o["fy"] = m_fySpin->value();
    o["cx"] = m_cxSpin->value();
    o["cy"] = m_cySpin->value();
    o["distortionModel"] = m_distModelCombo->currentText();
    o["k1"] = m_k1Spin->value();
    o["k2"] = m_k2Spin->value();
    o["p1"] = m_p1Spin->value();
    o["p2"] = m_p2Spin->value();
    // 模式 2
    o["cameraImportMode"] = m_cameraImportModeCombo->currentIndex();
    o["targetImagePath"]  = m_targetImageCombo->currentData().toString();
    o["targetImageName"]  = m_targetImageCombo->currentText();
    o["cameraFormat"] = m_cameraFormatCombo->currentText();
    return o;
}

void InitCameraPoseDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("mode"))
    {
        m_modeCombo->setCurrentIndex(s["mode"].toInt());
    }
    if (s.contains("applyScope"))
    {
        m_applyScopeCombo->setCurrentIndex(s["applyScope"].toInt());
    }
    if (s.contains("applyTargetImagePath"))
    {
        int i = m_applyTargetImageCombo->findData(s["applyTargetImagePath"].toString());
        if (i >= 0)
        {
            m_applyTargetImageCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("overwriteExisting"))
    {
        m_overwriteExistingCheck->setChecked(s["overwriteExisting"].toBool());
    }
    if (s.contains("quality"))
    {
        const int idx = m_qualityCombo->findData(s["quality"].toInt());
        if (idx >= 0)
        {
            m_qualityCombo->setCurrentIndex(idx);
        }
    }
    if (s.contains("threads")) m_threadsSpin->setValue(s["threads"].toInt());
    if (s.contains("match_algorithm"))
    {
        setComboDataOrFirst(m_matchAlgorithmCombo,
                            s.value(QStringLiteral("match_algorithm")).toString(QStringLiteral("lightglue")));
        refreshFeatureSuffixChoices();
    }
    if (s.contains("feature_suffix"))
    {
        const QString suffix = normalizeFeatureSuffix(s.value(QStringLiteral("feature_suffix")).toString());
        const int idx = m_featureSuffixCombo->findData(suffix);
        if (idx >= 0)
        {
            m_featureSuffixCombo->setCurrentIndex(idx);
        }
    }
    if (s.contains("exifAuto"))      m_exifAutoCheck->setChecked(s["exifAuto"].toBool());
    if (s.contains("defaultFocal"))  m_defaultFocalSpin->setValue(s["defaultFocal"].toDouble());
    if (s.contains("sensorWidth"))   m_sensorWidthSpin->setValue(s["sensorWidth"].toDouble());
    if (s.contains("fx")) m_fxSpin->setValue(s["fx"].toDouble());
    if (s.contains("fy")) m_fySpin->setValue(s["fy"].toDouble());
    if (s.contains("cx")) m_cxSpin->setValue(s["cx"].toDouble());
    if (s.contains("cy")) m_cySpin->setValue(s["cy"].toDouble());
    if (s.contains("distortionModel"))
    {
        int i = m_distModelCombo->findText(s["distortionModel"].toString());
        if (i >= 0) m_distModelCombo->setCurrentIndex(i);
    }
    if (s.contains("k1")) m_k1Spin->setValue(s["k1"].toDouble());
    if (s.contains("k2")) m_k2Spin->setValue(s["k2"].toDouble());
    if (s.contains("p1")) m_p1Spin->setValue(s["p1"].toDouble());
    if (s.contains("p2")) m_p2Spin->setValue(s["p2"].toDouble());
    if (s.contains("cameraImportMode"))
        m_cameraImportModeCombo->setCurrentIndex(s["cameraImportMode"].toInt());
    if (s.contains("targetImagePath"))
    {
        int i = m_targetImageCombo->findData(s["targetImagePath"].toString());
        if (i >= 0)
        {
            m_targetImageCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("cameraFormat"))
    {
        int i = m_cameraFormatCombo->findText(s["cameraFormat"].toString());
        if (i >= 0) m_cameraFormatCombo->setCurrentIndex(i);
    }
    updateTargetUi();
    updateStatusText();
}

QString InitCameraPoseDialog::selectedFeatureSuffix() const
{
    if (!m_featureSuffixCombo)
    {
        return QStringLiteral(".dsk");
    }
    const QVariant data = m_featureSuffixCombo->currentData();
    const QString suffix = data.isValid() ? data.toString() : m_featureSuffixCombo->currentText();
    const QString normalized = normalizeFeatureSuffix(suffix);
    return normalized.isEmpty() ? QStringLiteral(".dsk") : normalized;
}

QString InitCameraPoseDialog::selectedFeatureAlgorithm() const
{
    const QString algorithm = featureAlgorithmForSuffix(selectedFeatureSuffix());
    return algorithm.isEmpty() ? QStringLiteral("disk") : algorithm;
}

void InitCameraPoseDialog::refreshFeatureSuffixChoices()
{
    if (!m_matchAlgorithmCombo || !m_featureSuffixCombo)
    {
        return;
    }

    const QString previousSuffix = selectedFeatureSuffix();
    const QString algo = m_matchAlgorithmCombo->currentData().toString();
    const QStringList compatibleSuffixes = xjw::feature_match::compatibleFeatureSuffixes(algo);

    QStringList suffixes;
    for (const QString &suffix : compatibleSuffixes)
    {
        const QString normalized = normalizeFeatureSuffix(suffix);
        if (normalized.isEmpty())
        {
            continue;
        }
        if (m_projectFeatureSuffixes.isEmpty() || m_projectFeatureSuffixes.contains(normalized))
        {
            suffixes.append(normalized);
        }
    }

    if (suffixes.isEmpty())
    {
        suffixes = compatibleSuffixes.isEmpty()
            ? QStringList{QStringLiteral(".dsk")}
            : normalizeFeatureSuffixes(compatibleSuffixes);
    }

    m_featureSuffixCombo->blockSignals(true);
    m_featureSuffixCombo->clear();
    for (const QString &suffix : suffixes)
    {
        m_featureSuffixCombo->addItem(suffix, suffix);
    }

    int restoreIndex = m_featureSuffixCombo->findData(previousSuffix);
    if (restoreIndex < 0 && m_featureSuffixCombo->count() > 0)
    {
        restoreIndex = 0;
    }
    if (restoreIndex >= 0)
    {
        m_featureSuffixCombo->setCurrentIndex(restoreIndex);
    }
    m_featureSuffixCombo->blockSignals(false);
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
    const int mode = m_modeCombo ? m_modeCombo->currentIndex() : 0;
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
        const bool exactImport = (m_cameraImportModeCombo && m_cameraImportModeCombo->currentIndex() == 0);
        text = exactImport
            ? tr("当前将复用现有“单影像 → 单相机文件”导入流程；成功后写入该影像的相机参数。")
            : tr("当前将复用现有“目录批量导入”流程；程序会按文件名自动匹配相机文件与项目影像。");
    }
    if (m_statusLabel)
    {
        m_statusLabel->setText(text);
    }
}

void InitCameraPoseDialog::updateTargetUi()
{
    const bool singleImage = (m_applyScopeCombo && m_applyScopeCombo->currentIndex() == 1);
    if (m_applyTargetImageCombo)
    {
        m_applyTargetImageCombo->setVisible(singleImage);
    }
    if (m_applyForm && m_applyTargetImageCombo)
    {
        if (QWidget *label = m_applyForm->labelForField(m_applyTargetImageCombo))
        {
            label->setVisible(singleImage);
        }
    }
    if (m_applyHintLabel)
    {
        m_applyHintLabel->setText(singleImage
            ? tr("求解时仍会联合当前项目中可用影像进行相对定向，但最终只回写这张目标影像的相机位姿。")
            : tr("会对当前项目中的全部参与影像运行相对定向 / 增量 SFM，并将成功恢复的相机位姿批量回写；未勾选覆盖时会跳过已有相机参数。"));
    }
}

void InitCameraPoseDialog::onRun()
{
    emit runRequested(collectSettings());
}

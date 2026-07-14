#include "DepthMapEstimateDialog.h"
#include "ui_DepthMapEstimateDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QFileInfo>
#include <QDialogButtonBox>
#include <QPushButton>

#include <iterator>

namespace
{

QString costFunctionToolTip(const QString &name)
{
    if (name == QStringLiteral("NCC"))
    {
        return QStringLiteral("NCC（归一化互相关）：比较两个局部窗口的相关性，对整体亮度缩放较稳，常用于 PatchMatch 深度估计。计算量比 AD/SAD 更高。");
    }
    if (name == QStringLiteral("Census"))
    {
        return QStringLiteral("Census：把局部灰度排序编码成二进制描述子，再比较汉明距离。对光照变化和低纹理区域更稳，但可能损失精细灰度差异。");
    }
    if (name == QStringLiteral("SAD"))
    {
        return QStringLiteral("SAD/AD（灰度绝对差之和）：在窗口内累加左右影像灰度绝对差。速度快、直观，但对曝光差、阴影和辐射差异敏感。");
    }
    if (name == QStringLiteral("ZNCC"))
    {
        return QStringLiteral("ZNCC（零均值归一化互相关）：先去除窗口平均亮度再做 NCC，对局部亮度偏移更稳，适合有曝光差的影像。");
    }

    return QStringLiteral("SD（平方差）会放大较大灰度误差，Ternary Census（三值 Census）会给微小灰度变化留容差；二者是密集匹配中常见代价函数。");
}

void updateCostFunctionToolTip(QComboBox *combo)
{
    if (!combo)
    {
        return;
    }

    const QString summary =
        QStringLiteral("代价函数决定如何评价同一地面点在两幅影像局部窗口中的相似程度。当前选项：%1\n\n%2")
            .arg(combo->currentText(), costFunctionToolTip(combo->currentText()));
    combo->setToolTip(summary);
}

void installCostFunctionToolTips(QComboBox *combo)
{
    if (!combo)
    {
        return;
    }

    for (int i = 0; i < combo->count(); ++i)
    {
        combo->setItemData(i, costFunctionToolTip(combo->itemText(i)), Qt::ToolTipRole);
    }
    updateCostFunctionToolTip(combo);
}

} // namespace

DepthMapEstimateDialog::DepthMapEstimateDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::DepthMapEstimateDialog form;
    form.setupUi(this);

    _atResultCombo = form.m_atResultCombo;
    _presetCombo = form.m_presetCombo;
    _resScaleSpin = form.m_resScaleSpin;
    _iterationsSpin = form.m_iterationsSpin;
    _costFuncCombo = form.m_costFuncCombo;
    _propagCombo = form.m_propagCombo;
    _patchSizeSpin = form.m_patchSizeSpin;
    _minViewsSpin = form.m_minViewsSpin;
    _depthMinSpin = form.m_depthMinSpin;
    _depthMaxSpin = form.m_depthMaxSpin;
    _confidenceSpin = form.m_confidenceSpin;
    _normalMapCheck = form.m_normalMapCheck;
    _sceneProfileCombo = form.m_sceneProfileCombo;
    _depthFilterCombo = form.m_depthFilterCombo;
    _savePyramidLevelsCheck = form.m_savePyramidLevelsCheck;
    _cudaCheck = form.m_cudaCheck;
    _tileWSpin = form.m_tileWSpin;
    _tileHSpin = form.m_tileHSpin;
    _threadsSpin = form.m_threadsSpin;

    _presetCombo->clear();
    _presetCombo->addItem(tr("最高"), QStringLiteral("highest"));
    _presetCombo->addItem(tr("高"), QStringLiteral("high"));
    _presetCombo->addItem(tr("中"), QStringLiteral("medium"));
    _presetCombo->addItem(tr("低"), QStringLiteral("low"));
    _presetCombo->addItem(tr("最低"), QStringLiteral("lowest"));
    _presetCombo->addItem(tr("自定义"), QStringLiteral("custom"));

    _sceneProfileCombo->clear();
    _sceneProfileCombo->addItem(tr("自动识别"), QStringLiteral("auto"));
    _sceneProfileCombo->addItem(tr("环拍任意 3D"), QStringLiteral("orbital_object"));
    _sceneProfileCombo->addItem(tr("航测地形"), QStringLiteral("aerial_terrain"));

    _depthFilterCombo->clear();
    _depthFilterCombo->addItem(tr("自动"), QStringLiteral("auto"));
    _depthFilterCombo->addItem(tr("温和"), QStringLiteral("mild"));
    _depthFilterCombo->addItem(tr("中等"), QStringLiteral("moderate"));
    _depthFilterCombo->addItem(tr("强过滤"), QStringLiteral("aggressive"));

    _estimateLabel = nullptr; // 占位，不再显示"预计: -"
    form.buttonBox->button(QDialogButtonBox::Ok)->setText(tr("开始估计"));
    _atResultCombo->setItemData(0, -1);
    installCostFunctionToolTips(_costFuncCombo);

    connect(_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DepthMapEstimateDialog::onPresetChanged);

    // 任意参数变更时：若不在应用预设过程中，则自动切换到"自定义"
    auto changed = [this]()
    {
        if (!_applyingPreset)
        {
            _presetCombo->blockSignals(true);
            _presetCombo->setCurrentIndex(_presetCombo->findData(QStringLiteral("custom")));
            _presetCombo->blockSignals(false);
        }
        emitSettingsNow();
    };
    connect(_resScaleSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_atResultCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_iterationsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_costFuncCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, changed]()
    {
        updateCostFunctionToolTip(_costFuncCombo);
        changed();
    });
    connect(_propagCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_patchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_minViewsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_depthMinSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_depthMaxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_confidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_normalMapCheck, &QCheckBox::toggled, this, changed);
    connect(_sceneProfileCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_depthFilterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_savePyramidLevelsCheck, &QCheckBox::toggled, this, changed);
    connect(_cudaCheck, &QCheckBox::toggled, this, changed);
    connect(_tileWSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_tileHSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(form.buttonBox, &QDialogButtonBox::accepted, this, &DepthMapEstimateDialog::onRun);
    connect(form.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onPresetChanged(_presetCombo->findData(QStringLiteral("medium")));
}

void DepthMapEstimateDialog::setAvailableAtResults(const QJsonArray &atResults)
{
    if (!_atResultCombo)
    {
        return;
    }

    _atResultCombo->blockSignals(true);
    _atResultCombo->clear();

    if (atResults.isEmpty())
    {
        _atResultCombo->addItem(tr("（请先运行空三）"), -2);
        _atResultCombo->setEnabled(false);
    }
    else
    {
        _atResultCombo->setEnabled(true);
        _atResultCombo->addItem(tr("最新 AT 结果（推荐）"), -1);
        for (int i = 0; i < atResults.size(); ++i)
        {
            const QJsonObject item = atResults.at(i).toObject();
            const int idx = item.value(QStringLiteral("index")).toInt(i);
            const int imgCnt = item.value(QStringLiteral("image_count")).toInt(0);
            const int ptsCnt = item.value(QStringLiteral("sparse_point_count")).toInt(0);
            const QString createdAt = item.value(QStringLiteral("created_at")).toString().left(10);
            const QString outDir = QFileInfo(item.value(QStringLiteral("output_dir")).toString()).fileName();
            const QString label = QStringLiteral("[%1] %2  %3 张影像  %4 点  (%5)")
                                      .arg(idx)
                                      .arg(outDir)
                                      .arg(imgCnt)
                                      .arg(ptsCnt)
                                      .arg(createdAt);
            _atResultCombo->addItem(label, idx);
        }

        const int desired = _pendingAtIndex;
        int comboIndex = -1;
        if (desired >= -1)
        {
            comboIndex = _atResultCombo->findData(desired);
        }
        if (comboIndex < 0)
        {
            comboIndex = _atResultCombo->findData(-1);
        }
        if (comboIndex < 0)
        {
            comboIndex = 0;
        }
        _atResultCombo->setCurrentIndex(comboIndex);
    }

    _atResultCombo->blockSignals(false);
    emitSettingsNow();
}

void DepthMapEstimateDialog::onPresetChanged(int index)
{
    if (index < 0 || _presetCombo->itemData(index).toString() == QStringLiteral("custom"))
    {
        return;
    }
    struct P
    {
        double rs;
        int it;
        int cf;
        int pr;
        int ps;
        int mv;
        double co;
    };
    static const P kP[] = {
        {1.0000, 16, 0, 0, 15, 8, 0.72},
        {0.5000, 12, 0, 0, 13, 7, 0.68},
        {0.2500, 8,  0, 0, 11, 6, 0.60},
        {0.1250, 4,  0, 0, 9,  3, 0.30},
        {0.0625, 3,  0, 0, 7,  2, 0.22},
    };
    if (index >= static_cast<int>(std::size(kP)))
    {
        return;
    }
    const auto &p = kP[index];
    _applyingPreset = true;
    _resScaleSpin->setValue(p.rs);
    _iterationsSpin->setValue(p.it);
    _costFuncCombo->setCurrentIndex(p.cf);
    _propagCombo->setCurrentIndex(p.pr);
    _patchSizeSpin->setValue(p.ps);
    _minViewsSpin->setValue(p.mv);
    _confidenceSpin->setValue(p.co);
    _applyingPreset = false;
    emitSettingsNow();
}

QJsonObject DepthMapEstimateDialog::collectSettings() const
{
    QJsonObject o;
    const QVariant atData = _atResultCombo ? _atResultCombo->currentData() : QVariant(-1);
    const int atIndex = atData.isValid() ? atData.toInt() : -1;
    o["at_index"] = atIndex;
    o["at_selection_mode"] = (atIndex < 0) ? QStringLiteral("latest") : QStringLiteral("fixed");
    o["preset"] = _presetCombo->currentText();
    o["qualityProfile"] = _presetCombo->currentData().toString();
    o["resScale"] = _resScaleSpin->value();
    o["iterations"] = _iterationsSpin->value();
    o["costFunction"] = _costFuncCombo->currentText();
    o["propagation"] = _propagCombo->currentText();
    o["patchSize"] = _patchSizeSpin->value();
    o["minViews"] = _minViewsSpin->value();
    o["depthMin"] = _depthMinSpin->value();
    o["depthMax"] = _depthMaxSpin->value();
    o["confidence"] = _confidenceSpin->value();
    o["normalMap"] = _normalMapCheck->isChecked();
    o["sceneProfile"] = _sceneProfileCombo->currentData().toString();
    o["depthFilterMode"] = _depthFilterCombo->currentData().toString();
    o["saveIntermediatePyramidLevels"] = _savePyramidLevelsCheck->isChecked();
    o["cuda"] = _cudaCheck->isChecked();
    o["tileWidth"] = _tileWSpin->value();
    o["tileHeight"] = _tileHSpin->value();
    o["threads"] = _threadsSpin->value();
    return o;
}

void DepthMapEstimateDialog::applySettings(const QJsonObject &s)
{
    _applyingPreset = true;
    const QString profile_id = s.value(QStringLiteral("qualityProfile")).toString();
    if (!profile_id.isEmpty())
    {
        const int profile_index = _presetCombo->findData(profile_id);
        if (profile_index >= 0)
        {
            _presetCombo->setCurrentIndex(profile_index);
        }
    }

    if (s.contains("at_selection_mode") && s.value("at_selection_mode").toString() == QStringLiteral("latest"))
    {
        _pendingAtIndex = -1;
    }
    else if (s.contains("at_index"))
    {
        _pendingAtIndex = s.value("at_index").toInt(-1);
    }

    if (_atResultCombo && _atResultCombo->count() > 0)
    {
        const int idx = _atResultCombo->findData(_pendingAtIndex);
        if (idx >= 0)
        {
            _atResultCombo->setCurrentIndex(idx);
        }
    }

    if (s.contains("resScale"))
    {
        _resScaleSpin->setValue(s["resScale"].toDouble());
    }
    if (s.contains("iterations"))
    {
        _iterationsSpin->setValue(s["iterations"].toInt());
    }
    if (s.contains("costFunction"))
    {
        int i = _costFuncCombo->findText(s["costFunction"].toString());
        if (i >= 0)
        {
            _costFuncCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("propagation"))
    {
        int i = _propagCombo->findText(s["propagation"].toString());
        if (i >= 0)
        {
            _propagCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("patchSize"))
    {
        _patchSizeSpin->setValue(s["patchSize"].toInt());
    }
    if (s.contains("minViews"))
    {
        _minViewsSpin->setValue(s["minViews"].toInt());
    }
    if (s.contains("depthMin"))
    {
        _depthMinSpin->setValue(s["depthMin"].toDouble());
    }
    if (s.contains("depthMax"))
    {
        _depthMaxSpin->setValue(s["depthMax"].toDouble());
    }
    if (s.contains("confidence"))
    {
        _confidenceSpin->setValue(s["confidence"].toDouble());
    }
    if (s.contains("normalMap"))
    {
        _normalMapCheck->setChecked(s["normalMap"].toBool());
    }
    if (s.contains("sceneProfile"))
    {
        const int index = _sceneProfileCombo->findData(s["sceneProfile"].toString());
        if (index >= 0)
        {
            _sceneProfileCombo->setCurrentIndex(index);
        }
    }
    if (s.contains("depthFilterMode"))
    {
        const int index = _depthFilterCombo->findData(s["depthFilterMode"].toString());
        if (index >= 0)
        {
            _depthFilterCombo->setCurrentIndex(index);
        }
    }
    if (s.contains("saveIntermediatePyramidLevels"))
    {
        _savePyramidLevelsCheck->setChecked(s["saveIntermediatePyramidLevels"].toBool());
    }
    if (s.contains("cuda"))
    {
        _cudaCheck->setChecked(s["cuda"].toBool());
    }
    if (s.contains("tileWidth"))
    {
        _tileWSpin->setValue(s["tileWidth"].toInt());
    }
    if (s.contains("tileHeight"))
    {
        _tileHSpin->setValue(s["tileHeight"].toInt());
    }
    if (s.contains("threads"))
    {
        _threadsSpin->setValue(s["threads"].toInt());
    }
    _applyingPreset = false;
    emitSettingsNow();
}

void DepthMapEstimateDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void DepthMapEstimateDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}

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

DepthMapEstimateDialog::DepthMapEstimateDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::DepthMapEstimateDialog form;
    form.setupUi(this);

    m_atResultCombo = form.m_atResultCombo;
    m_presetCombo = form.m_presetCombo;
    m_resScaleSpin = form.m_resScaleSpin;
    m_iterationsSpin = form.m_iterationsSpin;
    m_costFuncCombo = form.m_costFuncCombo;
    m_propagCombo = form.m_propagCombo;
    m_patchSizeSpin = form.m_patchSizeSpin;
    m_minViewsSpin = form.m_minViewsSpin;
    m_depthMinSpin = form.m_depthMinSpin;
    m_depthMaxSpin = form.m_depthMaxSpin;
    m_confidenceSpin = form.m_confidenceSpin;
    m_normalMapCheck = form.m_normalMapCheck;
    m_cudaCheck = form.m_cudaCheck;
    m_tileWSpin = form.m_tileWSpin;
    m_tileHSpin = form.m_tileHSpin;
    m_threadsSpin = form.m_threadsSpin;

    m_estimateLabel = nullptr; // 占位，不再显示"预计: -"
    form.buttonBox->button(QDialogButtonBox::Ok)->setText(tr("开始估计"));
    m_atResultCombo->setItemData(0, -1);

    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DepthMapEstimateDialog::onPresetChanged);

    // 任意参数变更时：若不在应用预设过程中，则自动切换到"自定义"
    auto changed = [this]() {
        if (!m_applyingPreset) {
            m_presetCombo->blockSignals(true);
            m_presetCombo->setCurrentIndex(3);
            m_presetCombo->blockSignals(false);
        }
        emitSettingsNow();
    };
    connect(m_resScaleSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_atResultCombo,  QOverload<int>::of(&QComboBox::currentIndexChanged),  this, changed);
    connect(m_iterationsSpin, QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_costFuncCombo,  QOverload<int>::of(&QComboBox::currentIndexChanged),  this, changed);
    connect(m_propagCombo,    QOverload<int>::of(&QComboBox::currentIndexChanged),  this, changed);
    connect(m_patchSizeSpin,  QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_minViewsSpin,   QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_depthMinSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_depthMaxSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_confidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_normalMapCheck, &QCheckBox::toggled,                                  this, changed);
    connect(m_cudaCheck,      &QCheckBox::toggled,                                  this, changed);
    connect(m_tileWSpin,      QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_tileHSpin,      QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_threadsSpin,    QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);

    connect(form.buttonBox, &QDialogButtonBox::accepted, this, &DepthMapEstimateDialog::onRun);
    connect(form.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onPresetChanged(1);
}

void DepthMapEstimateDialog::setAvailableAtResults(const QJsonArray &atResults)
{
    if (!m_atResultCombo)
    {
        return;
    }

    m_atResultCombo->blockSignals(true);
    m_atResultCombo->clear();

    if (atResults.isEmpty())
    {
        m_atResultCombo->addItem(tr("（请先运行空三）"), -2);
        m_atResultCombo->setEnabled(false);
    }
    else
    {
        m_atResultCombo->setEnabled(true);
        m_atResultCombo->addItem(tr("最新 AT 结果（推荐）"), -1);
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
            m_atResultCombo->addItem(label, idx);
        }

        const int desired = m_pendingAtIndex;
        int comboIndex = -1;
        if (desired >= -1)
        {
            comboIndex = m_atResultCombo->findData(desired);
        }
        if (comboIndex < 0)
        {
            comboIndex = m_atResultCombo->findData(-1);
        }
        if (comboIndex < 0)
        {
            comboIndex = 0;
        }
        m_atResultCombo->setCurrentIndex(comboIndex);
    }

    m_atResultCombo->blockSignals(false);
    emitSettingsNow();
}

void DepthMapEstimateDialog::onPresetChanged(int index)
{
    if (index == 3) return;
    struct P { double rs; int it; int cf; int pr; int ps; int mv; double co; };
    static const P kP[] = {
        {0.25, 3, 0, 0, 7,  2, 0.3},
        {0.50, 6, 0, 0, 11, 3, 0.5},
        {1.00, 10,0, 0, 15, 4, 0.7},
    };
    const auto &p = kP[index];
    m_applyingPreset = true;
    m_resScaleSpin->setValue(p.rs);
    m_iterationsSpin->setValue(p.it);
    m_costFuncCombo->setCurrentIndex(p.cf);
    m_propagCombo->setCurrentIndex(p.pr);
    m_patchSizeSpin->setValue(p.ps);
    m_minViewsSpin->setValue(p.mv);
    m_confidenceSpin->setValue(p.co);
    m_applyingPreset = false;
    emitSettingsNow();
}

QJsonObject DepthMapEstimateDialog::collectSettings() const
{
    QJsonObject o;
    const QVariant atData = m_atResultCombo ? m_atResultCombo->currentData() : QVariant(-1);
    const int atIndex = atData.isValid() ? atData.toInt() : -1;
    o["at_index"] = atIndex;
    o["at_selection_mode"] = (atIndex < 0) ? QStringLiteral("latest") : QStringLiteral("fixed");
    o["preset"]       = m_presetCombo->currentText();
    o["resScale"]     = m_resScaleSpin->value();
    o["iterations"]   = m_iterationsSpin->value();
    o["costFunction"] = m_costFuncCombo->currentText();
    o["propagation"]  = m_propagCombo->currentText();
    o["patchSize"]    = m_patchSizeSpin->value();
    o["minViews"]     = m_minViewsSpin->value();
    o["depthMin"]     = m_depthMinSpin->value();
    o["depthMax"]     = m_depthMaxSpin->value();
    o["confidence"]   = m_confidenceSpin->value();
    o["normalMap"]    = m_normalMapCheck->isChecked();
    o["cuda"]         = m_cudaCheck->isChecked();
    o["tileWidth"]    = m_tileWSpin->value();
    o["tileHeight"]   = m_tileHSpin->value();
    o["threads"]      = m_threadsSpin->value();
    return o;
}

void DepthMapEstimateDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("at_selection_mode") && s.value("at_selection_mode").toString() == QStringLiteral("latest"))
    {
        m_pendingAtIndex = -1;
    }
    else if (s.contains("at_index"))
    {
        m_pendingAtIndex = s.value("at_index").toInt(-1);
    }

    if (m_atResultCombo && m_atResultCombo->count() > 0)
    {
        const int idx = m_atResultCombo->findData(m_pendingAtIndex);
        if (idx >= 0)
        {
            m_atResultCombo->setCurrentIndex(idx);
        }
    }

    if (s.contains("resScale"))     m_resScaleSpin->setValue(s["resScale"].toDouble());
    if (s.contains("iterations"))   m_iterationsSpin->setValue(s["iterations"].toInt());
    if (s.contains("costFunction"))
    {
        int i = m_costFuncCombo->findText(s["costFunction"].toString());
        if (i >= 0) m_costFuncCombo->setCurrentIndex(i);
    }
    if (s.contains("propagation"))
    {
        int i = m_propagCombo->findText(s["propagation"].toString());
        if (i >= 0) m_propagCombo->setCurrentIndex(i);
    }
    if (s.contains("patchSize"))    m_patchSizeSpin->setValue(s["patchSize"].toInt());
    if (s.contains("minViews"))     m_minViewsSpin->setValue(s["minViews"].toInt());
    if (s.contains("depthMin"))     m_depthMinSpin->setValue(s["depthMin"].toDouble());
    if (s.contains("depthMax"))     m_depthMaxSpin->setValue(s["depthMax"].toDouble());
    if (s.contains("confidence"))   m_confidenceSpin->setValue(s["confidence"].toDouble());
    if (s.contains("normalMap"))    m_normalMapCheck->setChecked(s["normalMap"].toBool());
    if (s.contains("cuda"))         m_cudaCheck->setChecked(s["cuda"].toBool());
    if (s.contains("tileWidth"))    m_tileWSpin->setValue(s["tileWidth"].toInt());
    if (s.contains("tileHeight"))   m_tileHSpin->setValue(s["tileHeight"].toInt());
    if (s.contains("threads"))      m_threadsSpin->setValue(s["threads"].toInt());
}

void DepthMapEstimateDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void DepthMapEstimateDialog::onRun() { emit runRequested(collectSettings()); accept(); }

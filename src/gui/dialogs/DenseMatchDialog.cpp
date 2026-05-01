// =============================================================================
// 文件: DenseMatchDialog.cpp
// 功能: 密集匹配对话框实现 — 业务逻辑（UI 构建在 DenseMatchDialogUi.cpp）
// =============================================================================
#include "DenseMatchDialog.h"
#include "ProjectManager.h"
#include "ProjectIO.h"
#include "ProjectData.h"

#include <QLabel>
#include <QListWidget>
#include <QTableWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QDataStream>

DenseMatchDialog::DenseMatchDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
{
    setWindowTitle(tr("密集匹配"));
    resize(900, 560);
    setupUi();
    loadProjectImages();
}

void DenseMatchDialog::loadProjectImages()
{
    if (!m_projectManager) return;

    m_allImages = m_projectManager->getAllImages();
    if (m_allImages.isEmpty()) return;

    m_imageList->blockSignals(true);
    for (const QString &imgPath : m_allImages)
    {
        QFileInfo fi(imgPath);
        auto *item = new QListWidgetItem(fi.fileName());
        item->setData(Qt::UserRole, imgPath);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
        m_imageList->addItem(item);
    }
    m_imageList->blockSignals(false);

    if (m_projectManager && !m_projectManager->currentProjectPath().isEmpty())
    {
        const QString assetsDir = ProjectIO::projectAssetsDir(
            m_projectManager->currentProjectPath());
        m_outputEdit->setText(
            QDir(assetsDir).filePath(QStringLiteral("dense_match")));
    }

    refreshMatchPairs();
}

void DenseMatchDialog::refreshMatchPairs()
{
    if (!m_projectManager) return;

    m_matchPairs.clear();
    m_matchTable->setRowCount(0);

    QStringList selected;
    for (int i = 0; i < m_imageList->count(); ++i)
    {
        auto *item = m_imageList->item(i);
        if (item->checkState() == Qt::Checked)
            selected.append(item->data(Qt::UserRole).toString());
    }

    if (selected.size() < 2)
    {
        m_matchCountLabel->setText(
            tr("请至少选择 2 张影像（当前选中 %1 张）").arg(selected.size()));
        return;
    }

    const QString assetsDir = ProjectIO::projectAssetsDir(
        m_projectManager->currentProjectPath());
    const QString matchDir = QDir(assetsDir).filePath(QStringLiteral("matches"));

    const auto *projData = m_projectManager->projectData();
    if (!projData) return;

    for (int i = 0; i < selected.size(); ++i)
    {
        for (int j = i + 1; j < selected.size(); ++j)
        {
            const QString &imgA = selected[i];
            const QString &imgB = selected[j];

            QString mf = projData->findMatchFile(imgA, imgB);
            if (mf.isEmpty())
            {
                QFileInfo fiA(imgA);
                QFileInfo fiB(imgB);
                QString base = fiA.completeBaseName() + "__"
                               + fiB.completeBaseName() + ".match";
                QString candidate = QDir(matchDir).filePath(base);
                if (QFile::exists(candidate))
                    mf = candidate;
                else
                {
                    base = fiB.completeBaseName() + "__"
                           + fiA.completeBaseName() + ".match";
                    candidate = QDir(matchDir).filePath(base);
                    if (QFile::exists(candidate))
                        mf = candidate;
                }
            }

            if (!mf.isEmpty())
            {
                MatchPairInfo info;
                info.imgA = imgA;
                info.imgB = imgB;
                info.matchFile = mf;
                QFile mFile(mf);
                if (mFile.open(QIODevice::ReadOnly))
                {
                    QDataStream ds(&mFile);
                    ds.setByteOrder(QDataStream::BigEndian);
                    qint32 nMatches = 0;
                    ds >> nMatches;
                    info.numMatches = nMatches;
                    mFile.close();
                }
                m_matchPairs.append(info);
            }
        }
    }

    for (const auto &info : m_matchPairs)
    {
        int row = m_matchTable->rowCount();
        m_matchTable->insertRow(row);
        m_matchTable->setItem(row, 0,
            new QTableWidgetItem(QFileInfo(info.imgA).fileName()));
        m_matchTable->setItem(row, 1,
            new QTableWidgetItem(QFileInfo(info.imgB).fileName()));
        m_matchTable->setItem(row, 2,
            new QTableWidgetItem(QString::number(info.numMatches)));
    }

    m_matchCountLabel->setText(
        tr("共 %1 个匹配对（从 %2 张选中影像中）")
            .arg(m_matchPairs.size())
            .arg(selected.size()));
}

void DenseMatchDialog::onSelectAll()
{
    m_imageList->blockSignals(true);
    for (int i = 0; i < m_imageList->count(); ++i)
        m_imageList->item(i)->setCheckState(Qt::Checked);
    m_imageList->blockSignals(false);
    refreshMatchPairs();
    emitSettingsNow();
}

void DenseMatchDialog::onDeselectAll()
{
    m_imageList->blockSignals(true);
    for (int i = 0; i < m_imageList->count(); ++i)
        m_imageList->item(i)->setCheckState(Qt::Unchecked);
    m_imageList->blockSignals(false);
    refreshMatchPairs();
    emitSettingsNow();
}

void DenseMatchDialog::onImageSelectionChanged()
{
    refreshMatchPairs();
    emitSettingsNow();
}

void DenseMatchDialog::onBrowseOutput()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("选择输出目录"));
    if (!dir.isEmpty())
        m_outputEdit->setText(dir);
}

void DenseMatchDialog::onAlgorithmChanged(int /*index*/)
{
    int algo = m_algorithmCombo->currentData().toInt();
    bool isSGM = (algo == 1 || algo == 2);
    bool isOpenCV = (algo == 3);
    m_p1Spin->setEnabled(isSGM);
    m_p2Spin->setEnabled(isSGM);
    m_directionsSpin->setEnabled(isSGM);
    m_pyramidSpin->setEnabled(isSGM);
    m_useCudaChk->setEnabled(!isOpenCV);
    m_deviceSpin->setEnabled(!isOpenCV);
    emitSettingsNow();
}

void DenseMatchDialog::onRun()
{
    const QJsonObject settings = collectSettings();
    const QJsonArray pairs = settings.value(QStringLiteral("match_pairs")).toArray();

    if (pairs.isEmpty())
    {
        m_matchCountLabel->setText(tr("没有可处理的匹配对"));
        return;
    }

    m_runBtn->setEnabled(false);
    m_cancelBtn->setEnabled(false);
    emit runRequested(settings);
}

void DenseMatchDialog::onProcessingFinished()
{
    m_runBtn->setEnabled(true);
    m_cancelBtn->setEnabled(true);
}

void DenseMatchDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

QJsonObject DenseMatchDialog::collectSettings() const
{
    QJsonObject s;

    QJsonArray images;
    for (int i = 0; i < m_imageList->count(); ++i)
    {
        auto *item = m_imageList->item(i);
        if (item->checkState() == Qt::Checked)
            images.append(item->data(Qt::UserRole).toString());
    }
    s["selected_images"] = images;

    QJsonArray pairs;
    for (int r = 0; r < m_matchTable->rowCount(); ++r)
    {
        QJsonObject pair;
        pair["imgA"]       = m_matchPairs[r].imgA;
        pair["imgB"]       = m_matchPairs[r].imgB;
        pair["match_file"] = m_matchPairs[r].matchFile;
        pairs.append(pair);
    }
    s["match_pairs"] = pairs;

    s["output_dir"]     = m_outputEdit->text();
    s["algorithm"]      = m_algorithmCombo->currentData().toInt();
    s["cost_func"]      = m_costFuncCombo->currentData().toInt();
    s["subpixel_mode"]  = m_subpixelCombo->currentData().toInt();
    s["min_disparity"]  = m_minDispSpin->value();
    s["max_disparity"]  = m_maxDispSpin->value();
    s["kernel_w"]       = m_kernelWSpin->value();
    s["kernel_h"]       = m_kernelHSpin->value();
    s["p1"]             = m_p1Spin->value();
    s["p2"]             = m_p2Spin->value();
    s["directions"]     = m_directionsSpin->value();
    s["pyramid"]        = m_pyramidSpin->value();
    s["use_cuda"]       = m_useCudaChk->isChecked();
    s["cuda_device"]    = m_deviceSpin->value();
    s["threads"]        = m_threadsSpin->value();
    s["opencv_compare"] = m_opencvCompareChk->isChecked();
    s["lr_threshold"]   = m_lrThresholdSpin->value();
    s["median_filter"]  = m_medianFilterSpin->value();

    return s;
}

void DenseMatchDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty()) return;

    if (settings.contains("output_dir"))
        m_outputEdit->setText(settings.value("output_dir").toString());

    int algo = settings.value("algorithm").toInt(2);
    for (int i = 0; i < m_algorithmCombo->count(); ++i)
        if (m_algorithmCombo->itemData(i).toInt() == algo)
        { m_algorithmCombo->setCurrentIndex(i); break; }

    int cost = settings.value("cost_func").toInt(3);
    for (int i = 0; i < m_costFuncCombo->count(); ++i)
        if (m_costFuncCombo->itemData(i).toInt() == cost)
        { m_costFuncCombo->setCurrentIndex(i); break; }

    int sub = settings.value("subpixel_mode").toInt(1);
    for (int i = 0; i < m_subpixelCombo->count(); ++i)
        if (m_subpixelCombo->itemData(i).toInt() == sub)
        { m_subpixelCombo->setCurrentIndex(i); break; }

    m_minDispSpin->setValue(settings.value("min_disparity").toInt(0));
    m_maxDispSpin->setValue(settings.value("max_disparity").toInt(256));
    m_kernelWSpin->setValue(settings.value("kernel_w").toInt(15));
    m_kernelHSpin->setValue(settings.value("kernel_h").toInt(15));
    m_p1Spin->setValue(settings.value("p1").toInt(8));
    m_p2Spin->setValue(settings.value("p2").toInt(32));
    m_directionsSpin->setValue(settings.value("directions").toInt(8));
    m_pyramidSpin->setValue(settings.value("pyramid").toInt(2));
    m_useCudaChk->setChecked(settings.value("use_cuda").toBool(true));
    m_deviceSpin->setValue(settings.value("cuda_device").toInt(0));
    m_threadsSpin->setValue(settings.value("threads").toInt(4));
    m_opencvCompareChk->setChecked(settings.value("opencv_compare").toBool(false));
    m_lrThresholdSpin->setValue(settings.value("lr_threshold").toDouble(1.0));
    m_medianFilterSpin->setValue(settings.value("median_filter").toInt(3));
}

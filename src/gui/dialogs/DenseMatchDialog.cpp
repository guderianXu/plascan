// =============================================================================
// 文件: DenseMatchDialog.cpp
// 功能: 密集匹配对话框实现 — 业务逻辑（UI 构建在 DenseMatchDialogUi.cpp）
// =============================================================================
#include "DenseMatchDialog.h"
#include "ProjectManager.h"
#include "project/ProjectIO.h"
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
    , _projectManager(projectManager)
{
    setWindowTitle(tr("密集匹配"));
    resize(900, 560);
    setupUi();
    loadProjectImages();
}

void DenseMatchDialog::loadProjectImages()
{
    if (!_projectManager) return;

    _allImages = _projectManager->getAllImages();
    if (_allImages.isEmpty()) return;

    _imageList->blockSignals(true);
    for (const QString &imgPath : _allImages)
    {
        QFileInfo fi(imgPath);
        auto *item = new QListWidgetItem(fi.fileName());
        item->setData(Qt::UserRole, imgPath);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
        _imageList->addItem(item);
    }
    _imageList->blockSignals(false);

    if (_projectManager && !_projectManager->currentProjectPath().isEmpty())
    {
        const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(
            _projectManager->currentProjectPath());
        _outputEdit->setText(
            QDir(assetsDir).filePath(QStringLiteral("dense_match")));
    }

    refreshMatchPairs();
}

void DenseMatchDialog::refreshMatchPairs()
{
    if (!_projectManager) return;

    _matchPairs.clear();
    _matchTable->setRowCount(0);

    QStringList selected;
    for (int i = 0; i < _imageList->count(); ++i)
    {
        auto *item = _imageList->item(i);
        if (item->checkState() == Qt::Checked)
            selected.append(item->data(Qt::UserRole).toString());
    }

    if (selected.size() < 2)
    {
        _matchCountLabel->setText(
            tr("请至少选择 2 张影像（当前选中 %1 张）").arg(selected.size()));
        return;
    }

    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(
        _projectManager->currentProjectPath());
    const QString matchDir = QDir(assetsDir).filePath(QStringLiteral("matches"));

    const auto *projData = _projectManager->projectData();
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
                _matchPairs.append(info);
            }
        }
    }

    for (const auto &info : _matchPairs)
    {
        int row = _matchTable->rowCount();
        _matchTable->insertRow(row);
        _matchTable->setItem(row, 0,
            new QTableWidgetItem(QFileInfo(info.imgA).fileName()));
        _matchTable->setItem(row, 1,
            new QTableWidgetItem(QFileInfo(info.imgB).fileName()));
        _matchTable->setItem(row, 2,
            new QTableWidgetItem(QString::number(info.numMatches)));
    }

    _matchCountLabel->setText(
        tr("共 %1 个匹配对（从 %2 张选中影像中）")
            .arg(_matchPairs.size())
            .arg(selected.size()));
}

void DenseMatchDialog::onSelectAll()
{
    _imageList->blockSignals(true);
    for (int i = 0; i < _imageList->count(); ++i)
        _imageList->item(i)->setCheckState(Qt::Checked);
    _imageList->blockSignals(false);
    refreshMatchPairs();
    emitSettingsNow();
}

void DenseMatchDialog::onDeselectAll()
{
    _imageList->blockSignals(true);
    for (int i = 0; i < _imageList->count(); ++i)
        _imageList->item(i)->setCheckState(Qt::Unchecked);
    _imageList->blockSignals(false);
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
        _outputEdit->setText(dir);
}

void DenseMatchDialog::onAlgorithmChanged(int /*index*/)
{
    int algo = _algorithmCombo->currentData().toInt();
    bool isSGM = (algo == 1 || algo == 2);
    bool isOpenCV = (algo == 3);
    _p1Spin->setEnabled(isSGM);
    _p2Spin->setEnabled(isSGM);
    _directionsSpin->setEnabled(isSGM);
    _pyramidSpin->setEnabled(isSGM);
    _useCudaChk->setEnabled(!isOpenCV);
    _deviceSpin->setEnabled(!isOpenCV);
    emitSettingsNow();
}

void DenseMatchDialog::onRun()
{
    const QJsonObject settings = collectSettings();
    const QJsonArray pairs = settings.value(QStringLiteral("match_pairs")).toArray();

    if (pairs.isEmpty())
    {
        _matchCountLabel->setText(tr("没有可处理的匹配对"));
        return;
    }

    _runBtn->setEnabled(false);
    _cancelBtn->setEnabled(false);
    emit runRequested(settings);
}

void DenseMatchDialog::onProcessingFinished()
{
    _runBtn->setEnabled(true);
    _cancelBtn->setEnabled(true);
}

void DenseMatchDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

QJsonObject DenseMatchDialog::collectSettings() const
{
    QJsonObject s;

    QJsonArray images;
    for (int i = 0; i < _imageList->count(); ++i)
    {
        auto *item = _imageList->item(i);
        if (item->checkState() == Qt::Checked)
            images.append(item->data(Qt::UserRole).toString());
    }
    s["selected_images"] = images;

    QJsonArray pairs;
    for (int r = 0; r < _matchTable->rowCount(); ++r)
    {
        QJsonObject pair;
        pair["imgA"]       = _matchPairs[r].imgA;
        pair["imgB"]       = _matchPairs[r].imgB;
        pair["match_file"] = _matchPairs[r].matchFile;
        pairs.append(pair);
    }
    s["match_pairs"] = pairs;

    s["output_dir"]     = _outputEdit->text();
    s["algorithm"]      = _algorithmCombo->currentData().toInt();
    s["cost_func"]      = _costFuncCombo->currentData().toInt();
    s["subpixel_mode"]  = _subpixelCombo->currentData().toInt();
    s["min_disparity"]  = _minDispSpin->value();
    s["max_disparity"]  = _maxDispSpin->value();
    s["kernel_w"]       = _kernelWSpin->value();
    s["kernel_h"]       = _kernelHSpin->value();
    s["p1"]             = _p1Spin->value();
    s["p2"]             = _p2Spin->value();
    s["directions"]     = _directionsSpin->value();
    s["pyramid"]        = _pyramidSpin->value();
    s["use_cuda"]       = _useCudaChk->isChecked();
    s["cuda_device"]    = _deviceSpin->value();
    s["threads"]        = _threadsSpin->value();
    s["opencv_compare"] = _opencvCompareChk->isChecked();
    s["lr_threshold"]   = _lrThresholdSpin->value();
    s["median_filter"]  = _medianFilterSpin->value();

    return s;
}

void DenseMatchDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty()) return;

    if (settings.contains("output_dir"))
        _outputEdit->setText(settings.value("output_dir").toString());

    int algo = settings.value("algorithm").toInt(2);
    for (int i = 0; i < _algorithmCombo->count(); ++i)
        if (_algorithmCombo->itemData(i).toInt() == algo)
        { _algorithmCombo->setCurrentIndex(i); break; }

    int cost = settings.value("cost_func").toInt(3);
    for (int i = 0; i < _costFuncCombo->count(); ++i)
        if (_costFuncCombo->itemData(i).toInt() == cost)
        { _costFuncCombo->setCurrentIndex(i); break; }

    int sub = settings.value("subpixel_mode").toInt(1);
    for (int i = 0; i < _subpixelCombo->count(); ++i)
        if (_subpixelCombo->itemData(i).toInt() == sub)
        { _subpixelCombo->setCurrentIndex(i); break; }

    _minDispSpin->setValue(settings.value("min_disparity").toInt(0));
    _maxDispSpin->setValue(settings.value("max_disparity").toInt(256));
    _kernelWSpin->setValue(settings.value("kernel_w").toInt(15));
    _kernelHSpin->setValue(settings.value("kernel_h").toInt(15));
    _p1Spin->setValue(settings.value("p1").toInt(8));
    _p2Spin->setValue(settings.value("p2").toInt(32));
    _directionsSpin->setValue(settings.value("directions").toInt(8));
    _pyramidSpin->setValue(settings.value("pyramid").toInt(2));
    _useCudaChk->setChecked(settings.value("use_cuda").toBool(true));
    _deviceSpin->setValue(settings.value("cuda_device").toInt(0));
    _threadsSpin->setValue(settings.value("threads").toInt(4));
    _opencvCompareChk->setChecked(settings.value("opencv_compare").toBool(false));
    _lrThresholdSpin->setValue(settings.value("lr_threshold").toDouble(1.0));
    _medianFilterSpin->setValue(settings.value("median_filter").toInt(3));
}

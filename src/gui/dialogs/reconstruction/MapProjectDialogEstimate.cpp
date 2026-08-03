#include "reconstruction/MapProjectDialog.h"

#include "TerrainPipeline.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QTimer>
#include <QThreadPool>
#include <QPointer>

#include <algorithm>
#include <cmath>

namespace
{

double positiveJsonValue(const QJsonObject &object,
                         const QString &primaryKey,
                         const QString &fallbackKey = QString())
{
    double value = object.value(primaryKey).toDouble(0.0);
    if (!(value > 0.0) && !fallbackKey.isEmpty())
    {
        value = object.value(fallbackKey).toDouble(0.0);
    }
    return value;
}

QString formatMemory(qint64 bytes)
{
    if (bytes <= 0)
    {
        return QStringLiteral("未知");
    }
    const double mebibytes = static_cast<double>(bytes) / (1024.0 * 1024.0);
    if (mebibytes < 1024.0)
    {
        return QStringLiteral("%1 MiB").arg(mebibytes, 0, 'f', mebibytes < 10.0 ? 1 : 0);
    }
    return QStringLiteral("%1 GiB").arg(mebibytes / 1024.0, 0, 'f', 2);
}

QByteArray pointCloudEstimateSignature(QJsonObject settings)
{
    settings.remove(QStringLiteral("images"));
    settings.remove(QStringLiteral("output_path"));
    settings.remove(QStringLiteral("blend_mode"));
    settings.remove(QStringLiteral("color_correction"));
    settings.remove(QStringLiteral("sharpness_weighting"));
    settings.remove(QStringLiteral("ghost_filter"));
    settings.remove(QStringLiteral("use_project_masks"));
    if (settings.value(QStringLiteral("pixel_size_auto")).toBool())
    {
        settings.remove(QStringLiteral("pixel_size_x"));
        settings.remove(QStringLiteral("pixel_size_y"));
        settings.remove(QStringLiteral("resolution"));
    }
    if (settings.value(QStringLiteral("bounds_auto")).toBool())
    {
        settings.remove(QStringLiteral("min_x"));
        settings.remove(QStringLiteral("min_y"));
        settings.remove(QStringLiteral("max_x"));
        settings.remove(QStringLiteral("max_y"));
    }
    if (settings.value(QStringLiteral("body_reference_auto")).toBool())
    {
        settings.remove(QStringLiteral("body_center_x"));
        settings.remove(QStringLiteral("body_center_y"));
        settings.remove(QStringLiteral("body_center_z"));
        settings.remove(QStringLiteral("reference_radius"));
    }
    return QJsonDocument(settings).toJson(QJsonDocument::Compact);
}

} // namespace

void MapProjectDialog::restoreDemPixelSize()
{
    if (!(_demPixelSizeX > 0.0) || !(_demPixelSizeY > 0.0))
    {
        _pixelSizeUserEdited = false;
        runEstimate(true);
        return;
    }
    _updatingEstimate = true;
    _pixelSizeXSpin->setValue(_demPixelSizeX);
    _pixelSizeYSpin->setValue(_demPixelSizeY);
    _updatingEstimate = false;
    _pixelSizeUserEdited = false;
    onSettingsModified();
}

void MapProjectDialog::resetBoundsToDem()
{
    if (!_hasDemEstimate)
    {
        _boundsUserEdited = false;
        runEstimate(true);
        return;
    }
    _updatingEstimate = true;
    _minXSpin->setValue(_demMinX);
    _maxXSpin->setValue(_demMaxX);
    _minYSpin->setValue(_demMinY);
    _maxYSpin->setValue(_demMaxY);
    _updatingEstimate = false;
    _boundsUserEdited = false;
    onSettingsModified();
}

void MapProjectDialog::estimateNow()
{
    runEstimate(true);
}

void MapProjectDialog::invalidateDemEstimate()
{
    _lastEstimate = QJsonObject();
    _hasDemEstimate = false;
    _demPixelSizeX = 0.0;
    _demPixelSizeY = 0.0;
    _demMinX = 0.0;
    _demMaxX = 0.0;
    _demMinY = 0.0;
    _demMaxY = 0.0;
    const bool pointCloud = _surfaceCombo
        && _surfaceCombo->currentData().toString() == QStringLiteral("point_cloud");
    _coordinateSystemLabel->setText(pointCloud
        ? tr("等待读取点云投影范围") : tr("等待读取 DEM 坐标系"));
    _coordinateSystemLabel->setToolTip(QString());
    _totalSizeLabel->setText(tr("总尺寸（像素）：等待有效表面范围"));
    _memoryEstimateLabel->setText(tr("预计处理内存：未知"));
}

void MapProjectDialog::scheduleEstimate()
{
    if (!_estimateTimer || _running || _applyingSettings)
    {
        return;
    }
    const QString demPath = _demEdit->text().trimmed();
    if (demPath.isEmpty() || !QFileInfo::exists(demPath))
    {
        return;
    }
    _estimateTimer->start();
}

bool MapProjectDialog::runEstimate(bool reportError)
{
    if (_running)
    {
        return false;
    }

    const QString demPath = _demEdit->text().trimmed();
    if (demPath.isEmpty() || !QFileInfo::exists(demPath))
    {
        if (reportError)
        {
            const bool pointCloud =
                _surfaceCombo->currentData().toString() == QStringLiteral("point_cloud");
            QMessageBox::warning(this, tr("无法预计"), pointCloud
                ? tr("请先选择有效的彩色点云文件。")
                : tr("请先选择有效的 DEM 文件。"));
        }
        return false;
    }

    QJsonObject estimateSettings = currentSettings();
    if (!_pixelSizeUserEdited)
    {
        estimateSettings[QStringLiteral("pixel_size_x")] = 0.0;
        estimateSettings[QStringLiteral("pixel_size_y")] = 0.0;
        estimateSettings[QStringLiteral("resolution")] = 0.0;
    }
    if (!_boundsUserEdited)
    {
        estimateSettings[QStringLiteral("bounds_enabled")] = false;
    }

    const bool pointCloud =
        estimateSettings.value(QStringLiteral("surface_type")).toString()
        == QStringLiteral("point_cloud");
    if (pointCloud)
    {
        const QByteArray signature = pointCloudEstimateSignature(estimateSettings);
        if (!_lastEstimate.isEmpty() && signature == _lastPointCloudEstimateSignature)
        {
            return true;
        }
        _reportPointCloudEstimateError = _reportPointCloudEstimateError || reportError;
        if (_pointCloudEstimateRunning)
        {
            return false;
        }

        _pointCloudEstimateRunning = true;
        _estimateButton->setEnabled(false);
        _coordinateSystemLabel->setText(tr("正在后台解析彩色点云..."));
        _totalSizeLabel->setText(tr("总尺寸（像素）：正在后台预计"));
        QPointer<MapProjectDialog> self(this);
        QThreadPool::globalInstance()->start(
            [self, demPath, estimateSettings, signature]()
            {
                QJsonObject outcome;
                QJsonObject estimate;
                QString error;
                outcome[QStringLiteral("ok")] =
                    xjw::TerrainPipeline::estimateOrthoProduct(
                        demPath, estimateSettings, &estimate, &error);
                outcome[QStringLiteral("estimate")] = estimate;
                outcome[QStringLiteral("error")] = error;
                if (!self)
                {
                    return;
                }
                QMetaObject::invokeMethod(
                    self,
                    [self, outcome, signature]()
                    {
                        if (!self)
                        {
                            return;
                        }
                        self->_pointCloudEstimateRunning = false;
                        const QByteArray currentSignature =
                            pointCloudEstimateSignature(self->currentSettings());
                        if (signature != currentSignature)
                        {
                            self->runEstimate(false);
                            return;
                        }
                        const bool ok = outcome.value(QStringLiteral("ok")).toBool();
                        const QString error = outcome.value(QStringLiteral("error")).toString();
                        if (ok)
                        {
                            self->_lastEstimate =
                                outcome.value(QStringLiteral("estimate")).toObject();
                            self->_lastPointCloudEstimateSignature = signature;
                            self->updateEstimateSummary(self->_lastEstimate);
                        }
                        else
                        {
                            self->invalidateDemEstimate();
                            self->_totalSizeLabel->setText(tr("总尺寸（像素）：无法预计"));
                            self->_estimateButton->setToolTip(error);
                        }
                        self->updateControlAvailability();
                        const bool report = self->_reportPointCloudEstimateError;
                        self->_reportPointCloudEstimateError = false;
                        if (!ok && report)
                        {
                            QMessageBox::warning(self, tr("无法预计"), error);
                        }
                        if (!ok)
                        {
                            self->_runAfterPointCloudEstimate = false;
                        }
                        if (ok && self->_runAfterPointCloudEstimate)
                        {
                            self->_runAfterPointCloudEstimate = false;
                            QTimer::singleShot(0, self, &MapProjectDialog::onRun);
                        }
                    },
                    Qt::QueuedConnection);
            });
        return false;
    }

    QJsonObject estimate;
    QString errorMessage;
    if (!xjw::TerrainPipeline::estimateOrthoProduct(
            demPath, estimateSettings, &estimate, &errorMessage))
    {
        invalidateDemEstimate();
        _totalSizeLabel->setText(tr("总尺寸（像素）：无法预计"));
        _estimateButton->setToolTip(errorMessage);
        if (reportError)
        {
            QMessageBox::warning(
                this,
                tr("无法预计"),
                errorMessage.isEmpty() ? tr("无法读取输入表面的范围和像元信息。") : errorMessage);
        }
        return false;
    }

    _estimateButton->setToolTip(QString());
    _lastEstimate = estimate;
    updateEstimateSummary(estimate);
    return true;
}

void MapProjectDialog::updateEstimateSummary(const QJsonObject &estimate)
{
    _updatingEstimate = true;

    QString coordinateSystem =
        estimate.value(QStringLiteral("coordinate_system")).toString().trimmed();
    if (coordinateSystem.isEmpty())
    {
        coordinateSystem = tr("本地坐标（跟随 DEM 网格）");
    }
    _coordinateSystemLabel->setText(coordinateSystem);
    _coordinateSystemLabel->setToolTip(coordinateSystem);

    const QJsonObject resolvedSettings =
        estimate.value(QStringLiteral("resolved_settings")).toObject();
    if (!resolvedSettings.isEmpty()
        && _surfaceCombo->currentData().toString() == QStringLiteral("point_cloud"))
    {
        _bodyCenterXSpin->setValue(
            resolvedSettings.value(QStringLiteral("body_center_x")).toDouble());
        _bodyCenterYSpin->setValue(
            resolvedSettings.value(QStringLiteral("body_center_y")).toDouble());
        _bodyCenterZSpin->setValue(
            resolvedSettings.value(QStringLiteral("body_center_z")).toDouble());
        _referenceRadiusSpin->setValue(
            resolvedSettings.value(QStringLiteral("reference_radius")).toDouble());
    }

    _demPixelSizeX = positiveJsonValue(
        estimate, QStringLiteral("dem_pixel_size_x"), QStringLiteral("pixel_size_x"));
    _demPixelSizeY = positiveJsonValue(
        estimate, QStringLiteral("dem_pixel_size_y"), QStringLiteral("pixel_size_y"));
    if (!(_demPixelSizeY > 0.0))
    {
        _demPixelSizeY = _demPixelSizeX;
    }
    _demMinX = estimate.value(QStringLiteral("dem_min_x"))
                   .toDouble(estimate.value(QStringLiteral("min_x")).toDouble());
    _demMaxX = estimate.value(QStringLiteral("dem_max_x"))
                   .toDouble(estimate.value(QStringLiteral("max_x")).toDouble());
    _demMinY = estimate.value(QStringLiteral("dem_min_y"))
                   .toDouble(estimate.value(QStringLiteral("min_y")).toDouble());
    _demMaxY = estimate.value(QStringLiteral("dem_max_y"))
                   .toDouble(estimate.value(QStringLiteral("max_y")).toDouble());
    _hasDemEstimate = _demPixelSizeX > 0.0
        && _demPixelSizeY > 0.0
        && _demMaxX > _demMinX
        && _demMaxY > _demMinY;

    if (_hasDemEstimate && !_pixelSizeUserEdited)
    {
        _pixelSizeXSpin->setValue(_demPixelSizeX);
        _pixelSizeYSpin->setValue(_demPixelSizeY);
    }
    if (_hasDemEstimate && !_boundsUserEdited)
    {
        _minXSpin->setValue(_demMinX);
        _maxXSpin->setValue(_demMaxX);
        _minYSpin->setValue(_demMinY);
        _maxYSpin->setValue(_demMaxY);
    }

    _updatingEstimate = false;
    updateControlAvailability();

    const int width = estimate.value(QStringLiteral("width")).toInt();
    const int height = estimate.value(QStringLiteral("height")).toInt();
    const qint64 memoryBytes = static_cast<qint64>(
        estimate.value(QStringLiteral("estimated_memory_bytes")).toDouble(0.0));
    if (width > 0 && height > 0)
    {
        _totalSizeLabel->setText(tr("总尺寸（像素）：%1 × %2").arg(width).arg(height));
        const qint64 fallbackMemory = static_cast<qint64>(width) * height * 16;
        _memoryEstimateLabel->setText(
            tr("预计处理内存：%1").arg(formatMemory(memoryBytes > 0 ? memoryBytes : fallbackMemory)));
    }
    else
    {
        updateLocalEstimateSummary();
    }
}

void MapProjectDialog::updateLocalEstimateSummary()
{
    double minX = _demMinX;
    double maxX = _demMaxX;
    double minY = _demMinY;
    double maxY = _demMaxY;
    if (_boundsEnabledCheck->isChecked())
    {
        minX = _minXSpin->value();
        maxX = _maxXSpin->value();
        minY = _minYSpin->value();
        maxY = _maxYSpin->value();
    }
    const double spanX = maxX - minX;
    const double spanY = maxY - minY;
    if (!(spanX > 0.0) || !(spanY > 0.0))
    {
        _totalSizeLabel->setText(tr("总尺寸（像素）：等待有效表面范围"));
        _memoryEstimateLabel->setText(tr("预计处理内存：未知"));
        return;
    }

    int width = 0;
    int height = 0;
    if (_maximumDimensionRadio->isChecked())
    {
        const int maximumDimension = _maximumDimensionSpin->value();
        if (_demPixelSizeX > 0.0 && _demPixelSizeY > 0.0)
        {
            const double scaleX =
                spanX / (static_cast<double>(maximumDimension) * _demPixelSizeX);
            const double scaleY =
                spanY / (static_cast<double>(maximumDimension) * _demPixelSizeY);
            const double scale = std::max({1.0, scaleX, scaleY});
            width = std::max(
                1,
                static_cast<int>(std::ceil(spanX / (_demPixelSizeX * scale))));
            height = std::max(
                1,
                static_cast<int>(std::ceil(spanY / (_demPixelSizeY * scale))));
        }
        else if (spanX >= spanY)
        {
            width = maximumDimension;
            height = std::max(
                1,
                static_cast<int>(std::ceil(maximumDimension * spanY / spanX)));
        }
        else
        {
            height = maximumDimension;
            width = std::max(
                1,
                static_cast<int>(std::ceil(maximumDimension * spanX / spanY)));
        }
    }
    else
    {
        const double pixelSizeX = _pixelSizeXSpin->value();
        const double pixelSizeY = _pixelSizeYSpin->value();
        if (!(pixelSizeX > 0.0) || !(pixelSizeY > 0.0))
        {
            _totalSizeLabel->setText(tr("总尺寸（像素）：等待有效像元大小"));
            _memoryEstimateLabel->setText(tr("预计处理内存：未知"));
            return;
        }
        width = std::max(1, static_cast<int>(std::ceil(spanX / pixelSizeX)));
        height = std::max(1, static_cast<int>(std::ceil(spanY / pixelSizeY)));
    }

    _totalSizeLabel->setText(tr("总尺寸（像素）：%1 × %2").arg(width).arg(height));
    const qint64 memoryBytes = static_cast<qint64>(width) * height * 16;
    _memoryEstimateLabel->setText(tr("预计处理内存：%1").arg(formatMemory(memoryBytes)));
}

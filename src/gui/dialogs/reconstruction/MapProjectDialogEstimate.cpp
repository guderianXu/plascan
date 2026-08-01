#include "reconstruction/MapProjectDialog.h"

#include "TerrainPipeline.h"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QTimer>

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
    _coordinateSystemLabel->setText(tr("等待读取 DEM 坐标系"));
    _coordinateSystemLabel->setToolTip(QString());
    _totalSizeLabel->setText(tr("总尺寸（像素）：等待有效 DEM 范围"));
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
            QMessageBox::warning(this, tr("无法预计"), tr("请先选择有效的 DEM 文件。"));
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
                errorMessage.isEmpty() ? tr("无法读取 DEM 范围和像元信息。") : errorMessage);
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
        _totalSizeLabel->setText(tr("总尺寸（像素）：等待有效 DEM 范围"));
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

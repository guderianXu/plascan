#include "reconstruction/MapProjectDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QToolButton>
#include <QWidget>

namespace
{

QString comboValue(const QComboBox *comboBox, const QString &fallback)
{
    const QString value = comboBox ? comboBox->currentData().toString() : QString();
    return value.isEmpty() ? fallback : value;
}

void selectComboValue(QComboBox *comboBox, const QString &value)
{
    const int index = comboBox && !value.isEmpty() ? comboBox->findData(value) : -1;
    if (index >= 0)
    {
        comboBox->setCurrentIndex(index);
    }
}

bool samePath(const QString &left, const QString &right)
{
#ifdef Q_OS_WIN
    return QString::compare(
               QDir::cleanPath(left), QDir::cleanPath(right), Qt::CaseInsensitive) == 0;
#else
    return QDir::cleanPath(left) == QDir::cleanPath(right);
#endif
}

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

} // namespace

void MapProjectDialog::applySettings(const QJsonObject &settings)
{
    _applyingSettings = true;

    const QString surfaceType =
        settings.value(QStringLiteral("surface_type")).toString(QStringLiteral("dem"));
    selectComboValue(_surfaceCombo, surfaceType);
    const QString projectionType = settings.value(QStringLiteral("projection_type"))
        .toString(surfaceType == QStringLiteral("point_cloud")
                      ? QStringLiteral("planar")
                      : QStringLiteral("dem_grid"));
    _demGridProjectionRadio->setChecked(projectionType == QStringLiteral("dem_grid"));
    _planarProjectionRadio->setChecked(projectionType == QStringLiteral("planar"));
    _cylindricalProjectionRadio->setChecked(
        projectionType == QStringLiteral("cylindrical")
        || projectionType == QStringLiteral("simple_cylindrical"));

    const QString savedSurfacePath = surfaceType == QStringLiteral("point_cloud")
        ? settings.value(QStringLiteral("point_cloud_path")).toString().trimmed()
        : settings.value(QStringLiteral("dem_path")).toString().trimmed();
    if (!savedSurfacePath.isEmpty() && QFileInfo::exists(savedSurfacePath))
    {
        _demEdit->setText(savedSurfacePath);
    }
    else if (surfaceType == QStringLiteral("point_cloud")
             && !_defaultPointCloudPath.isEmpty())
    {
        _demEdit->setText(_defaultPointCloudPath);
    }
    else if (surfaceType == QStringLiteral("dem") && !_defaultDemPath.isEmpty())
    {
        _demEdit->setText(_defaultDemPath);
    }
    if (settings.contains(QStringLiteral("output_path")))
    {
        _outputEdit->setText(settings.value(QStringLiteral("output_path")).toString().trimmed());
    }

    selectComboValue(_blendCombo,
                     settings.value(QStringLiteral("blend_mode")).toString(QStringLiteral("mosaic")));
    _fillHolesCheck->setChecked(settings.value(QStringLiteral("fill_holes"))
                                    .toBool(surfaceType != QStringLiteral("point_cloud")));
    _ghostFilterCheck->setChecked(settings.value(QStringLiteral("ghost_filter")).toBool(false));
    _colorCorrectionCheck->setChecked(
        settings.value(QStringLiteral("color_correction")).toBool(true));
    _sharpnessWeightingCheck->setChecked(
        settings.value(QStringLiteral("sharpness_weighting")).toBool(false));
    _requestedUseProjectMasks =
        settings.value(QStringLiteral("use_project_masks")).toBool(false);
    _useProjectMasksCheck->setChecked(_requestedUseProjectMasks && _maskCount > 0);
    _bodyReferenceAutoCheck->setChecked(
        settings.value(QStringLiteral("body_reference_auto")).toBool(true));
    _bodyCenterXSpin->setValue(settings.value(QStringLiteral("body_center_x")).toDouble());
    _bodyCenterYSpin->setValue(settings.value(QStringLiteral("body_center_y")).toDouble());
    _bodyCenterZSpin->setValue(settings.value(QStringLiteral("body_center_z")).toDouble());
    _referenceRadiusSpin->setValue(
        settings.value(QStringLiteral("reference_radius")).toDouble());
    _centralMeridianSpin->setValue(
        settings.value(QStringLiteral("central_meridian")).toDouble());

    QString sizingMode = settings.value(QStringLiteral("sizing_mode")).toString().trimmed();
    if (sizingMode.isEmpty())
    {
        sizingMode = settings.value(QStringLiteral("resolution_mode")).toString().trimmed();
    }
    if (sizingMode.isEmpty())
    {
        sizingMode = QStringLiteral("pixel_size");
    }
    _maximumDimensionRadio->setChecked(sizingMode == QStringLiteral("maximum_dimension"));
    _pixelSizeRadio->setChecked(!_maximumDimensionRadio->isChecked());

    const double legacyResolution = settings.value(QStringLiteral("resolution")).toDouble(0.0);
    double pixelSizeX = positiveJsonValue(
        settings, QStringLiteral("pixel_size_x"), QStringLiteral("resolution"));
    double pixelSizeY = positiveJsonValue(
        settings, QStringLiteral("pixel_size_y"), QStringLiteral("resolution"));
    if (!(pixelSizeX > 0.0))
    {
        pixelSizeX = legacyResolution;
    }
    if (!(pixelSizeY > 0.0))
    {
        pixelSizeY = pixelSizeX;
    }
    if (pixelSizeX > 0.0)
    {
        _pixelSizeXSpin->setValue(pixelSizeX);
        _pixelSizeYSpin->setValue(pixelSizeY);
        _pixelSizeUserEdited =
            !settings.value(QStringLiteral("pixel_size_auto")).toBool(false);
    }
    _maximumDimensionSpin->setValue(
        qMax(1, settings.value(QStringLiteral("maximum_dimension")).toInt(4096)));

    _boundsEnabledCheck->setChecked(
        settings.value(QStringLiteral("bounds_enabled")).toBool(false));
    if (settings.contains(QStringLiteral("min_x"))
        && settings.contains(QStringLiteral("max_x"))
        && settings.contains(QStringLiteral("min_y"))
        && settings.contains(QStringLiteral("max_y")))
    {
        _minXSpin->setValue(settings.value(QStringLiteral("min_x")).toDouble());
        _maxXSpin->setValue(settings.value(QStringLiteral("max_x")).toDouble());
        _minYSpin->setValue(settings.value(QStringLiteral("min_y")).toDouble());
        _maxYSpin->setValue(settings.value(QStringLiteral("max_y")).toDouble());
        _boundsUserEdited =
            settings.value(QStringLiteral("bounds_enabled")).toBool(false)
            && !settings.value(QStringLiteral("bounds_auto")).toBool(false);
    }

    if (settings.contains(QStringLiteral("images")))
    {
        _pendingSelectedImages.clear();
        const QJsonArray savedImages = settings.value(QStringLiteral("images")).toArray();
        for (const QJsonValue &value : savedImages)
        {
            const QString path = value.toString().trimmed();
            if (!path.isEmpty())
            {
                _pendingSelectedImages.append(path);
            }
        }
        _hasPendingImageSelection = true;
        applyPendingImageSelection();
    }

    _applyingSettings = false;
    updateControlAvailability();
    updateImageSummary();
    scheduleEstimate();
}

void MapProjectDialog::onSettingsModified()
{
    if (_applyingSettings || _updatingEstimate)
    {
        return;
    }
    _requestedUseProjectMasks = _useProjectMasksCheck->isChecked();
    updateControlAvailability();
    updateLocalEstimateSummary();
    scheduleEstimate();
    emit settingsChanged(currentSettings());
}

void MapProjectDialog::onDemPathChanged()
{
    invalidateDemEstimate();
    if (!_applyingSettings)
    {
        _pixelSizeUserEdited = false;
        _boundsUserEdited = false;
    }
    onSettingsModified();
}

void MapProjectDialog::onSurfaceChanged()
{
    if (!_applyingSettings)
    {
        _runAfterPointCloudEstimate = false;
        const bool pointCloud =
            _surfaceCombo->currentData().toString() == QStringLiteral("point_cloud");
        _demEdit->setText(pointCloud ? _defaultPointCloudPath : _defaultDemPath);
        if (pointCloud)
        {
            _planarProjectionRadio->setChecked(true);
            _fillHolesCheck->setChecked(false);
            const QString currentOutput = QDir::cleanPath(_outputEdit->text().trimmed());
            if (currentOutput.endsWith(QStringLiteral("assets/ortho/relative_dom.tif")))
            {
                _outputEdit->setText(
                    QDir(_projectRoot).filePath(QStringLiteral("assets/ortho/point_cloud_dom.tif")));
            }
        }
        else
        {
            _demGridProjectionRadio->setChecked(true);
        }
        _pixelSizeUserEdited = false;
        _boundsUserEdited = false;
        invalidateDemEstimate();
    }
    onSettingsModified();
}

void MapProjectDialog::onPixelSizeEdited()
{
    if (!_applyingSettings && !_updatingEstimate)
    {
        _pixelSizeUserEdited = true;
    }
    onSettingsModified();
}

void MapProjectDialog::onBoundsEdited()
{
    if (!_applyingSettings && !_updatingEstimate)
    {
        _boundsUserEdited = true;
    }
    onSettingsModified();
}

void MapProjectDialog::onResolutionModeChanged()
{
    updateControlAvailability();
    onSettingsModified();
}

void MapProjectDialog::onBoundsEnabledChanged()
{
    updateControlAvailability();
    onSettingsModified();
}

void MapProjectDialog::onImageListToggled(bool expanded)
{
    _imagePanel->setVisible(expanded);
    _imageToggleButton->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
}

void MapProjectDialog::onImageSelectionChanged()
{
    if (_applyingSettings)
    {
        return;
    }
    updateImageSummary();
    onSettingsModified();
}

void MapProjectDialog::updateControlAvailability()
{
    const bool pointCloud =
        _surfaceCombo->currentData().toString() == QStringLiteral("point_cloud");
    const bool cylindrical = pointCloud && _cylindricalProjectionRadio->isChecked();
    _restorePixelSizeButton->setText(pointCloud ? tr("恢复估算值") : tr("恢复 DEM 值"));
    _resetBoundsButton->setText(pointCloud ? tr("重置为点云范围") : tr("重置为 DEM 范围"));
    const bool pixelMode = _pixelSizeRadio->isChecked();
    _pixelSizeXSpin->setEnabled(!_running && pixelMode);
    _pixelSizeYSpin->setEnabled(!_running && pixelMode);
    _restorePixelSizeButton->setEnabled(!_running && pixelMode && _hasDemEstimate);
    _maximumDimensionSpin->setEnabled(!_running && !pixelMode);

    const bool boundsEnabled = _boundsEnabledCheck->isChecked();
    for (QDoubleSpinBox *spinBox : {_minXSpin, _maxXSpin, _minYSpin, _maxYSpin})
    {
        spinBox->setEnabled(!_running && boundsEnabled);
    }
    _resetBoundsButton->setEnabled(!_running && boundsEnabled && _hasDemEstimate);
    _estimateButton->setEnabled(!_running);
    _useProjectMasksCheck->setEnabled(!_running && !pointCloud && _maskCount > 0);
    _demGridProjectionRadio->setEnabled(!_running && !pointCloud);
    _planarProjectionRadio->setEnabled(!_running && pointCloud);
    _cylindricalProjectionRadio->setEnabled(!_running && pointCloud);
    _blendCombo->setEnabled(!_running && !pointCloud);
    _colorSourceCombo->setCurrentIndex(pointCloud ? 1 : 0);
    _colorSourceCombo->setEnabled(false);
    _ghostFilterCheck->setEnabled(!_running && !pointCloud);
    _colorCorrectionCheck->setEnabled(!_running && !pointCloud);
    _sharpnessWeightingCheck->setEnabled(!_running && !pointCloud);
    _imageToggleButton->setEnabled(!_running && !pointCloud);
    if (pointCloud)
    {
        _imagePanel->setVisible(false);
        _imageToggleButton->setChecked(false);
    }
    _bodyReferenceWidget->setVisible(cylindrical);
    const bool manualBody = cylindrical && !_bodyReferenceAutoCheck->isChecked();
    for (QDoubleSpinBox *spinBox : {
             _bodyCenterXSpin, _bodyCenterYSpin, _bodyCenterZSpin, _referenceRadiusSpin})
    {
        spinBox->setEnabled(!_running && manualBody);
    }
    _centralMeridianSpin->setEnabled(!_running && cylindrical);
}

void MapProjectDialog::updateImageSummary()
{
    const int selectedCount = selectedImages().size();
    const int totalCount = _imageList ? _imageList->count() : 0;
    _imageToggleButton->setText(
        tr("输入影像（已选 %1 / %2）").arg(selectedCount).arg(totalCount));

    if (!_imageReadinessSet)
    {
        _imageReadinessLabel->setText(tr("相机参数与蒙版状态将在打开项目后检查"));
    }
}

void MapProjectDialog::applyPendingImageSelection()
{
    if (!_hasPendingImageSelection || !_imageList)
    {
        return;
    }

    for (int index = 0; index < _imageList->count(); ++index)
    {
        QListWidgetItem *item = _imageList->item(index);
        const bool ready = !_imageReadinessSet || item->data(Qt::UserRole).toBool();
        bool selected = false;
        for (const QString &path : _pendingSelectedImages)
        {
            if (samePath(path, item->text()))
            {
                selected = ready;
                break;
            }
        }
        item->setCheckState(selected ? Qt::Checked : Qt::Unchecked);
    }
    _hasPendingImageSelection = false;
}

QStringList MapProjectDialog::selectedImages() const
{
    QStringList images;
    if (!_imageList)
    {
        return images;
    }
    for (int index = 0; index < _imageList->count(); ++index)
    {
        const QListWidgetItem *item = _imageList->item(index);
        if (item && item->checkState() == Qt::Checked)
        {
            images.append(item->text());
        }
    }
    return images;
}

QJsonObject MapProjectDialog::currentSettings() const
{
    QJsonObject settings;
    const bool pointCloud =
        _surfaceCombo->currentData().toString() == QStringLiteral("point_cloud");
    const QString projectionType = !pointCloud
        ? QStringLiteral("dem_grid")
        : (_cylindricalProjectionRadio->isChecked()
               ? QStringLiteral("cylindrical")
               : QStringLiteral("planar"));
    settings.insert(QStringLiteral("projection_type"), projectionType);
    settings.insert(QStringLiteral("surface_type"), pointCloud
        ? QStringLiteral("point_cloud") : QStringLiteral("dem"));
    settings.insert(QStringLiteral("blend_mode"), comboValue(_blendCombo, QStringLiteral("mosaic")));
    settings.insert(QStringLiteral("color_source"), pointCloud
        ? QStringLiteral("point_colors") : QStringLiteral("images"));
    settings.insert(QStringLiteral("fill_holes"), _fillHolesCheck->isChecked());
    settings.insert(QStringLiteral("ghost_filter"), _ghostFilterCheck->isChecked());
    settings.insert(QStringLiteral("color_correction"), _colorCorrectionCheck->isChecked());
    settings.insert(QStringLiteral("sharpness_weighting"), _sharpnessWeightingCheck->isChecked());
    settings.insert(QStringLiteral("use_project_masks"), _useProjectMasksCheck->isChecked());

    const QString sizingMode = _maximumDimensionRadio->isChecked()
        ? QStringLiteral("maximum_dimension")
        : QStringLiteral("pixel_size");
    settings.insert(QStringLiteral("sizing_mode"), sizingMode);
    settings.insert(QStringLiteral("resolution_mode"), sizingMode);
    settings.insert(QStringLiteral("pixel_size_x"), _pixelSizeXSpin->value());
    settings.insert(QStringLiteral("pixel_size_y"), _pixelSizeYSpin->value());
    settings.insert(QStringLiteral("pixel_size_auto"), !_pixelSizeUserEdited);
    settings.insert(QStringLiteral("maximum_dimension"), _maximumDimensionSpin->value());
    settings.insert(QStringLiteral("resolution"), _pixelSizeXSpin->value());

    settings.insert(QStringLiteral("bounds_enabled"), _boundsEnabledCheck->isChecked());
    settings.insert(QStringLiteral("min_x"), _minXSpin->value());
    settings.insert(QStringLiteral("max_x"), _maxXSpin->value());
    settings.insert(QStringLiteral("min_y"), _minYSpin->value());
    settings.insert(QStringLiteral("max_y"), _maxYSpin->value());
    settings.insert(QStringLiteral("bounds_auto"), !_boundsUserEdited);
    settings.insert(QStringLiteral("dem_path"),
                    pointCloud ? QString() : _demEdit->text().trimmed());
    settings.insert(QStringLiteral("point_cloud_path"),
                    pointCloud ? _demEdit->text().trimmed() : QString());
    settings.insert(QStringLiteral("body_reference_auto"),
                    _bodyReferenceAutoCheck->isChecked());
    settings.insert(QStringLiteral("body_center_x"), _bodyCenterXSpin->value());
    settings.insert(QStringLiteral("body_center_y"), _bodyCenterYSpin->value());
    settings.insert(QStringLiteral("body_center_z"), _bodyCenterZSpin->value());
    settings.insert(QStringLiteral("reference_radius"), _referenceRadiusSpin->value());
    settings.insert(QStringLiteral("central_meridian"), _centralMeridianSpin->value());
    settings.insert(QStringLiteral("output_path"), _outputEdit->text().trimmed());
    settings.insert(QStringLiteral("images"), QJsonArray::fromStringList(selectedImages()));
    return settings;
}

bool MapProjectDialog::validateSettings(const QJsonObject &settings, QString *errorMessage) const
{
    const auto fail = [errorMessage](const QString &message)
    {
        if (errorMessage)
        {
            *errorMessage = message;
        }
        return false;
    };

    const bool pointCloud =
        settings.value(QStringLiteral("surface_type")).toString() == QStringLiteral("point_cloud");
    if (!pointCloud && settings.value(QStringLiteral("images")).toArray().isEmpty())
    {
        return fail(tr("请至少勾选一张输入影像。"));
    }
    if (!pointCloud && _imageReadinessSet && _cameraReadyCount <= 0)
    {
        return fail(tr("所选项目没有已就绪的相机参数，请先完成相机初始化或空中三角测量。"));
    }
    const QFileInfo surfaceInfo(settings.value(
        pointCloud ? QStringLiteral("point_cloud_path") : QStringLiteral("dem_path")).toString());
    if (!surfaceInfo.exists() || !surfaceInfo.isFile())
    {
        return fail(pointCloud
            ? tr("点云路径不是有效的 PLY、OBJ 或 XYZ 文件。")
            : tr("DEM 路径不是有效的 DEM 文件。"));
    }
    if (pointCloud && _cylindricalProjectionRadio->isChecked()
        && !_bodyReferenceAutoCheck->isChecked()
        && !(_referenceRadiusSpin->value() > 0.0))
    {
        return fail(tr("手动小天体参考半径必须大于 0。"));
    }

    const QString outputPath = settings.value(QStringLiteral("output_path")).toString().trimmed();
    if (outputPath.isEmpty())
    {
        return fail(tr("请指定正射影像输出路径。"));
    }
    if (QFileInfo(outputPath).exists() && QFileInfo(outputPath).isDir())
    {
        return fail(tr("正射影像输出路径不能是目录。"));
    }
    const QString suffix = QFileInfo(outputPath).suffix().toLower();
    if (suffix != QStringLiteral("tif")
        && suffix != QStringLiteral("tiff")
        && suffix != QStringLiteral("png"))
    {
        return fail(tr("输出文件必须使用 .tif、.tiff 或 .png 扩展名。"));
    }

    QString sizingMode = settings.value(QStringLiteral("sizing_mode")).toString();
    if (sizingMode.isEmpty())
    {
        sizingMode = settings.value(QStringLiteral("resolution_mode")).toString();
    }
    if (sizingMode == QStringLiteral("pixel_size")
        && (!(settings.value(QStringLiteral("pixel_size_x")).toDouble() > 0.0)
            || !(settings.value(QStringLiteral("pixel_size_y")).toDouble() > 0.0)))
    {
        return fail(tr("X/Y 像元大小必须大于 0。"));
    }
    if (sizingMode == QStringLiteral("maximum_dimension")
        && settings.value(QStringLiteral("maximum_dimension")).toInt() <= 0)
    {
        return fail(tr("最大尺寸必须大于 0。"));
    }
    if (settings.value(QStringLiteral("bounds_enabled")).toBool()
        && (!(settings.value(QStringLiteral("max_x")).toDouble()
              > settings.value(QStringLiteral("min_x")).toDouble())
            || !(settings.value(QStringLiteral("max_y")).toDouble()
                 > settings.value(QStringLiteral("min_y")).toDouble())))
    {
        return fail(tr("区域边界无效：最大 X/Y 必须分别大于最小 X/Y。"));
    }
    return true;
}

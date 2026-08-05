#include "CameraCalibrationDialog.h"

#include <QAbstractItemView>
#include <QColor>
#include <QDialogButtonBox>
#include <QHeaderView>
#include <QLabel>
#include <QListWidget>
#include <QMap>
#include <QSplitter>
#include <QStyle>
#include <QTableWidget>
#include <QTabWidget>
#include <QVBoxLayout>

#include <cmath>
#include <optional>

namespace
{

struct ParameterDescriptor
{
    QString key;
    QString label;
    bool percentageUseful;
};

const QVector<ParameterDescriptor> &parameters()
{
    static const QVector<ParameterDescriptor> descriptors{
        {QStringLiteral("f"), QStringLiteral("f (等效焦距)"), true},
        {QStringLiteral("fu"), QStringLiteral("fx (fu)"), true},
        {QStringLiteral("fv"), QStringLiteral("fy (fv)"), true},
        {QStringLiteral("cx"), QStringLiteral("cx (相对图像中心)"), false},
        {QStringLiteral("cy"), QStringLiteral("cy (相对图像中心)"), false},
        {QStringLiteral("k1"), QStringLiteral("k1"), false},
        {QStringLiteral("k2"), QStringLiteral("k2"), false},
        {QStringLiteral("k3"), QStringLiteral("k3"), false},
        {QStringLiteral("p1"), QStringLiteral("p1"), false},
        {QStringLiteral("p2"), QStringLiteral("p2"), false}};
    return descriptors;
}

std::optional<double> calibrationParameter(
    const xjw::gui::camera_calibration::CameraCalibrationRecord &record,
    const QJsonObject &camera,
    const QString &key)
{
    const bool millimeters = camera.value(QStringLiteral("intrinsics_unit"))
                                 .toString()
                                 .compare(QStringLiteral("mm"), Qt::CaseInsensitive) == 0;
    const bool directValue = key.startsWith(QLatin1Char('k')) ||
        key.startsWith(QLatin1Char('p')) || !millimeters;
    if (directValue && camera.contains(key) && camera.value(key).isDouble())
    {
        return camera.value(key).toDouble();
    }
    const double pitch = millimeters
        ? std::max(1e-12, camera.value(QStringLiteral("pitch")).toDouble(1.0))
        : 1.0;
    const auto pixelValue = [&camera, pitch](const QString &parameter) -> std::optional<double>
    {
        if (!camera.contains(parameter) || !camera.value(parameter).isDouble())
        {
            return std::nullopt;
        }
        return camera.value(parameter).toDouble() / pitch;
    };

    if (key == QStringLiteral("f"))
    {
        const auto focal = pixelValue(key);
        if (focal.has_value())
        {
            return focal;
        }
        const auto focalX = pixelValue(QStringLiteral("fu"));
        const auto focalY = pixelValue(QStringLiteral("fv"));
        if (focalX.has_value() && focalY.has_value())
        {
            return std::sqrt(std::abs(*focalX * *focalY));
        }
    }
    if (key == QStringLiteral("fu") || key == QStringLiteral("fv"))
    {
        return pixelValue(key);
    }
    if (key == QStringLiteral("cx") || key == QStringLiteral("cy"))
    {
        const auto centeredPrincipal = pixelValue(key);
        if (centeredPrincipal.has_value())
        {
            return centeredPrincipal;
        }
        const bool horizontal = key == QStringLiteral("cx");
        const auto principal = pixelValue(
            horizontal ? QStringLiteral("cu") : QStringLiteral("cv"));
        const int extent = horizontal ? record.imageWidth : record.imageHeight;
        if (principal.has_value() && extent > 0)
        {
            return *principal - extent * 0.5;
        }
    }
    return std::nullopt;
}

std::optional<double> meanParameter(
    const QVector<xjw::gui::camera_calibration::CameraCalibrationRecord> &records,
    const QVector<int> &indices,
    const QString &key,
    bool adjusted)
{
    double sum = 0.0;
    int count = 0;
    for (const int index : indices)
    {
        const auto &record = records.at(index);
        const bool available = adjusted ? record.hasAdjusted : record.hasInitial;
        const QJsonObject &camera = adjusted ? record.adjusted : record.initial;
        if (!available)
        {
            continue;
        }
        const auto value = calibrationParameter(record, camera, key);
        if (value.has_value() && std::isfinite(*value))
        {
            sum += *value;
            ++count;
        }
    }
    if (count == 0)
    {
        return std::nullopt;
    }
    return sum / count;
}

QString formatValue(const std::optional<double> &value)
{
    if (!value.has_value())
    {
        return QStringLiteral("—");
    }
    return QString::number(*value, 'g', 10);
}

QTableWidgetItem *readOnlyItem(const QString &text)
{
    auto *item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}

QString cameraUnit()
{
    return QStringLiteral("px");
}

bool parameterWasOptimized(
    const QVector<xjw::gui::camera_calibration::CameraCalibrationRecord> &records,
    const QVector<int> &indices,
    const QString &key)
{
    for (const int index : indices)
    {
        const QStringList &optimized = records.at(index).optimizedParameters;
        if (optimized.contains(key) ||
            (key == QStringLiteral("f") &&
             (optimized.contains(QStringLiteral("fu")) ||
              optimized.contains(QStringLiteral("fv")))))
        {
            return true;
        }
    }
    return false;
}

QString initialSourceLabel(const QString &source)
{
    if (source == QStringLiteral("automatic_focal_seed"))
    {
        return QStringLiteral("影像尺寸与焦距搜索自动生成");
    }
    if (source == QStringLiteral("project_camera_prior"))
    {
        return QStringLiteral("项目相机/标定先验");
    }
    if (source == QStringLiteral("image_metadata_focal_prior"))
    {
        return QStringLiteral("EXIF/影像元数据焦距先验");
    }
    return source.isEmpty() ? QStringLiteral("旧版记录（来源未注明）") : source;
}

QString adjustmentStatusLabel(const QString &status)
{
    if (status == QStringLiteral("refined"))
    {
        return QStringLiteral("束平差已优化内参");
    }
    if (status == QStringLiteral("coarse_seed_only"))
    {
        return QStringLiteral("仅采用焦距搜索种子，联合调整未被接受");
    }
    if (status == QStringLiteral("fixed_coarse_seed"))
    {
        return QStringLiteral("焦距种子固定，内参未释放");
    }
    if (status == QStringLiteral("trusted_prior_limited_refinement"))
    {
        return QStringLiteral("可信先验约束下的小范围内参优化");
    }
    if (status == QStringLiteral("trusted_prior_fixed") ||
        status == QStringLiteral("trusted_prior"))
    {
        return QStringLiteral("可信先验固定，内参未释放");
    }
    if (status == QStringLiteral("not_run"))
    {
        return QStringLiteral("尚未运行空三，没有调整值");
    }
    if (status == QStringLiteral("legacy_adjusted_only"))
    {
        return QStringLiteral("旧版空三只保存了最终值，无法可靠还原初始值");
    }
    return status.isEmpty() ? QStringLiteral("旧版记录（优化状态未注明）") : status;
}

} // namespace

CameraCalibrationDialog::CameraCalibrationDialog(const QJsonObject &projectMetadata,
                                                 const QString &projectAssetsDir,
                                                 QWidget *parent)
    : QDialog(parent)
{
    const QJsonObject report =
        xjw::gui::camera_calibration::readLatestCameraCalibrationReport(
            projectAssetsDir,
            &_reportError);
    _reportTimestamp = report.value(QStringLiteral("timestamp")).toString();
    _records = xjw::gui::camera_calibration::buildCameraCalibrationRecords(
        projectMetadata,
        report);

    buildInterface();
    buildGroups();
    if (_groups.isEmpty())
    {
        showEmptyState();
    }
    else
    {
        _cameraGroups->setCurrentRow(0);
    }
}

void CameraCalibrationDialog::buildInterface()
{
    setWindowTitle(tr("相机校准"));
    setObjectName(QStringLiteral("cameraCalibrationDialog"));
    resize(1040, 720);
    setMinimumSize(820, 560);

    auto *root = new QVBoxLayout(this);
    auto *intro = new QLabel(
        tr("“初始”是本次空三开始时使用的内方位先验；“调整”是连接点与束平差得到的内方位结果。"
           "cx/cy 按相对图像中心的像素偏移显示。本窗口不显示外方位 R/C，也不会修改项目相机。"),
        this);
    intro->setWordWrap(true);
    intro->setStyleSheet(QStringLiteral(
        "QLabel { background: #eef5ff; border: 1px solid #cbdcf5; border-radius: 5px; "
        "padding: 9px; color: #314b67; }"));
    root->addWidget(intro);

    auto *splitter = new QSplitter(Qt::Horizontal, this);
    _cameraGroups = new QListWidget(splitter);
    _cameraGroups->setObjectName(QStringLiteral("cameraCalibrationGroups"));
    _cameraGroups->setMinimumWidth(225);
    _cameraGroups->setMaximumWidth(310);
    _cameraGroups->setAlternatingRowColors(true);

    auto *right = new QWidget(splitter);
    auto *rightLayout = new QVBoxLayout(right);
    rightLayout->setContentsMargins(10, 0, 0, 0);
    _summaryLabel = new QLabel(right);
    _summaryLabel->setObjectName(QStringLiteral("cameraCalibrationSummary"));
    _summaryLabel->setWordWrap(true);
    rightLayout->addWidget(_summaryLabel);

    _calibrationTabs = new QTabWidget(right);
    _calibrationTabs->setObjectName(QStringLiteral("cameraCalibrationTabs"));
    _initialParameters = new QTableWidget(parameters().size(), 2, _calibrationTabs);
    _initialParameters->setObjectName(QStringLiteral("initialCalibrationParameters"));
    _initialParameters->setHorizontalHeaderLabels({tr("参数"), tr("初始值")});
    _adjustedParameters = new QTableWidget(parameters().size(), 5, _calibrationTabs);
    _adjustedParameters->setObjectName(QStringLiteral("adjustedCalibrationParameters"));
    _adjustedParameters->setHorizontalHeaderLabels(
        {tr("参数"), tr("初始值"), tr("调整值"), tr("变化"), tr("本次状态")});

    for (QTableWidget *table : {_initialParameters, _adjustedParameters})
    {
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setSelectionMode(QAbstractItemView::NoSelection);
        table->setAlternatingRowColors(true);
        table->verticalHeader()->setVisible(false);
        table->horizontalHeader()->setStretchLastSection(true);
    }
    _initialParameters->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _adjustedParameters->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _adjustedParameters->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    _calibrationTabs->addTab(_initialParameters, tr("初始"));
    _calibrationTabs->addTab(_adjustedParameters, tr("调整"));
    rightLayout->addWidget(_calibrationTabs, 3);

    auto *photoLabel = new QLabel(tr("组内照片"), right);
    QFont photoFont = photoLabel->font();
    photoFont.setBold(true);
    photoLabel->setFont(photoFont);
    rightLayout->addWidget(photoLabel);

    _photoTable = new QTableWidget(0, 4, right);
    _photoTable->setObjectName(QStringLiteral("cameraCalibrationPhotos"));
    _photoTable->setHorizontalHeaderLabels(
        {tr("图像"), tr("分辨率"), tr("相机模型"), tr("校准状态")});
    _photoTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _photoTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _photoTable->setAlternatingRowColors(true);
    _photoTable->verticalHeader()->setVisible(false);
    _photoTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    _photoTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _photoTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    _photoTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    rightLayout->addWidget(_photoTable, 2);

    splitter->addWidget(_cameraGroups);
    splitter->addWidget(right);
    splitter->setStretchFactor(1, 1);
    root->addWidget(splitter, 1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);

    connect(_cameraGroups, &QListWidget::currentRowChanged,
            this, &CameraCalibrationDialog::showSelectedCameraGroup);
}

void CameraCalibrationDialog::buildGroups()
{
    QMap<QString, QVector<int>> groupedIndices;
    for (int index = 0; index < _records.size(); ++index)
    {
        const auto &record = _records.at(index);
        const QString model = record.model.trimmed().isEmpty()
            ? tr("通用相机")
            : record.model;
        const QString resolution = record.imageWidth > 0 && record.imageHeight > 0
            ? QStringLiteral("%1×%2").arg(record.imageWidth).arg(record.imageHeight)
            : tr("分辨率未记录");
        groupedIndices[model + QLatin1Char('|') + resolution].append(index);
    }

    int groupNumber = 1;
    for (auto it = groupedIndices.constBegin(); it != groupedIndices.constEnd(); ++it)
    {
        const QStringList parts = it.key().split(QLatin1Char('|'));
        CameraGroup group;
        group.label = tr("相机组 %1 · %2\n%3 张照片，%4")
                          .arg(groupNumber++)
                          .arg(parts.value(0))
                          .arg(it.value().size())
                          .arg(parts.value(1));
        group.recordIndices = it.value();
        _groups.append(group);

        auto *item = new QListWidgetItem(
            style()->standardIcon(QStyle::SP_ComputerIcon),
            group.label,
            _cameraGroups);
        item->setSizeHint(QSize(item->sizeHint().width(), 54));
    }
}

void CameraCalibrationDialog::showSelectedCameraGroup(int row)
{
    if (row < 0 || row >= _groups.size())
    {
        return;
    }
    const CameraGroup &group = _groups.at(row);
    populateParameterTables(group);
    populatePhotoTable(group);

    int adjustedCount = 0;
    QString initialSource;
    QString adjustmentStatus;
    bool requiresReview = false;
    for (const int index : group.recordIndices)
    {
        const auto &record = _records.at(index);
        adjustedCount += record.hasAdjusted ? 1 : 0;
        if (initialSource.isEmpty())
        {
            initialSource = record.initialSource;
        }
        if (adjustmentStatus.isEmpty())
        {
            adjustmentStatus = record.adjustmentStatus;
        }
        requiresReview = requiresReview || record.requiresReview;
    }
    QString summary = tr("%1　·　初始来源：%2　·　内参状态：%3　·　记录：%4/%5 张")
                          .arg(group.label.section(QLatin1Char('\n'), 0, 0))
                          .arg(initialSourceLabel(initialSource))
                          .arg(adjustmentStatusLabel(adjustmentStatus))
                          .arg(adjustedCount)
                          .arg(group.recordIndices.size());
    if (!_reportTimestamp.isEmpty())
    {
        summary += tr("　·　记录时间：%1").arg(_reportTimestamp);
    }
    if (!_reportError.isEmpty())
    {
        summary += tr("\n注意：%1").arg(_reportError);
    }
    if (requiresReview)
    {
        summary += tr("\n警告：本次自标定被标记为需要复核，请结合连接点分布、穹顶效应和外方位检查。");
    }
    _summaryLabel->setText(summary);
}

void CameraCalibrationDialog::populateParameterTables(const CameraGroup &group)
{
    const QString unit = cameraUnit();
    _initialParameters->setHorizontalHeaderLabels(
        {tr("参数"), tr("初始值 (%1)").arg(unit)});
    _adjustedParameters->setHorizontalHeaderLabels(
        {tr("参数"), tr("初始值 (%1)").arg(unit), tr("调整值 (%1)").arg(unit),
         tr("变化"), tr("本次状态")});

    for (int row = 0; row < parameters().size(); ++row)
    {
        const ParameterDescriptor &parameter = parameters().at(row);
        const auto initial = meanParameter(_records, group.recordIndices, parameter.key, false);
        const auto adjusted = meanParameter(_records, group.recordIndices, parameter.key, true);
        const bool optimized = parameterWasOptimized(
            _records,
            group.recordIndices,
            parameter.key);

        _initialParameters->setItem(row, 0, readOnlyItem(parameter.label));
        _initialParameters->setItem(row, 1, readOnlyItem(formatValue(initial)));

        _adjustedParameters->setItem(row, 0, readOnlyItem(parameter.label));
        _adjustedParameters->setItem(row, 1, readOnlyItem(formatValue(initial)));
        _adjustedParameters->setItem(row, 2, readOnlyItem(formatValue(adjusted)));

        QString deltaText = QStringLiteral("—");
        double relativeDelta = 0.0;
        if (initial.has_value() && adjusted.has_value())
        {
            const double delta = *adjusted - *initial;
            deltaText = QStringLiteral("%1%2")
                            .arg(delta >= 0.0 ? QStringLiteral("+") : QString())
                            .arg(QString::number(delta, 'g', 8));
            if (parameter.percentageUseful && std::abs(*initial) > 1e-12)
            {
                relativeDelta = std::abs(delta / *initial * 100.0);
                deltaText += QStringLiteral("  (%1%2%)")
                                 .arg(delta >= 0.0 ? QStringLiteral("+") : QString())
                                 .arg(QString::number(delta / *initial * 100.0, 'f', 3));
            }
        }
        auto *deltaItem = readOnlyItem(deltaText);
        if (optimized && deltaText != QStringLiteral("—") && parameter.percentageUseful)
        {
            deltaItem->setForeground(relativeDelta > 2.0
                                         ? QColor(190, 55, 45)
                                         : (relativeDelta > 0.5
                                                ? QColor(190, 125, 35)
                                                : QColor(35, 135, 70)));
        }
        _adjustedParameters->setItem(row, 3, deltaItem);
        _adjustedParameters->setItem(
            row,
            4,
            readOnlyItem(!adjusted.has_value()
                             ? tr("无调整记录")
                             : (optimized ? tr("已优化") : tr("固定/未释放"))));
    }
}

void CameraCalibrationDialog::populatePhotoTable(const CameraGroup &group)
{
    _photoTable->setRowCount(group.recordIndices.size());
    for (int row = 0; row < group.recordIndices.size(); ++row)
    {
        const auto &record = _records.at(group.recordIndices.at(row));
        const QString resolution = record.imageWidth > 0 && record.imageHeight > 0
            ? QStringLiteral("%1×%2").arg(record.imageWidth).arg(record.imageHeight)
            : QStringLiteral("—");
        const QString model = record.model.isEmpty() ? tr("通用相机") : record.model;
        QString status = tr("无相机参数");
        if (record.hasInitial && record.hasAdjusted && !record.optimizedParameters.isEmpty())
        {
            status = tr("内参已优化");
        }
        else if (record.hasInitial && record.hasAdjusted)
        {
            status = tr("内参固定");
        }
        else if (record.hasInitial)
        {
            status = tr("仅初始值");
        }
        else if (record.hasAdjusted)
        {
            status = tr("仅调整值");
        }
        _photoTable->setItem(row, 0, readOnlyItem(record.name));
        _photoTable->item(row, 0)->setToolTip(record.path);
        _photoTable->setItem(row, 1, readOnlyItem(resolution));
        _photoTable->setItem(row, 2, readOnlyItem(model));
        _photoTable->setItem(row, 3, readOnlyItem(status));
    }
}

void CameraCalibrationDialog::showEmptyState()
{
    _summaryLabel->setText(
        tr("当前项目没有可显示的相机参数。请先导入/初始化相机，再运行空中三角测量以生成调整记录。"));
    _cameraGroups->addItem(tr("暂无相机"));
    _cameraGroups->setEnabled(false);
    _calibrationTabs->setEnabled(false);
    _photoTable->setRowCount(0);
}

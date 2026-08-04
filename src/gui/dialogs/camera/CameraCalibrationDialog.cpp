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
        {QStringLiteral("fu"), QStringLiteral("fx (fu)"), true},
        {QStringLiteral("fv"), QStringLiteral("fy (fv)"), true},
        {QStringLiteral("cu"), QStringLiteral("cx (cu)"), false},
        {QStringLiteral("cv"), QStringLiteral("cy (cv)"), false},
        {QStringLiteral("k1"), QStringLiteral("k1"), false},
        {QStringLiteral("k2"), QStringLiteral("k2"), false},
        {QStringLiteral("k3"), QStringLiteral("k3"), false},
        {QStringLiteral("p1"), QStringLiteral("p1"), false},
        {QStringLiteral("p2"), QStringLiteral("p2"), false}};
    return descriptors;
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
        if (!available || !camera.contains(key) || !camera.value(key).isDouble())
        {
            continue;
        }
        const double value = camera.value(key).toDouble();
        if (std::isfinite(value))
        {
            sum += value;
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

QString cameraUnit(const xjw::gui::camera_calibration::CameraCalibrationRecord &record)
{
    const QJsonObject camera = record.hasAdjusted ? record.adjusted : record.initial;
    return camera.value(QStringLiteral("intrinsics_unit")).toString() == QStringLiteral("mm")
        ? QStringLiteral("mm")
        : QStringLiteral("px");
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
        tr("查看空中三角测量（光束法平差）前后的相机内参。此窗口为只读，不会修改项目相机。"),
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
    _adjustedParameters = new QTableWidget(parameters().size(), 4, _calibrationTabs);
    _adjustedParameters->setObjectName(QStringLiteral("adjustedCalibrationParameters"));
    _adjustedParameters->setHorizontalHeaderLabels(
        {tr("参数"), tr("初始值"), tr("调整值"), tr("变化")});

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
    for (const int index : group.recordIndices)
    {
        adjustedCount += _records.at(index).hasAdjusted ? 1 : 0;
    }
    QString summary = tr("%1　·　空三调整记录：%2/%3 张")
                          .arg(group.label.section(QLatin1Char('\n'), 0, 0))
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
    _summaryLabel->setText(summary);
}

void CameraCalibrationDialog::populateParameterTables(const CameraGroup &group)
{
    const QString unit = cameraUnit(_records.at(group.recordIndices.first()));
    _initialParameters->setHorizontalHeaderLabels(
        {tr("参数"), tr("初始值 (%1)").arg(unit)});
    _adjustedParameters->setHorizontalHeaderLabels(
        {tr("参数"), tr("初始值 (%1)").arg(unit), tr("调整值 (%1)").arg(unit), tr("变化")});

    for (int row = 0; row < parameters().size(); ++row)
    {
        const ParameterDescriptor &parameter = parameters().at(row);
        const auto initial = meanParameter(_records, group.recordIndices, parameter.key, false);
        const auto adjusted = meanParameter(_records, group.recordIndices, parameter.key, true);

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
        if (deltaText != QStringLiteral("—") && parameter.percentageUseful)
        {
            deltaItem->setForeground(relativeDelta > 2.0
                                         ? QColor(190, 55, 45)
                                         : (relativeDelta > 0.5
                                                ? QColor(190, 125, 35)
                                                : QColor(35, 135, 70)));
        }
        _adjustedParameters->setItem(row, 3, deltaItem);
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
        if (record.hasInitial && record.hasAdjusted)
        {
            status = tr("已调整");
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

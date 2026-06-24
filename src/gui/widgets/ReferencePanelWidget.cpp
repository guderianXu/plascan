#include "ReferencePanelWidget.h"
#include "ui_ReferencePanelWidget.h"

#include <QTableWidget>
#include <QHeaderView>
#include <QPushButton>
#include <QItemSelectionModel>
#include <QJsonArray>
#include <QJsonDocument>
#include <QFileInfo>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool computeYprFromR(const QJsonObject &camObj, double *yawDeg, double *pitchDeg, double *rollDeg)
{
    if (!yawDeg || !pitchDeg || !rollDeg) return false;
    const QJsonArray r = camObj.value(QStringLiteral("R")).toArray();
    if (r.size() < 9) return false;

    // 约定：R 为 camera->world，按 ZYX 欧拉角分解（yaw-pitch-roll）。
    const double r00 = r.at(0).toDouble();
    const double r10 = r.at(3).toDouble();
    const double r20 = r.at(6).toDouble();
    const double r21 = r.at(7).toDouble();
    const double r22 = r.at(8).toDouble();

    const double pitch = std::asin(std::clamp(-r20, -1.0, 1.0));
    double yaw = 0.0;
    double roll = 0.0;

    if (std::abs(std::cos(pitch)) > 1e-8) {
        yaw = std::atan2(r10, r00);
        roll = std::atan2(r21, r22);
    } else {
        yaw = std::atan2(-r.at(1).toDouble(), r.at(4).toDouble());
        roll = 0.0;
    }

    constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
    *yawDeg = yaw * kRadToDeg;
    *pitchDeg = pitch * kRadToDeg;
    *rollDeg = roll * kRadToDeg;
    return true;
}

} // namespace

ReferencePanelWidget::ReferencePanelWidget(QWidget *parent)
    : QWidget(parent)
{
    Ui::ReferencePanelWidget ui;
    ui.setupUi(this);

    _table = ui.m_table;
    _exactImportBtn = ui.m_exactImportBtn;
    _batchImportBtn = ui.m_batchImportBtn;
    _clearCameraBtn = ui.m_clearCameraBtn;

    _table->setColumnCount(10);
    _table->setHorizontalHeaderLabels({
        tr("影像"),
        tr("状态"),
        tr("X (m)"),
        tr("Y (m)"),
        tr("Z (m)"),
        tr("偏航 (deg)"),
        tr("俯仰 (deg)"),
        tr("滚转 (deg)"),
        tr("fu"),
        tr("fv")
    });
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(7, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(8, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(9, QHeaderView::Stretch);
    _table->verticalHeader()->setVisible(false);
    _table->setAlternatingRowColors(true);
    _table->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    _table->setWordWrap(false);

    connect(_exactImportBtn, &QPushButton::clicked, this, &ReferencePanelWidget::onExactImportClicked);
    connect(_batchImportBtn, &QPushButton::clicked, this, &ReferencePanelWidget::onBatchImportClicked);
    connect(_clearCameraBtn, &QPushButton::clicked, this, &ReferencePanelWidget::onClearCameraClicked);
    connect(_table, &QTableWidget::cellDoubleClicked, this, &ReferencePanelWidget::onCellDoubleClicked);
    connect(_table->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &ReferencePanelWidget::onSelectionChanged);
}

ReferencePanelWidget::~ReferencePanelWidget() = default;

void ReferencePanelWidget::loadFromJson(const QJsonObject &meta)
{
    if (meta.isEmpty()) return;

    QJsonObject normalized = normalizeMeta(meta);
    if (normalized.isEmpty()) return;

    QJsonArray images = normalized.value(QStringLiteral("images")).toArray();
    rebuildTable(images);
}

void ReferencePanelWidget::onExactImportClicked()
{
    const QString imagePath = selectedImagePath();
    if (imagePath.isEmpty()) return;
    emit exactImportRequested(imagePath);
}

void ReferencePanelWidget::onBatchImportClicked()
{
    emit batchImportRequested();
}

void ReferencePanelWidget::onClearCameraClicked()
{
    const QStringList paths = selectedImagePaths();
    if (!paths.isEmpty())
        emit clearCameraRequested(paths);
}

void ReferencePanelWidget::onSelectionChanged()
{
    const bool hasSel = !selectedImagePath().isEmpty();
    _exactImportBtn->setEnabled(hasSel);
    _clearCameraBtn->setEnabled(hasSel);
}

void ReferencePanelWidget::onCellDoubleClicked(int row, int column)
{
    Q_UNUSED(column);
    if (row < 0 || row >= _table->rowCount()) return;
    QString path = _table->item(row, 0) ? _table->item(row, 0)->data(Qt::UserRole).toString() : QString();
    if (!path.isEmpty()) emit imageActivated(path);
}

QJsonObject ReferencePanelWidget::normalizeMeta(const QJsonObject &meta) const
{
    if (meta.contains(QStringLiteral("images"))) {
        return meta;
    }
    if (meta.value(QStringLiteral("project_files")).isObject()) {
        return meta.value(QStringLiteral("project_files")).toObject();
    }
    return QJsonObject();
}

QString ReferencePanelWidget::selectedImagePath() const
{
    const int row = _table->currentRow();
    if (row < 0 || row >= _table->rowCount()) return QString();
    if (!_table->item(row, 0)) return QString();
    return _table->item(row, 0)->data(Qt::UserRole).toString();
}

QStringList ReferencePanelWidget::selectedImagePaths() const
{
    QStringList paths;
    const auto rows = _table->selectionModel()->selectedRows();
    for (const auto &idx : rows) {
        if (auto *item = _table->item(idx.row(), 0)) {
            const QString p = item->data(Qt::UserRole).toString();
            if (!p.isEmpty()) paths.append(p);
        }
    }
    if (paths.isEmpty()) {
        const QString p = selectedImagePath();
        if (!p.isEmpty()) paths.append(p);
    }
    return paths;
}

void ReferencePanelWidget::rebuildTable(const QJsonArray &images)
{
    _table->setRowCount(0);
    _table->setRowCount(images.size());

    for (int i = 0; i < images.size(); ++i) {
        const QJsonObject imgObj = images.at(i).toObject();
        const QString imagePath = imgObj.value(QStringLiteral("path")).toString();
        const QString imageName = QFileInfo(imagePath).fileName().isEmpty()
                                  ? imagePath
                                  : QFileInfo(imagePath).fileName();

        const QJsonObject camObj = imgObj.value(QStringLiteral("camera")).toObject();
        const bool hasCamera = !camObj.isEmpty();

        auto *nameItem = new QTableWidgetItem(imageName);
        nameItem->setData(Qt::UserRole, imagePath);
        auto *statusItem = new QTableWidgetItem(hasCamera ? tr("C, OK") : tr("NC, NA"));

        const QJsonArray c = camObj.value(QStringLiteral("C")).toArray();
        const QString x = c.size() > 0 ? QString::number(c.at(0).toDouble(), 'f', 6) : QString();
        const QString y = c.size() > 1 ? QString::number(c.at(1).toDouble(), 'f', 6) : QString();
        const QString z = c.size() > 2 ? QString::number(c.at(2).toDouble(), 'f', 6) : QString();

        // 兼容旧项目：若未存 yaw/pitch/roll，则由 R 即时反算并显示。
        double yawVal = camObj.value(QStringLiteral("yaw_deg")).toDouble(std::numeric_limits<double>::quiet_NaN());
        double pitchVal = camObj.value(QStringLiteral("pitch_deg")).toDouble(std::numeric_limits<double>::quiet_NaN());
        double rollVal = camObj.value(QStringLiteral("roll_deg")).toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(yawVal) || !std::isfinite(pitchVal) || !std::isfinite(rollVal)) {
            double y = 0.0, p = 0.0, rDeg = 0.0;
            if (computeYprFromR(camObj, &y, &p, &rDeg)) {
                yawVal = y;
                pitchVal = p;
                rollVal = rDeg;
            }
        }

        const QString yaw = std::isfinite(yawVal) ? QString::number(yawVal, 'f', 6) : QString();
        const QString pitch = std::isfinite(pitchVal) ? QString::number(pitchVal, 'f', 6) : QString();
        const QString roll = std::isfinite(rollVal) ? QString::number(rollVal, 'f', 6) : QString();

        const QString fu = camObj.contains(QStringLiteral("fu"))
                         ? QString::number(camObj.value(QStringLiteral("fu")).toDouble(), 'f', 6)
                         : QString();
        const QString fv = camObj.contains(QStringLiteral("fv"))
                         ? QString::number(camObj.value(QStringLiteral("fv")).toDouble(), 'f', 6)
                         : QString();

        _table->setItem(i, 0, nameItem);
        _table->setItem(i, 1, statusItem);
        _table->setItem(i, 2, new QTableWidgetItem(x));
        _table->setItem(i, 3, new QTableWidgetItem(y));
        _table->setItem(i, 4, new QTableWidgetItem(z));
        _table->setItem(i, 5, new QTableWidgetItem(yaw));
        _table->setItem(i, 6, new QTableWidgetItem(pitch));
        _table->setItem(i, 7, new QTableWidgetItem(roll));
        _table->setItem(i, 8, new QTableWidgetItem(fu));
        _table->setItem(i, 9, new QTableWidgetItem(fv));
    }

    _exactImportBtn->setEnabled(!selectedImagePath().isEmpty());
    _clearCameraBtn->setEnabled(!selectedImagePath().isEmpty());
}

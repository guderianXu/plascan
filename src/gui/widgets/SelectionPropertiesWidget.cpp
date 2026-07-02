#include "SelectionPropertiesWidget.h"

#include <QAbstractItemView>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonValue>
#include <QLabel>
#include <QSize>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

namespace
{

QString cleanPath(const QString &path)
{
    return QDir::cleanPath(path);
}

bool samePathText(const QString &left, const QString &right)
{
    if (left.isEmpty() || right.isEmpty())
    {
        return false;
    }
    return QString::compare(cleanPath(left), cleanPath(right), Qt::CaseInsensitive) == 0;
}

bool imageEntryMatches(const QJsonObject &entry,
                       const QString &targetPath,
                       const QString &targetAbsolutePath,
                       const QString &targetName)
{
    const QString path = entry.value(QStringLiteral("path")).toString();
    const QFileInfo pathInfo(path);
    const QString absolutePath = pathInfo.absoluteFilePath();
    const QString name = entry.value(QStringLiteral("name")).toString();
    const QString fileName = pathInfo.fileName();

    return samePathText(path, targetPath)
        || samePathText(path, targetAbsolutePath)
        || samePathText(absolutePath, targetPath)
        || samePathText(absolutePath, targetAbsolutePath)
        || (!targetName.isEmpty() && (name == targetName || fileName == targetName));
}

QJsonObject findImageEntryInArray(const QJsonArray &images,
                                  const QString &targetPath,
                                  const QString &targetAbsolutePath,
                                  const QString &targetName)
{
    for (const QJsonValue &value : images)
    {
        const QJsonObject entry = value.toObject();
        if (imageEntryMatches(entry, targetPath, targetAbsolutePath, targetName))
        {
            return entry;
        }
    }
    return {};
}

} // namespace

SelectionPropertiesWidget::SelectionPropertiesWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(4);

    _title = new QLabel(tr("未选择"), this);
    _title->setTextInteractionFlags(Qt::TextSelectableByMouse);

    _table = new QTableWidget(this);
    _table->setColumnCount(2);
    _table->setHorizontalHeaderLabels({tr("属性"), tr("值")});
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->verticalHeader()->setVisible(false);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setSelectionMode(QAbstractItemView::NoSelection);
    _table->setAlternatingRowColors(true);

    layout->addWidget(_title);
    layout->addWidget(_table, 1);
}

void SelectionPropertiesWidget::clearSelection()
{
    setRows(tr("未选择"), {});
}

void SelectionPropertiesWidget::showPhotoProperties(const QJsonObject &meta, const QString &imagePath)
{
    QVector<PropertyRow> rows;
    const QFileInfo info(imagePath);
    const QJsonObject entry = findImageEntry(meta, imagePath);

    rows.push_back({tr("名称"), info.fileName()});
    rows.push_back({tr("路径"), imagePath});
    rows.push_back({tr("定向状态"), imageAlignedText(entry)});

    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    const QSize size = reader.size();
    if (size.isValid())
    {
        rows.push_back({tr("尺寸"), QStringLiteral("%1 x %2").arg(size.width()).arg(size.height())});
    }
    const QByteArray format = reader.format();
    rows.push_back({tr("格式"), format.isEmpty() ? tr("未知") : QString::fromLatin1(format).toUpper()});

    appendFileRows(&rows, imagePath);

    const QString center = cameraCenterText(entry);
    if (!center.isEmpty())
    {
        rows.push_back({tr("相机中心"), center});
    }
    const QString intrinsics = intrinsicsText(entry);
    if (!intrinsics.isEmpty())
    {
        rows.push_back({tr("内方位"), intrinsics});
    }

    setRows(tr("照片属性"), rows);
}

void SelectionPropertiesWidget::showResourceProperties(const QJsonObject &meta,
                                                       const QString &section,
                                                       const QString &resourcePath)
{
    Q_UNUSED(meta)

    QVector<PropertyRow> rows;
    const QFileInfo info(resourcePath);
    rows.push_back({tr("类型"), section});
    rows.push_back({tr("名称"), info.fileName().isEmpty() ? section : info.fileName()});
    rows.push_back({tr("路径"), resourcePath});
    rows.push_back({tr("扩展名"), info.suffix().isEmpty() ? tr("无") : info.suffix().toLower()});
    appendFileRows(&rows, resourcePath);

    if (section.contains(tr("点云")) || section.contains(tr("连接点")))
    {
        rows.push_back({tr("详细属性"), tr("未扫描详细属性，避免在主界面阻塞大点云加载")});
    }

    setRows(tr("资源属性"), rows);
}

void SelectionPropertiesWidget::setRows(const QString &title, const QVector<PropertyRow> &rows)
{
    if (_title)
    {
        _title->setText(title);
    }
    if (!_table)
    {
        return;
    }

    _table->setRowCount(rows.size());
    for (int row = 0; row < rows.size(); ++row)
    {
        auto *nameItem = new QTableWidgetItem(rows[row].name);
        auto *valueItem = new QTableWidgetItem(rows[row].value);
        nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
        valueItem->setFlags(valueItem->flags() & ~Qt::ItemIsEditable);
        _table->setItem(row, 0, nameItem);
        _table->setItem(row, 1, valueItem);
    }
    _table->resizeRowsToContents();
}

void SelectionPropertiesWidget::appendFileRows(QVector<PropertyRow> *rows, const QString &path) const
{
    if (!rows)
    {
        return;
    }
    const QFileInfo info(path);
    rows->push_back({tr("存在"), info.exists() ? tr("是") : tr("否")});
    if (info.exists())
    {
        rows->push_back({tr("文件大小"), fileSizeText(info.size())});
        rows->push_back({tr("修改时间"), info.lastModified().toString(Qt::ISODate)});
    }
}

QJsonObject SelectionPropertiesWidget::findImageEntry(const QJsonObject &meta, const QString &imagePath) const
{
    const QFileInfo targetInfo(imagePath);
    const QString targetPath = imagePath;
    const QString targetAbsolutePath = targetInfo.absoluteFilePath();
    const QString targetName = targetInfo.fileName();

    const QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    QJsonObject entry = findImageEntryInArray(images, targetPath, targetAbsolutePath, targetName);
    if (!entry.isEmpty())
    {
        return entry;
    }

    const QJsonObject projectFiles = meta.value(QStringLiteral("project_files")).toObject();
    const QJsonArray projectImages = projectFiles.value(QStringLiteral("images")).toArray();
    return findImageEntryInArray(projectImages, targetPath, targetAbsolutePath, targetName);
}

QString SelectionPropertiesWidget::imageAlignedText(const QJsonObject &entry) const
{
    if (entry.isEmpty())
    {
        return tr("未知");
    }
    const QJsonObject camera = entry.value(QStringLiteral("camera")).toObject();
    const bool hasCenter = entry.contains(QStringLiteral("center"))
        || entry.contains(QStringLiteral("camera_center"))
        || camera.contains(QStringLiteral("C"));
    const bool aligned = entry.value(QStringLiteral("aligned")).toBool(hasCenter);
    return aligned ? tr("已定向") : tr("未定向");
}

QString SelectionPropertiesWidget::cameraCenterText(const QJsonObject &entry) const
{
    QJsonArray center = entry.value(QStringLiteral("center")).toArray();
    if (center.isEmpty())
    {
        center = entry.value(QStringLiteral("camera_center")).toArray();
    }
    if (center.isEmpty())
    {
        const QJsonObject camera = entry.value(QStringLiteral("camera")).toObject();
        center = camera.value(QStringLiteral("C")).toArray();
    }
    if (center.size() < 3)
    {
        return {};
    }
    return QStringLiteral("%1, %2, %3")
        .arg(center.at(0).toDouble(), 0, 'f', 3)
        .arg(center.at(1).toDouble(), 0, 'f', 3)
        .arg(center.at(2).toDouble(), 0, 'f', 3);
}

QString SelectionPropertiesWidget::intrinsicsText(const QJsonObject &entry) const
{
    const QJsonObject intrinsics = entry.value(QStringLiteral("intrinsics")).toObject();
    if (!intrinsics.isEmpty())
    {
        return QStringLiteral("fx=%1, fy=%2, cx=%3, cy=%4")
            .arg(intrinsics.value(QStringLiteral("fx")).toDouble(), 0, 'f', 2)
            .arg(intrinsics.value(QStringLiteral("fy")).toDouble(), 0, 'f', 2)
            .arg(intrinsics.value(QStringLiteral("cx")).toDouble(), 0, 'f', 2)
            .arg(intrinsics.value(QStringLiteral("cy")).toDouble(), 0, 'f', 2);
    }

    const QJsonObject camera = entry.value(QStringLiteral("camera")).toObject();
    if (camera.isEmpty())
    {
        return {};
    }
    return QStringLiteral("fu=%1, fv=%2, cu=%3, cv=%4")
        .arg(camera.value(QStringLiteral("fu")).toDouble(), 0, 'f', 2)
        .arg(camera.value(QStringLiteral("fv")).toDouble(), 0, 'f', 2)
        .arg(camera.value(QStringLiteral("cu")).toDouble(), 0, 'f', 2)
        .arg(camera.value(QStringLiteral("cv")).toDouble(), 0, 'f', 2);
}

QString SelectionPropertiesWidget::fileSizeText(qint64 bytes)
{
    if (bytes < 1024)
    {
        return QStringLiteral("%1 B").arg(bytes);
    }
    if (bytes < 1024 * 1024)
    {
        return QStringLiteral("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    }
    return QStringLiteral("%1 MB").arg(bytes / 1024.0 / 1024.0, 0, 'f', 1);
}

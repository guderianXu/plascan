#pragma once

#include <QJsonObject>
#include <QString>
#include <QVector>
#include <QWidget>

class QLabel;
class QTableWidget;

class SelectionPropertiesWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SelectionPropertiesWidget(QWidget *parent = nullptr);

public slots:
    void clearSelection();
    void showPhotoProperties(const QJsonObject &meta, const QString &imagePath);
    void showResourceProperties(const QJsonObject &meta,
                                const QString &section,
                                const QString &resourcePath);

signals:
    void selectionStateChanged(bool hasSelection);

private:
    struct PropertyRow
    {
        QString name;
        QString value;
        bool sectionHeader = false;
    };

    void setRows(const QString &title, const QVector<PropertyRow> &rows);
    void appendFileRows(QVector<PropertyRow> *rows, const QString &path) const;
    QJsonObject findImageEntry(const QJsonObject &meta, const QString &imagePath) const;
    QJsonObject findResourceRecord(const QJsonObject &meta,
                                   const QString &section,
                                   const QString &resourcePath) const;
    QVector<PropertyRow> modelPropertyRows(const QJsonObject &meta,
                                           const QJsonObject &record,
                                           const QString &resourcePath) const;
    QString imageAlignedText(const QJsonObject &entry) const;
    QString cameraCenterText(const QJsonObject &entry) const;
    QString intrinsicsText(const QJsonObject &entry) const;
    static QString fileSizeText(qint64 bytes);

    QLabel *_title = nullptr;
    QTableWidget *_table = nullptr;
    bool _hasSelection = false;
};

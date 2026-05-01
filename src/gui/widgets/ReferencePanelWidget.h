#pragma once

#include <QWidget>
#include <QJsonObject>
#include <QJsonArray>

class QTableWidget;
class QPushButton;

class ReferencePanelWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ReferencePanelWidget(QWidget *parent = nullptr);
    ~ReferencePanelWidget() override;

public slots:
    void loadFromJson(const QJsonObject &meta);

signals:
    void exactImportRequested(const QString &imagePath);
    void batchImportRequested();
    void clearCameraRequested(const QStringList &imagePaths);
    void imageActivated(const QString &imagePath);

private slots:
    void onExactImportClicked();
    void onBatchImportClicked();
    void onClearCameraClicked();
    void onSelectionChanged();
    void onCellDoubleClicked(int row, int column);

private:
    QJsonObject normalizeMeta(const QJsonObject &meta) const;
    QString selectedImagePath() const;
    QStringList selectedImagePaths() const;
    void rebuildTable(const QJsonArray &images);

    QTableWidget *m_table{};
    QPushButton *m_exactImportBtn{};
    QPushButton *m_batchImportBtn{};
    QPushButton *m_clearCameraBtn{};
};

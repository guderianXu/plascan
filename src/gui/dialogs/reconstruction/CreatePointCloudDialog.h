#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QString>

class QCheckBox;
class QComboBox;
class QLabel;
class QPushButton;

/**
 * @brief Metashape 风格的创建点云参数对话框。
 *
 * 对话框只负责展示、恢复和提交点云生成参数；耗时任务由工作流控制器处理。
 */
class CreatePointCloudDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CreatePointCloudDialog(QWidget *parent = nullptr);

    void applySettings(const QJsonObject &settings);
    void setProjectState(bool hasProductionSparseResult,
                         bool hasReusableDepthMaps,
                         bool hasExistingPointCloud,
                         const QString &blockingReason = QString());

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private:
    QJsonObject collectSettings() const;
    void emitSettingsNow();
    void updateAvailability();
    void onRun();

    QComboBox *_qualityCombo = nullptr;
    QCheckBox *_reuseDepthMapsCheck = nullptr;
    QCheckBox *_saveEachStepCheck = nullptr;
    QComboBox *_depthFilterCombo = nullptr;
    QCheckBox *_calculateColorsCheck = nullptr;
    QCheckBox *_replaceDefaultCheck = nullptr;
    QLabel *_statusLabel = nullptr;
    QPushButton *_okButton = nullptr;

    bool _hasProductionSparseResult = false;
    bool _hasReusableDepthMaps = false;
    bool _hasExistingPointCloud = false;
    bool _reuseDepthMapsRequested = true;
    bool _replaceDefaultRequested = false;
    QString _blockingReason;
};

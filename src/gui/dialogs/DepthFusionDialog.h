#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;

/**
 * @brief 深度图融合生成密集点云对话框。
 *
 * 将多张深度图按一致性检查和融合策略合并为统一的密集点云。
 */
class DepthFusionDialog : public QDialog
{
    Q_OBJECT
public:
    explicit DepthFusionDialog(QWidget *parent = nullptr);
    void applySettings(const QJsonObject &settings);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void emitSettingsNow();
    void onRun();

private:
    QJsonObject collectSettings() const;

    QComboBox *_fusionMethodCombo = nullptr;       ///< 融合方法
    QDoubleSpinBox *_depthConsistSpin = nullptr;   ///< 深度一致性阈值
    QSpinBox *_minConsistViewSpin = nullptr;       ///< 最少一致视图
    QDoubleSpinBox *_normalConsistSpin = nullptr;  ///< 法向量一致性
    QDoubleSpinBox *_voxelSizeSpin = nullptr;      ///< 体素大小
    QDoubleSpinBox *_minConfidenceSpin = nullptr;  ///< 最小置信度
    QDoubleSpinBox *_maxReprojSpin = nullptr;      ///< 最大重投影误差
    QCheckBox *_colorCheck = nullptr;              ///< 保留颜色
    QCheckBox *_normalCheck = nullptr;             ///< 保留法向量
    QSpinBox *_threadsSpin = nullptr;
    QCheckBox *_cudaCheck = nullptr;
    QLabel *_infoLabel = nullptr;
};

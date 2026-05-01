#pragma once

#include <QDialog>
#include <QJsonObject>

class QGroupBox;
class QSpinBox;
class QDoubleSpinBox;
class QComboBox;

/**
 * @brief 密集点云后处理对话框。
 *
 * 包括统计离群点移除、体素下采样、法向量估计/平滑、颜色校正等。
 * 每个处理步骤通过 GroupBox 勾选框独立开关。
 */
class DenseCloudRefineDialog : public QDialog
{
    Q_OBJECT
public:
    explicit DenseCloudRefineDialog(QWidget *parent = nullptr);
    void applySettings(const QJsonObject &settings);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void emitSettingsNow();
    void onRun();

private:
    QJsonObject collectSettings() const;

    // 处理步骤开关（checkable GroupBox）
    QGroupBox      *m_sorGroup          = nullptr;  ///< 统计离群点移除
    QGroupBox      *m_voxelGroup        = nullptr;  ///< 体素下采样
    QGroupBox      *m_normalGroup       = nullptr;  ///< 法向量估计
    QGroupBox      *m_colorGroup        = nullptr;  ///< 颜色校正

    // SOR
    QSpinBox       *m_sorKSpin          = nullptr;
    QDoubleSpinBox *m_sorStdSpin        = nullptr;
    // 下采样
    QDoubleSpinBox *m_voxelSizeSpin     = nullptr;
    // 法向量
    QSpinBox       *m_normalKSpin       = nullptr;
    QSpinBox       *m_smoothIterSpin    = nullptr;
    // 颜色
    QComboBox      *m_colorMethodCombo  = nullptr;
    // 系统
    QSpinBox       *m_threadsSpin       = nullptr;
};

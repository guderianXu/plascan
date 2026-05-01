#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QLineEdit;
class QPushButton;

/**
 * @brief 模型导出对话框。
 *
 * 选择导出格式、坐标系变换、包含/排除数据通道，以及简化与输出路径。
 */
class ModelExportDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ModelExportDialog(QWidget *parent = nullptr);
    void applySettings(const QJsonObject &settings);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void emitSettingsNow();
    void onRun();
    void onBrowseOutput();

private:
    QJsonObject collectSettings() const;

    QComboBox      *m_formatCombo      = nullptr;  ///< OBJ/PLY/glTF/FBX/STL
    QComboBox      *m_coordSysCombo    = nullptr;  ///< 坐标系
    QCheckBox      *m_includeTexCheck  = nullptr;
    QCheckBox      *m_includeNormalCheck = nullptr;
    QCheckBox      *m_includeColorCheck = nullptr;
    QCheckBox      *m_simplifyCheck    = nullptr;
    QDoubleSpinBox *m_simplifyRatioSpin= nullptr;
    QComboBox      *m_upAxisCombo      = nullptr;  ///< Up 轴
    QLineEdit      *m_outputPathEdit   = nullptr;
    QPushButton    *m_browseBtn        = nullptr;
};

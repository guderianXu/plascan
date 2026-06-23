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

    QComboBox *_formatCombo = nullptr;       ///< OBJ/PLY/glTF/FBX/STL
    QComboBox *_coordSysCombo = nullptr;     ///< 坐标系
    QCheckBox *_includeTexCheck = nullptr;
    QCheckBox *_includeNormalCheck = nullptr;
    QCheckBox *_includeColorCheck = nullptr;
    QCheckBox *_simplifyCheck = nullptr;
    QDoubleSpinBox *_simplifyRatioSpin = nullptr;
    QComboBox *_upAxisCombo = nullptr;       ///< Up 轴
    QLineEdit *_outputPathEdit = nullptr;
    QPushButton *_browseBtn = nullptr;
};

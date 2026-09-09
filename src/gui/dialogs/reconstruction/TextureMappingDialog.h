#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QDoubleSpinBox;
class QCheckBox;

/**
 * @brief 纹理映射对话框。
 *
 * 使用 Natural 映射流程从已定向影像生成网格纹理。
 */
class TextureMappingDialog : public QDialog
{
    Q_OBJECT
public:
    /// 构造纹理映射参数对话框。
    /// @param parent 父窗口。
    explicit TextureMappingDialog(QWidget *parent = nullptr);

    /// 应用持久化保存的纹理映射配置。
    /// @param settings 配置参数 JSON 对象。
    void applySettings(const QJsonObject &settings);

signals:
    /// 用户确认执行纹理映射时发出。
    /// @param settings 当前收集到的配置参数。
    void runRequested(const QJsonObject &settings);

    /// 参数变化时发出，供外部实时持久化。
    /// @param settings 当前收集到的配置参数。
    void settingsChanged(const QJsonObject &settings);

private slots:
    /// 收集当前参数并发出 settingsChanged 信号。
    void emitSettingsNow();

    /// 响应“运行”按钮点击事件。
    void onRun();

private:
    /// 收集对话框当前配置。
    /// @return 配置参数 JSON 对象。
    QJsonObject collectSettings() const;

    QComboBox *_texSizeCombo = nullptr;     ///< 纹理分辨率
    QComboBox *_imageDownscaleCombo = nullptr; ///< 源影像下采样倍率
    QComboBox *_antiAliasingCombo = nullptr; ///< 烘焙抗锯齿采样倍率
    QCheckBox *_holeFillCheck = nullptr;    ///< 纹理孔洞填充
    QCheckBox *_colorCorrCheck = nullptr;   ///< 色彩一致性校正
    QCheckBox *_ghostFilterCheck = nullptr; ///< 去除鬼影
    QCheckBox *_outOfFocusFilterCheck = nullptr; ///< 焦外影像过滤
    QDoubleSpinBox *_seamsMarginSpin = nullptr; ///< 锐化强度
};

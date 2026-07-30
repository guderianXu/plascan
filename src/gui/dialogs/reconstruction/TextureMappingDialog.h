#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;

/**
 * @brief 纹理映射对话框。
 *
 * 将原始图像纹理投影到网格面上，支持多种混合策略和分辨率选择。
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

    QComboBox *_textureTypeCombo = nullptr; ///< 纹理生成类型
    QComboBox *_sourceCombo = nullptr;      ///< 纹理源数据
    QComboBox *_blendCombo = nullptr;       ///< 混合方式
    QComboBox *_texSizeCombo = nullptr;     ///< 纹理分辨率
    QComboBox *_uvMethodCombo = nullptr;    ///< UV 展开方式
    QComboBox *_imageDownscaleCombo = nullptr; ///< 源影像下采样倍率
    QCheckBox *_saveEachStepCheck = nullptr; ///< 每步完成后保存项目
    QCheckBox *_holeFillCheck = nullptr;    ///< 纹理孔洞填充
    QCheckBox *_colorCorrCheck = nullptr;   ///< 色彩一致性校正
    QCheckBox *_ghostFilterCheck = nullptr; ///< 去除鬼影
    QCheckBox *_outOfFocusFilterCheck = nullptr; ///< 焦外影像过滤
    QCheckBox *_useAssignedImagesCheck = nullptr; ///< 仅使用指定影像
    QCheckBox *_transferTextureCheck = nullptr; ///< 转移已有纹理
    QDoubleSpinBox *_seamsMarginSpin = nullptr; ///< 接缝边距
    QSpinBox *_paddingSpin = nullptr;       ///< 纹理填充边距
    QCheckBox *_keepUnmappedCheck = nullptr; ///< 保留无纹理区域
};

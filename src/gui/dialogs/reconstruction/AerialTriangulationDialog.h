#pragma once

#include <QDialog>
#include <QJsonObject>

#include <memory>

namespace Ui
{
class AerialTriangulationDialog;
}

/**
 * @brief 空中三角测量参数对话框。
 *
 * 该类只负责展示、恢复和收集 GUI 参数，不直接执行特征提取、匹配、SfM 或 BA。
 * collectSettings() 生成的 JSON 由上层工作流服务转换为实际运行配置。
 */
class AerialTriangulationDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AerialTriangulationDialog(QWidget *parent = nullptr);
    ~AerialTriangulationDialog() override;

    /// 更新当前项目的影像数量提示。
    void setImageCount(int count);

    /**
     * @brief 更新参考预选的可用性提示。
     *
     * 即使相机参考不完整，也不会直接禁用控件，因为“照片序列”模式不依赖相机文件。
     */
    void setReferencePreselectionAvailable(bool available,
                                           int cameraCount = 0,
                                           int imageCount = 0);

    /// 从项目或工作流 JSON 恢复界面状态；恢复期间不会发送 settingsChanged()。
    void applySettings(const QJsonObject &settings);

    /// 将当前界面状态序列化为供上层工作流使用的稳定 JSON 字段。
    QJsonObject collectSettings() const;

signals:
    /// 用户修改任一有效参数后发送完整配置快照。
    void settingsChanged(const QJsonObject &settings);

private:
    /// 初始化选项、默认值、提示文案以及控件间的联动关系。
    void setupUi();

    /// 展开或收起高级参数区域，并同步调整对话框尺寸。
    void setAdvancedExpanded(bool expanded);

    /// 在非批量恢复状态下收集并发送最新配置。
    void emitSettingsChanged();

    std::unique_ptr<Ui::AerialTriangulationDialog> _ui;

    // 防止 applySettings() 批量更新控件时产生中间状态信号。
    bool _applyingSettings = false;

};

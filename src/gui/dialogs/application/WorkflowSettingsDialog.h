#pragma once

/**
 * @file WorkflowSettingsDialog.h
 * @brief 工作流程级高级参数设置对话框。
 *
 * 普通“空中三角测量”对话框保留摄影测量用户经常调整的参数；本对话框承载
 * 设备、并行度和数值门限等实现级参数。两类参数在启动工作流时合并，核心层
 * 始终收到一份完整配置，避免 GUI 中存在只显示但不生效的选项。
 */

#include <QDialog>
#include <QJsonObject>

class QDoubleSpinBox;
class QSpinBox;

class WorkflowSettingsDialog final : public QDialog
{
public:
    explicit WorkflowSettingsDialog(QWidget *parent = nullptr);

    /// 返回稳定的默认配置；项目中尚未保存设置时也使用同一份默认值。
    static QJsonObject defaultSettings();

    /// 从项目级 JSON 恢复控件；缺失字段使用 defaultSettings() 补齐。
    void applySettings(const QJsonObject &settings);

    /// 收集可直接合并到 AerialTriangulationOptions 的稳定字段。
    QJsonObject collectSettings() const;

private:
    /// 创建控件、范围、特殊值和工具提示。
    void setupUi();

    QSpinBox *_cpuThreadsSpin = nullptr;
    QSpinBox *_cudaDeviceSpin = nullptr;
    QSpinBox *_cudaParallelPairsSpin = nullptr;
    QSpinBox *_featurePrefetchDepthSpin = nullptr;
    QSpinBox *_featureMaxImageDimSpin = nullptr;
    QDoubleSpinBox *_matchThresholdSpin = nullptr;
    QDoubleSpinBox *_geometryReprojectionSpin = nullptr;
    QSpinBox *_geometryMinInliersSpin = nullptr;
    QSpinBox *_geometryMaxIterationsSpin = nullptr;
    QSpinBox *_tiePointGridColumnsSpin = nullptr;
    QSpinBox *_tiePointGridRowsSpin = nullptr;
    QSpinBox *_tiePointGridCellLimitSpin = nullptr;
    QDoubleSpinBox *_stationaryMotionSpin = nullptr;
};

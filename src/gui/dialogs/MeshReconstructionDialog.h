#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;
class QPushButton;

/**
 * @brief 网格重建对话框。
 *
 * 从密集点云重建三角网格 (Mesh)，支持 Poisson、Delaunay、
 * Ball Pivoting 等方法。
 */
class MeshReconstructionDialog : public QDialog
{
    Q_OBJECT
public:
    explicit MeshReconstructionDialog(QWidget *parent = nullptr);
    void applySettings(const QJsonObject &settings);
    void setDenseCloudCandidates(const QStringList &paths);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void emitSettingsNow();
    void onRun();

private:
    QJsonObject collectSettings() const;

    QComboBox *_denseCloudCombo = nullptr;     ///< 密集点云输入
    QPushButton *_browseDenseBtn = nullptr;    ///< 浏览点云文件

    QComboBox *_methodCombo = nullptr;         ///< 重建方法
    QComboBox *_outputFormatCombo = nullptr;   ///< 最终输出格式
    QComboBox *_qualityProfileCombo = nullptr; ///< 网格质量档位
    QSpinBox *_octreeDepthSpin = nullptr;      ///< Poisson 八叉树深度
    QDoubleSpinBox *_meshResSpin = nullptr;    ///< 网格分辨率
    QSpinBox *_smoothIterSpin = nullptr;       ///< 平滑迭代次数
    QCheckBox *_holeFillCheck = nullptr;       ///< 补洞
    QDoubleSpinBox *_maxHoleSizeSpin = nullptr; ///< 最大补洞面积
    QCheckBox *_cleanCheck = nullptr;          ///< 清除小连通体
    QSpinBox *_minFacesSpin = nullptr;         ///< 最少面数阈值
    QComboBox *_voxelDensityCombo = nullptr;   ///< 体素重建面片密度
    QCheckBox *_decimateCheck = nullptr;       ///< 简化
    QDoubleSpinBox *_decimateRatioSpin = nullptr; ///< 简化比例
    QSpinBox *_threadsSpin = nullptr;
    QLabel *_infoLabel = nullptr;
};

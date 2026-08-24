// =============================================================================
// 文件: MatchViewerDialog.h
// 功能: 特征点匹配查看器对话框声明
// 职责:
//   - 使用 DualImageViewer 并排展示两张影像及其匹配连接线
//   - 支持在当前匹配集合中直接切换左右影像
//   - 提供同步缩放、视图控制和稀疏匹配显示选项
//   - 支持通过 project_dialog.json 持久化用户配置（项目级）
// =============================================================================
#pragma once

#include <QDialog>
#include <QString>
#include <QVector>

#include "preparation/MatchResultCatalog.h"

class DialogSettingStore;
class DualImageViewer;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;

class MatchViewerDialog : public QDialog
{
    Q_OBJECT

public:
    struct MatchPairOption
    {
        QString imageA;
        QString imageB;
        QString matchFile;
        QVector<xjw::aerial_triangulation::MatchVariant> variants;
    };

    explicit MatchViewerDialog(const QString &imgA,
                               const QString &imgB,
                               const QString &matchFile,
                               QWidget *parent = nullptr);
    ~MatchViewerDialog() override;

    void setMatchVariants(const QVector<xjw::aerial_triangulation::MatchVariant> &variants,
                          const QString &selectedMatchFile);
    void setAvailablePairs(const QVector<MatchPairOption> &pairs,
                           const QString &selectedImageA,
                           const QString &selectedImageB);
    void setProjectPath(const QString &plascanPath);

private slots:
    void onSyncModeToggled(bool checked);
    void onFitToView();
    void onResetView();
    void onZoomIn();
    void onZoomOut();

    void onLineWidthChanged(double value);
    void onOpacityChanged(int value);
    void onMaxCountChanged(int value);
    void onShowEndPointsToggled(bool checked);
    void onShowOnlyInliersToggled(bool checked);
    void onVariantChanged(int index);
    void onImageSelectionChanged();

    void onMatchDataLoaded(int count);
    void onLoadFailed(const QString &error);
    void onMatchValidityLoaded(int validCount, int invalidCount);
    void updateStatusBar();

private:
    void loadSettings();
    void saveSettings();
    void applyMatchVariant(const xjw::aerial_triangulation::MatchVariant &variant, bool forceReload);
    void applySelectedPair();
    int findPairOption(const QString &imageA, const QString &imageB) const;
    QString counterpartForImage(const QString &imagePath) const;

    DualImageViewer *_viewer = nullptr;
    bool _sparseMatchFileMissing = false;

    QCheckBox *_syncModeChk = nullptr;
    QPushButton *_fitBtn = nullptr;
    QPushButton *_resetBtn = nullptr;
    QPushButton *_zoomInBtn = nullptr;
    QPushButton *_zoomOutBtn = nullptr;
    QComboBox *_leftImageCombo = nullptr;
    QComboBox *_rightImageCombo = nullptr;
    QComboBox *_variantCombo = nullptr;

    QDoubleSpinBox *_lineWidthSpin = nullptr;
    QSlider *_opacitySlider = nullptr;
    QSpinBox *_maxCountSpin = nullptr;
    QCheckBox *_showEndPointsChk = nullptr;
    QCheckBox *_showOnlyInliersChk = nullptr;

    QLabel *_statusLabel = nullptr;

    QString _imageA;
    QString _imageB;
    QString _matchFile;
    int _totalMatches = 0;
    int _validMatches = -1;
    int _invalidMatches = -1;
    QVector<xjw::aerial_triangulation::MatchVariant> _matchVariants;
    QVector<MatchPairOption> _pairOptions;
    QString _currentVariantSummary;
    bool _updatingImageSelectors = false;
    DialogSettingStore *_setting = nullptr;
};

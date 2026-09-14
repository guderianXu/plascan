#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QLabel;
class QPushButton;
class QSpinBox;
class QToolButton;

/**
 * @brief Reference-model-compatible mesh generation dialog.
 *
 * The widget layout mirrors the recovered reference GUI while source
 * candidates and submitted settings remain backed by PlaScan project data.
 */
class GenerateModelDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GenerateModelDialog(QWidget* parent = nullptr);

    void applySettings(const QJsonObject& settings);
    void setSourceCandidates(const QJsonArray& candidates);

signals:
    void runRequested(const QJsonObject& settings);
    void settingsChanged(const QJsonObject& settings);

private slots:
    void emitSettingsNow();
    void onRun();
    void onSourceTypeChanged();

private:
    QJsonObject collectSettings() const;
    QJsonObject currentCandidate() const;
    bool usesRpcHeightPlaneSweep() const;
    void refreshSourceTypes();
    void refreshSourceItems();
    void setAdvancedExpanded(bool expanded);
    void updateAvailability();
    void updateCustomFaceCountVisibility();

    QJsonArray _candidates;
    QString _pendingSourceData;
    QString _pendingSourcePath;
    QString _computeMode = QStringLiteral("hybrid");
    bool _hasReusableDepthMaps = false;
    bool _reuseDepthMapsRequested = true;

    QComboBox* _sourceCombo = nullptr;
    QComboBox* _sourceItemCombo = nullptr;
    QComboBox* _surfaceTypeCombo = nullptr;
    QLabel* _qualityLabel = nullptr;
    QComboBox* _qualityCombo = nullptr;
    QComboBox* _faceCountModeCombo = nullptr;
    QLabel* _customFaceCountLabel = nullptr;
    QSpinBox* _customFaceCountSpin = nullptr;
    QLabel* _rpcHeightMinLabel = nullptr;
    QLabel* _rpcHeightMaxLabel = nullptr;
    QDoubleSpinBox* _rpcHeightMinSpin = nullptr;
    QDoubleSpinBox* _rpcHeightMaxSpin = nullptr;
    QCheckBox* _saveAfterEachStepCheck = nullptr;

    QCheckBox* _splitInBlocksCheck = nullptr;
    QComboBox* _coordinateSystemCombo = nullptr;
    QDoubleSpinBox* _blockSizeSpin = nullptr;
    QDoubleSpinBox* _blockOriginXSpin = nullptr;
    QDoubleSpinBox* _blockOriginYSpin = nullptr;
    QCheckBox* _skipBoundaryBlocksCheck = nullptr;
    QPushButton* _blocksPreviewButton = nullptr;
    bool _exportCompletedBlocks = false;
    bool _generatePreviewTextures = true;
    QString _blockOutputFolder;

    QFrame* _advancedContent = nullptr;
    QToolButton* _advancedToggle = nullptr;
    QComboBox* _interpolationCombo = nullptr;
    QLabel* _depthFilteringLabel = nullptr;
    QComboBox* _depthFilteringCombo = nullptr;
    QLabel* _pointClassesLabel = nullptr;
    QPushButton* _selectPointClassesButton = nullptr;
    QCheckBox* _vertexColorsCheck = nullptr;
    QCheckBox* _strictVolumetricMasksCheck = nullptr;
    QCheckBox* _reuseDepthMapsCheck = nullptr;
    QCheckBox* _replaceDefaultCheck = nullptr;
    QPushButton* _okButton = nullptr;
};

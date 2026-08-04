#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>
#include <QMetaObject>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QScrollArea;
class QScreen;
class QShowEvent;
class QToolButton;
class QWidget;

/**
 * @brief Metashape-style model generation dialog.
 *
 * The dialog exposes source-data choices based on project artifacts, then maps
 * supported point-like sources to the existing mesh reconstruction pipeline.
 */
class GenerateModelDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GenerateModelDialog(QWidget *parent = nullptr);

    void applySettings(const QJsonObject &settings);
    void setSourceCandidates(const QJsonArray &candidates);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

protected:
    void showEvent(QShowEvent *event) override;

private slots:
    void emitSettingsNow();
    void onRun();
    void onSourceTypeChanged();

private:
    QJsonObject collectSettings() const;
    QJsonObject currentCandidate() const;
    void refreshSourceTypes();
    void refreshSourceItems();
    void setAdvancedExpanded(bool expanded);
    void bindScreenGeometryUpdates(QScreen *targetScreen);
    void refreshScrollableDialogSize();
    void updateAvailability();
    void updateBlockControlsAvailability();
    void updateScrollableContentHeight();

    QJsonArray _candidates;
    QString _pendingSourceData;
    QString _pendingSourcePath;
    bool _hasReusableDepthMaps = false;
    bool _reuseDepthMapsRequested = true;

    QComboBox *_sourceCombo = nullptr;
    QComboBox *_sourceItemCombo = nullptr;
    QComboBox *_surfaceTypeCombo = nullptr;
    QComboBox *_qualityCombo = nullptr;
    QLabel *_effectiveDepthQualityLabel = nullptr;
    QComboBox *_faceCountCombo = nullptr;
    QCheckBox *_saveEachStepCheck = nullptr;
    QCheckBox *_splitRegionCheck = nullptr;
    QLabel *_coordinateLabel = nullptr;
    QDoubleSpinBox *_blockSizeSpin = nullptr;
    QLabel *_originLabel = nullptr;
    QCheckBox *_skipBoundaryBlocksCheck = nullptr;
    QScrollArea *_contentScrollArea = nullptr;
    QMetaObject::Connection _screenGeometryConnection;
    bool _screenChangeConnected = false;
    QToolButton *_advancedToggle = nullptr;
    QWidget *_advancedContent = nullptr;
    QComboBox *_interpolationCombo = nullptr;
    QComboBox *_depthFilterCombo = nullptr;
    QCheckBox *_calculateColorsCheck = nullptr;
    QCheckBox *_strictMasksCheck = nullptr;
    QCheckBox *_reuseDepthMapsCheck = nullptr;
    QCheckBox *_replaceDefaultCheck = nullptr;
    QPushButton *_okButton = nullptr;
};

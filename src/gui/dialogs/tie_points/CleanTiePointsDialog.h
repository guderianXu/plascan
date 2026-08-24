#pragma once

#include <QDialog>
#include <QHash>
#include <QString>

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;

class CleanTiePointsDialog : public QDialog
{
    Q_OBJECT

public:
    enum class Criterion
    {
        None,
        ReprojectionError,
        ReconstructionUncertainty,
        ImageCount,
        ProjectionAccuracy,
        MinimumTriangulationAngle
    };
    Q_ENUM(Criterion)

    struct CriterionConfiguration
    {
        bool available = true;
        double minimum = 0.0;
        double maximum = 1.0;
        double defaultLevel = 0.5;
        double singleStep = 0.01;
        int decimals = 3;
        QString unavailableReason;
    };

    explicit CleanTiePointsDialog(QWidget *parent = nullptr);

    Criterion criterion() const;
    void setCriterion(Criterion criterion);
    QString criterionText() const;
    double level() const;
    bool deleteRequested() const;
    bool hasStagedDeletion(Criterion criterion) const;
    double stagedLevel(Criterion criterion) const;
    int stagedDeletionCount() const;
    int remainingPointCount() const;
    CriterionConfiguration criterionConfiguration(Criterion criterion) const;
    void setCriterionConfiguration(Criterion criterion,
                                   const CriterionConfiguration &configuration);
    void setCandidateCount(int candidateCount, int totalCount);
    void confirmStagedDeletion(Criterion criterion,
                               double level,
                               int stagedDeletionCount,
                               int remainingPointCount);
    int candidateCount() const;
    int totalPointCount() const;

signals:
    void previewRequested(CleanTiePointsDialog::Criterion criterion, double level);
    void previewCleared();
    void stageDeleteRequested(CleanTiePointsDialog::Criterion criterion, double level);

private:
    void initializeCriterionConfigurations();
    void buildUi();
    void updateCriterionState();
    void updateCriterionItem(Criterion criterion);
    void updateAvailabilityMessage(int highlightedIndex = -1);
    void applyCurrentCriterionConfiguration(bool useDefaultLevel);
    void synchronizeLevelFromSlider(int position);
    void synchronizeSliderFromLevel(double value);
    void requestPreview();
    void invalidateCandidateCount();
    void updateCandidateState();
    int sliderPositionForValue(double value) const;
    double valueForSliderPosition(int position) const;
    QString formattedLevel(double value) const;

    QComboBox *_criterionCombo{};
    QDoubleSpinBox *_levelSpin{};
    QSlider *_levelSlider{};
    QLabel *_minimumLabel{};
    QLabel *_maximumLabel{};
    QLabel *_availabilityLabel{};
    QLabel *_candidateCountLabel{};
    QPushButton *_okButton{};
    QPushButton *_deleteButton{};
    QHash<int, CriterionConfiguration> _criterionConfigurations;
    QHash<int, double> _stagedLevels;
    int _candidateCount = -1;
    int _totalPointCount = -1;
    int _stagedDeletionCount = 0;
    int _remainingPointCount = -1;
};

#pragma once

#include <QDialog>

class QComboBox;
class QLineEdit;
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
        ProjectionAccuracy
    };

    explicit CleanTiePointsDialog(QWidget *parent = nullptr);

    Criterion criterion() const;
    QString criterionText() const;
    double level() const;
    bool deleteRequested() const;

private:
    void buildUi();
    void updateCriterionState();

    QComboBox *_criterionCombo{};
    QLineEdit *_levelEdit{};
    QSlider *_levelSlider{};
    QPushButton *_okButton{};
    QPushButton *_deleteButton{};
    QPushButton *_cancelButton{};
    bool _deleteRequested = false;
};

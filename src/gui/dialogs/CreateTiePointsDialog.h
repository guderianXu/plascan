#pragma once

#include <QDialog>

class QCheckBox;
class QComboBox;
class QLineEdit;

class CreateTiePointsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CreateTiePointsDialog(QWidget *parent = nullptr);

    QString accuracy() const;
    int keypointLimit() const;
    int tiePointLimit() const;
    bool useGenericPreselection() const;
    bool useReferencePreselection() const;
    bool useGuidedMatching() const;
    bool excludePinnedTiePoints() const;

private:
    int intFromEdit(const QLineEdit *edit, int fallback) const;
    void buildUi();

    QComboBox *_accuracyCombo{};
    QCheckBox *_genericPreselectionCheck{};
    QCheckBox *_referencePreselectionCheck{};
    QLineEdit *_keypointLimitEdit{};
    QLineEdit *_tiePointLimitEdit{};
    QComboBox *_maskModeCombo{};
    QCheckBox *_guidedMatchingCheck{};
    QCheckBox *_excludePinnedTiePointsCheck{};
};

#pragma once

#include <QDialog>

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QToolButton;

class CreateTiePointsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CreateTiePointsDialog(QWidget *parent = nullptr);

    QString accuracy() const;
    int keypointLimit() const;
    int keypointLimitPerMegapixel() const;
    int tiePointLimit() const;
    bool useGenericPreselection() const;
    bool useReferencePreselection() const;
    bool useGuidedMatching() const;
    bool excludePinnedTiePoints() const;
    QString maskApplyMode() const;
    void setReferencePreselectionAvailable(bool available,
                                           int cameraCount = 0,
                                           int imageCount = 0);

private:
    int intFromEdit(const QLineEdit *edit, int fallback) const;
    QString formattedInteger(int value) const;
    void buildUi();
    void setAdvancedExpanded(bool expanded);
    void updateKeypointLimitMode(bool guided);

    QComboBox *_accuracyCombo{};
    QCheckBox *_genericPreselectionCheck{};
    QCheckBox *_referencePreselectionCheck{};
    QLabel *_keypointLimitLabel{};
    QLineEdit *_keypointLimitEdit{};
    QLineEdit *_tiePointLimitEdit{};
    QComboBox *_maskModeCombo{};
    QCheckBox *_guidedMatchingCheck{};
    QCheckBox *_excludePinnedTiePointsCheck{};
    QGroupBox *_generalGroup{};
    QGroupBox *_advancedGroup{};
    QToolButton *_advancedToggle{};
    QWidget *_advancedContent{};
    int _keypointLimit = 40000;
    int _keypointLimitPerMegapixel = 1000;
};

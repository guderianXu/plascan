#include "tie_points/CleanTiePointsDialog.h"

#include "shared/WorkflowParameterDialogStyle.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QStandardItemModel>
#include <QStringList>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>

namespace
{

constexpr int kSliderSteps = 1000;

} // namespace

CleanTiePointsDialog::CleanTiePointsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("清理连接点"));
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);
    initializeCriterionConfigurations();
    buildUi();
    updateCriterionState();
}

CleanTiePointsDialog::Criterion CleanTiePointsDialog::criterion() const
{
    if (!_criterionCombo)
    {
        return Criterion::None;
    }

    return static_cast<Criterion>(_criterionCombo->currentData().toInt());
}

void CleanTiePointsDialog::setCriterion(Criterion criterion)
{
    if (!_criterionCombo)
    {
        return;
    }

    const int index = _criterionCombo->findData(static_cast<int>(criterion));
    if (index < 0)
    {
        return;
    }

    if (criterion != Criterion::None && !criterionConfiguration(criterion).available)
    {
        updateAvailabilityMessage(index);
        return;
    }

    if (_criterionCombo->currentIndex() == index)
    {
        updateCriterionState();
        return;
    }
    _criterionCombo->setCurrentIndex(index);
}

QString CleanTiePointsDialog::criterionText() const
{
    return _criterionCombo ? _criterionCombo->currentText() : QString();
}

double CleanTiePointsDialog::level() const
{
    const Criterion selected = criterion();
    return _levelSpin && selected != Criterion::None
            && criterionConfiguration(selected).available
        ? _levelSpin->value()
        : 0.0;
}

bool CleanTiePointsDialog::deleteRequested() const
{
    return _stagedDeletionCount > 0;
}

bool CleanTiePointsDialog::hasStagedDeletion(Criterion criterion) const
{
    return _stagedLevels.contains(static_cast<int>(criterion));
}

double CleanTiePointsDialog::stagedLevel(Criterion criterion) const
{
    return _stagedLevels.value(static_cast<int>(criterion), 0.0);
}

int CleanTiePointsDialog::stagedDeletionCount() const
{
    return _stagedDeletionCount;
}

int CleanTiePointsDialog::remainingPointCount() const
{
    return _remainingPointCount;
}

CleanTiePointsDialog::CriterionConfiguration CleanTiePointsDialog::criterionConfiguration(
    Criterion criterion) const
{
    return _criterionConfigurations.value(static_cast<int>(criterion));
}

void CleanTiePointsDialog::setCriterionConfiguration(
    Criterion criterion,
    const CriterionConfiguration &configuration)
{
    if (criterion == Criterion::None)
    {
        return;
    }

    CriterionConfiguration normalized = configuration;
    if (!std::isfinite(normalized.minimum))
    {
        normalized.minimum = 0.0;
    }
    if (!std::isfinite(normalized.maximum) || normalized.maximum <= normalized.minimum)
    {
        normalized.maximum = normalized.minimum + 1.0;
    }
    if (!std::isfinite(normalized.defaultLevel))
    {
        normalized.defaultLevel = normalized.minimum;
    }
    normalized.defaultLevel = std::clamp(normalized.defaultLevel,
                                         normalized.minimum,
                                         normalized.maximum);
    if (!std::isfinite(normalized.singleStep) || normalized.singleStep <= 0.0)
    {
        normalized.singleStep = (normalized.maximum - normalized.minimum) / 100.0;
    }
    normalized.decimals = std::clamp(normalized.decimals, 0, 8);
    if (!normalized.available && normalized.unavailableReason.trimmed().isEmpty())
    {
        normalized.unavailableReason = tr("当前连接点结果未提供该逐点质量指标。");
    }

    _criterionConfigurations.insert(static_cast<int>(criterion), normalized);
    updateCriterionItem(criterion);
    updateAvailabilityMessage();

    if (this->criterion() == criterion)
    {
        if (normalized.available)
        {
            applyCurrentCriterionConfiguration(false);
        }
        else
        {
            updateCriterionState();
        }
    }
}

void CleanTiePointsDialog::setCandidateCount(int candidateCount, int totalCount)
{
    if (candidateCount < 0 || totalCount < 0)
    {
        invalidateCandidateCount();
        return;
    }

    _totalPointCount = totalCount;
    _candidateCount = std::clamp(candidateCount, 0, totalCount);
    updateCandidateState();
}

void CleanTiePointsDialog::confirmStagedDeletion(Criterion criterion,
                                                 double level,
                                                 int stagedDeletionCount,
                                                 int remainingPointCount)
{
    if (criterion == Criterion::None || stagedDeletionCount <= 0
        || remainingPointCount <= 0 || !std::isfinite(level))
    {
        return;
    }

    const int key = static_cast<int>(criterion);
    if (_stagedLevels.contains(key))
    {
        const double previous_level = _stagedLevels.value(key);
        level = criterion == Criterion::ReprojectionError
            ? std::min(previous_level, level)
            : std::max(previous_level, level);
    }
    _stagedLevels.insert(key, level);
    _stagedDeletionCount = stagedDeletionCount;
    _remainingPointCount = remainingPointCount;
    _candidateCount = 0;
    _totalPointCount = remainingPointCount;
    updateCandidateState();
}

int CleanTiePointsDialog::candidateCount() const
{
    return _candidateCount;
}

int CleanTiePointsDialog::totalPointCount() const
{
    return _totalPointCount;
}

void CleanTiePointsDialog::initializeCriterionConfigurations()
{
    _criterionConfigurations.insert(static_cast<int>(Criterion::ReprojectionError),
                                    CriterionConfiguration{
                                        true, 0.0, 10.0, 1.0, 0.01, 3, QString()});
    _criterionConfigurations.insert(static_cast<int>(Criterion::ReconstructionUncertainty),
                                    CriterionConfiguration{
                                        false,
                                        0.0,
                                        100.0,
                                        10.0,
                                        0.1,
                                        2,
                                        tr("当前连接点结果尚未生成重建不确定度指标。")});
    _criterionConfigurations.insert(static_cast<int>(Criterion::ImageCount),
                                    CriterionConfiguration{
                                        true, 2.0, 100.0, 3.0, 1.0, 0, QString()});
    _criterionConfigurations.insert(static_cast<int>(Criterion::ProjectionAccuracy),
                                    CriterionConfiguration{
                                        false,
                                        0.0,
                                        100.0,
                                        1.0,
                                        0.1,
                                        2,
                                        tr("当前连接点结果尚未生成投影精度指标。")});
    _criterionConfigurations.insert(static_cast<int>(Criterion::MinimumTriangulationAngle),
                                    CriterionConfiguration{
                                        true, 0.0, 180.0, 2.0, 0.1, 2, QString()});
}

void CleanTiePointsDialog::buildUi()
{
    auto *mainLayout = new QVBoxLayout(this);
    xjw::gui::dialogs::configureWorkflowDialogLayout(mainLayout);

    auto *generalGroup = new QGroupBox(tr("一般"), this);
    generalGroup->setObjectName(QStringLiteral("cleanTiePointsGeneralGroup"));
    auto *formLayout = new QFormLayout(generalGroup);
    xjw::gui::dialogs::configureWorkflowForm(formLayout);

    _criterionCombo = new QComboBox(generalGroup);
    _criterionCombo->setObjectName(QStringLiteral("cleanTiePointsCriterionCombo"));
    _criterionCombo->addItem(tr("请选择..."), static_cast<int>(Criterion::None));
    _criterionCombo->addItem(tr("重投影误差"), static_cast<int>(Criterion::ReprojectionError));
    _criterionCombo->addItem(tr("重建不确定度"), static_cast<int>(Criterion::ReconstructionUncertainty));
    _criterionCombo->addItem(tr("图像计数"), static_cast<int>(Criterion::ImageCount));
    _criterionCombo->addItem(tr("投影精度"), static_cast<int>(Criterion::ProjectionAccuracy));
    _criterionCombo->addItem(tr("最小交会角"), static_cast<int>(Criterion::MinimumTriangulationAngle));
    xjw::gui::dialogs::configureWorkflowComboBox(_criterionCombo);
    formLayout->addRow(tr("标准:"), _criterionCombo);

    auto *levelWidget = new QWidget(generalGroup);
    levelWidget->setObjectName(QStringLiteral("cleanTiePointsLevelWidget"));
    auto *levelLayout = new QVBoxLayout(levelWidget);
    levelLayout->setContentsMargins(0, 0, 0, 0);
    levelLayout->setSpacing(5);

    _levelSpin = new QDoubleSpinBox(levelWidget);
    _levelSpin->setObjectName(QStringLiteral("cleanTiePointsLevelSpin"));
    _levelSpin->setKeyboardTracking(true);
    xjw::gui::dialogs::configureWorkflowInputWidget(_levelSpin);
    levelLayout->addWidget(_levelSpin);

    auto *sliderLayout = new QHBoxLayout();
    sliderLayout->setContentsMargins(0, 0, 0, 0);
    sliderLayout->setSpacing(8);
    _minimumLabel = new QLabel(levelWidget);
    _minimumLabel->setObjectName(QStringLiteral("cleanTiePointsMinimumLabel"));
    _minimumLabel->setMinimumWidth(42);
    _minimumLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    _levelSlider = new QSlider(Qt::Horizontal, levelWidget);
    _levelSlider->setObjectName(QStringLiteral("cleanTiePointsLevelSlider"));
    _levelSlider->setRange(0, kSliderSteps);
    _levelSlider->setPageStep(50);
    _levelSlider->setTracking(true);
    _maximumLabel = new QLabel(levelWidget);
    _maximumLabel->setObjectName(QStringLiteral("cleanTiePointsMaximumLabel"));
    _maximumLabel->setMinimumWidth(42);
    _maximumLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    sliderLayout->addWidget(_minimumLabel);
    sliderLayout->addWidget(_levelSlider, 1);
    sliderLayout->addWidget(_maximumLabel);
    levelLayout->addLayout(sliderLayout);
    formLayout->addRow(tr("级别:"), levelWidget);

    _availabilityLabel = new QLabel(generalGroup);
    _availabilityLabel->setObjectName(QStringLiteral("cleanTiePointsAvailabilityLabel"));
    _availabilityLabel->setWordWrap(true);
    formLayout->addRow(QString(), _availabilityLabel);

    _candidateCountLabel = new QLabel(generalGroup);
    _candidateCountLabel->setObjectName(QStringLiteral("cleanTiePointsCandidateCountLabel"));
    _candidateCountLabel->setWordWrap(true);
    formLayout->addRow(QString(), _candidateCountLabel);
    mainLayout->addWidget(generalGroup);

    auto *buttonBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttonBox->setObjectName(QStringLiteral("workflowButtonBox"));
    _okButton = buttonBox->button(QDialogButtonBox::Ok);
    _deleteButton = buttonBox->addButton(tr("删除"), QDialogButtonBox::DestructiveRole);
    _deleteButton->setObjectName(QStringLiteral("cleanTiePointsDeleteButton"));
    xjw::gui::dialogs::configureWorkflowButtonBox(buttonBox);
    mainLayout->addWidget(buttonBox);

    for (Criterion criterion : {
             Criterion::ReprojectionError,
             Criterion::ReconstructionUncertainty,
             Criterion::ImageCount,
             Criterion::ProjectionAccuracy,
             Criterion::MinimumTriangulationAngle})
    {
        updateCriterionItem(criterion);
    }

    connect(_criterionCombo,
            &QComboBox::currentIndexChanged,
            this,
            &CleanTiePointsDialog::updateCriterionState);
    connect(_criterionCombo,
            &QComboBox::highlighted,
            this,
            &CleanTiePointsDialog::updateAvailabilityMessage);
    connect(_levelSlider,
            &QSlider::valueChanged,
            this,
            &CleanTiePointsDialog::synchronizeLevelFromSlider);
    connect(_levelSpin,
            &QDoubleSpinBox::valueChanged,
            this,
            &CleanTiePointsDialog::synchronizeSliderFromLevel);
    connect(_okButton, &QPushButton::clicked, this, [this]()
    {
        accept();
    });
    connect(_deleteButton, &QPushButton::clicked, this, [this]()
    {
        emit stageDeleteRequested(criterion(), level());
    });
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(this, &QDialog::rejected, this, &CleanTiePointsDialog::previewCleared);
}

void CleanTiePointsDialog::updateCriterionState()
{
    const int selectedIndex = _criterionCombo ? _criterionCombo->currentIndex() : -1;
    const Criterion selected = criterion();
    if (selected != Criterion::None && !criterionConfiguration(selected).available)
    {
        if (_criterionCombo)
        {
            const QSignalBlocker blocker(_criterionCombo);
            _criterionCombo->setCurrentIndex(
                _criterionCombo->findData(static_cast<int>(Criterion::None)));
        }
        applyCurrentCriterionConfiguration(true);
        updateAvailabilityMessage(selectedIndex);
        return;
    }

    applyCurrentCriterionConfiguration(true);
    updateAvailabilityMessage(_criterionCombo ? _criterionCombo->currentIndex() : -1);
}

void CleanTiePointsDialog::updateCriterionItem(Criterion criterion)
{
    if (!_criterionCombo)
    {
        return;
    }

    const int index = _criterionCombo->findData(static_cast<int>(criterion));
    if (index < 0)
    {
        return;
    }

    const CriterionConfiguration configuration = criterionConfiguration(criterion);
    _criterionCombo->setItemData(index,
                                 configuration.unavailableReason,
                                 Qt::ToolTipRole);
    auto *model = qobject_cast<QStandardItemModel *>(_criterionCombo->model());
    if (model && model->item(index))
    {
        model->item(index)->setEnabled(configuration.available);
    }
}

void CleanTiePointsDialog::updateAvailabilityMessage(int highlightedIndex)
{
    if (!_availabilityLabel || !_criterionCombo)
    {
        return;
    }

    if (highlightedIndex >= 0 && highlightedIndex < _criterionCombo->count())
    {
        const Criterion highlighted = static_cast<Criterion>(
            _criterionCombo->itemData(highlightedIndex).toInt());
        const CriterionConfiguration configuration = criterionConfiguration(highlighted);
        if (highlighted != Criterion::None && !configuration.available)
        {
            _availabilityLabel->setText(
                tr("%1不可用：%2")
                    .arg(_criterionCombo->itemText(highlightedIndex),
                         configuration.unavailableReason));
            _availabilityLabel->setVisible(true);
            return;
        }
    }

    QStringList unavailable;
    for (int index = 1; index < _criterionCombo->count(); ++index)
    {
        const Criterion itemCriterion = static_cast<Criterion>(
            _criterionCombo->itemData(index).toInt());
        const CriterionConfiguration configuration = criterionConfiguration(itemCriterion);
        if (!configuration.available)
        {
            unavailable.append(tr("%1：%2")
                                   .arg(_criterionCombo->itemText(index),
                                        configuration.unavailableReason));
        }
    }
    _availabilityLabel->setText(unavailable.join(QLatin1Char('\n')));
    _availabilityLabel->setVisible(!unavailable.isEmpty());
}

void CleanTiePointsDialog::applyCurrentCriterionConfiguration(bool useDefaultLevel)
{
    const Criterion selected = criterion();
    const CriterionConfiguration configuration = criterionConfiguration(selected);
    const bool available = selected != Criterion::None && configuration.available;

    if (_levelSpin && _levelSlider)
    {
        const QSignalBlocker spinBlocker(_levelSpin);
        const QSignalBlocker sliderBlocker(_levelSlider);
        _levelSpin->setEnabled(available);
        _levelSlider->setEnabled(available);

        if (available)
        {
            const double requestedLevel = useDefaultLevel
                ? configuration.defaultLevel
                : _levelSpin->value();
            _levelSpin->setDecimals(configuration.decimals);
            _levelSpin->setRange(configuration.minimum, configuration.maximum);
            _levelSpin->setSingleStep(configuration.singleStep);
            _levelSpin->setValue(std::clamp(requestedLevel,
                                            configuration.minimum,
                                            configuration.maximum));
            _levelSlider->setValue(sliderPositionForValue(_levelSpin->value()));
        }
        else
        {
            _levelSpin->clear();
            _levelSlider->setValue(0);
        }
    }

    if (_minimumLabel)
    {
        _minimumLabel->setText(available ? formattedLevel(configuration.minimum) : QString());
        _minimumLabel->setEnabled(available);
    }
    if (_maximumLabel)
    {
        _maximumLabel->setText(available ? formattedLevel(configuration.maximum) : QString());
        _maximumLabel->setEnabled(available);
    }
    if (_okButton)
    {
        _okButton->setEnabled(available);
    }

    invalidateCandidateCount();
    if (available)
    {
        emit previewRequested(selected, level());
    }
    else
    {
        emit previewCleared();
    }
}

void CleanTiePointsDialog::synchronizeLevelFromSlider(int position)
{
    if (!_levelSpin || !_levelSpin->isEnabled())
    {
        return;
    }

    {
        const QSignalBlocker blocker(_levelSpin);
        _levelSpin->setValue(valueForSliderPosition(position));
    }
    requestPreview();
}

void CleanTiePointsDialog::synchronizeSliderFromLevel(double value)
{
    if (!_levelSlider || !_levelSlider->isEnabled())
    {
        return;
    }

    {
        const QSignalBlocker blocker(_levelSlider);
        _levelSlider->setValue(sliderPositionForValue(value));
    }
    requestPreview();
}

void CleanTiePointsDialog::requestPreview()
{
    const Criterion selected = criterion();
    if (selected == Criterion::None || !criterionConfiguration(selected).available)
    {
        emit previewCleared();
        return;
    }

    invalidateCandidateCount();
    emit previewRequested(selected, level());
}

void CleanTiePointsDialog::invalidateCandidateCount()
{
    _candidateCount = -1;
    _totalPointCount = -1;
    updateCandidateState();
}

void CleanTiePointsDialog::updateCandidateState()
{
    if (!_candidateCountLabel || !_deleteButton || !_okButton)
    {
        return;
    }

    const Criterion selected = criterion();
    const bool available = selected != Criterion::None
        && criterionConfiguration(selected).available;
    if (!available)
    {
        _candidateCountLabel->setText(tr("请选择可用的清理标准。"));
        _okButton->setEnabled(_stagedDeletionCount > 0);
        _deleteButton->setEnabled(false);
        _deleteButton->setToolTip(tr("请先选择可用的清理标准。"));
        return;
    }

    if (_candidateCount < 0 || _totalPointCount < 0)
    {
        _candidateCountLabel->setText(_stagedDeletionCount > 0
            ? tr("已暂删：%1；剩余：%2；正在计算新的候选点...")
                  .arg(_stagedDeletionCount)
                  .arg(_remainingPointCount)
            : tr("候选点：正在等待预览结果..."));
        _okButton->setEnabled(_stagedDeletionCount > 0);
        _deleteButton->setEnabled(false);
        _deleteButton->setToolTip(tr("候选点计算完成后才能删除。"));
        return;
    }

    _candidateCountLabel->setText(_stagedDeletionCount > 0
        ? tr("已暂删：%1；剩余：%2；当前候选：%3")
              .arg(_stagedDeletionCount)
              .arg(_remainingPointCount)
              .arg(_candidateCount)
        : tr("候选点：%1 / %2").arg(_candidateCount).arg(_totalPointCount));
    _okButton->setEnabled(_stagedDeletionCount > 0);
    if (_candidateCount <= 0)
    {
        _deleteButton->setEnabled(false);
        _deleteButton->setToolTip(tr("当前阈值没有可删除的候选点。"));
        return;
    }
    if (_totalPointCount <= 0 || _candidateCount >= _totalPointCount)
    {
        _deleteButton->setEnabled(false);
        _deleteButton->setToolTip(tr("当前阈值会删除全部连接点，请调整阈值。"));
        return;
    }

    _deleteButton->setEnabled(true);
    _deleteButton->setToolTip(tr("暂时隐藏当前高亮候选点；确定后才会应用。"));
}

int CleanTiePointsDialog::sliderPositionForValue(double value) const
{
    const CriterionConfiguration configuration = criterionConfiguration(criterion());
    const double span = configuration.maximum - configuration.minimum;
    if (!(span > 0.0))
    {
        return 0;
    }
    const double normalized = std::clamp((value - configuration.minimum) / span,
                                         0.0,
                                         1.0);
    return static_cast<int>(std::lround(normalized * kSliderSteps));
}

double CleanTiePointsDialog::valueForSliderPosition(int position) const
{
    const CriterionConfiguration configuration = criterionConfiguration(criterion());
    const double normalized = std::clamp(static_cast<double>(position) / kSliderSteps,
                                         0.0,
                                         1.0);
    return configuration.minimum
        + normalized * (configuration.maximum - configuration.minimum);
}

QString CleanTiePointsDialog::formattedLevel(double value) const
{
    const CriterionConfiguration configuration = criterionConfiguration(criterion());
    return QString::number(value, 'f', configuration.decimals);
}

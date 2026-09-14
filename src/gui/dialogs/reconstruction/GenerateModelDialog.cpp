#include "reconstruction/GenerateModelDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

#include <cmath>
#include <utility>

namespace
{

    constexpr const char* kSourceData = "source_data";
    constexpr const char* kSourceLabel = "source_label";
    constexpr const char* kSourcePath = "source_path";
    constexpr const char* kDisplay = "display";
    constexpr const char* kSupported = "supported";
    constexpr const char* kNote = "note";
    constexpr const char* kAutomaticDepthMaps = "automatic_depth_maps";
    constexpr const char* kDepthBatchCompatible = "depth_batch_compatible";
    constexpr const char* kDepthBatchCompatibilityReason = "depth_batch_compatibility_reason";
    constexpr const char* kComputeMode = "compute_mode";

    class DialogTitleBar final : public QWidget
    {
    public:
        explicit DialogTitleBar(QDialog* dialog) : QWidget(dialog), _dialog(dialog)
        {
        }

    protected:
        void mousePressEvent(QMouseEvent* event) override
        {
            if (event->button() == Qt::LeftButton)
            {
                _dragOffset = event->globalPosition().toPoint() - _dialog->frameGeometry().topLeft();
                event->accept();
            }
        }

        void mouseMoveEvent(QMouseEvent* event) override
        {
            if (event->buttons().testFlag(Qt::LeftButton))
            {
                _dialog->move(event->globalPosition().toPoint() - _dragOffset);
                event->accept();
            }
        }

    private:
        QDialog* _dialog = nullptr;
        QPoint _dragOffset;
    };

    QComboBox* makeCombo(QWidget* parent, const char* objectName)
    {
        auto* combo = new QComboBox(parent);
        combo->setObjectName(QString::fromLatin1(objectName));
        combo->setMinimumContentsLength(10);
        combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        return combo;
    }

    QFrame* makeSection(QWidget* parent,
                        const QString& title,
                        const QString& sectionObjectName,
                        QToolButton** toggleOutput,
                        bool expanded)
    {
        auto* section = new QFrame(parent);
        section->setObjectName(sectionObjectName);
        section->setProperty("modelDialogSection", true);
        auto* layout = new QVBoxLayout(section);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);

        auto* toggle = new QToolButton(section);
        toggle->setObjectName(title == QStringLiteral("一般") ? QStringLiteral("GeneralHeader")
                                                              : QStringLiteral("SectionHeader"));
        toggle->setProperty("modelSectionHeader", true);
        toggle->setText(title);
        toggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        toggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
        toggle->setCheckable(true);
        toggle->setChecked(expanded);
        toggle->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        layout->addWidget(toggle);

        auto* body = new QFrame(section);
        body->setObjectName(QStringLiteral("SectionBody"));
        body->setProperty("modelSectionBody", true);
        body->setVisible(expanded);
        layout->addWidget(body);

        QObject::connect(toggle,
                         &QToolButton::toggled,
                         body,
                         [toggle, body](bool checked)
                         {
                             toggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
                             body->setVisible(checked);
                             if (auto* dialog = qobject_cast<QDialog*>(toggle->window()))
                             {
                                 dialog->adjustSize();
                             }
                         });

        *toggleOutput = toggle;
        return body;
    }

    QFormLayout* makeForm(QFrame* body)
    {
        auto* form = new QFormLayout(body);
        form->setContentsMargins(12, 10, 12, 10);
        form->setHorizontalSpacing(12);
        form->setVerticalSpacing(6);
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        return form;
    }

    void setLabelColumnWidth(QFormLayout* form, int width)
    {
        for (int row = 0; row < form->rowCount(); ++row)
        {
            if (QLayoutItem* item = form->itemAt(row, QFormLayout::LabelRole))
            {
                if (QWidget* label = item->widget())
                {
                    label->setMinimumWidth(width);
                }
            }
        }
    }

    QString defaultSourceLabel(const QString& sourceData)
    {
        if (sourceData == QStringLiteral("tie_points"))
        {
            return QStringLiteral("连接点");
        }
        if (sourceData == QStringLiteral("depth_maps"))
        {
            return QStringLiteral("深度图");
        }
        if (sourceData == QStringLiteral("point_cloud"))
        {
            return QStringLiteral("点云");
        }
        if (sourceData == QStringLiteral("model"))
        {
            return QStringLiteral("模型");
        }
        if (sourceData == QStringLiteral("rpc_height_plane_sweep"))
        {
            return QStringLiteral("RPC 高程平面扫描");
        }
        return sourceData;
    }

    bool hasValidRpcImagePair(const QJsonObject& candidate)
    {
        const QJsonArray paths = candidate.value(QStringLiteral("rpcImagePaths")).toArray();
        return paths.size() == 2 && paths.at(0).isString() && paths.at(1).isString() &&
               !paths.at(0).toString().trimmed().isEmpty() && !paths.at(1).toString().trimmed().isEmpty();
    }

    class BuildModelPreviewDialog final : public QDialog
    {
    public:
        BuildModelPreviewDialog(bool exportBlocks, bool buildTexture, QString outputFolder, QWidget* parent)
            : QDialog(parent)
        {
            setObjectName(QStringLiteral("BuildModelPreviewDialog"));
            setWindowTitle(QStringLiteral("区块模型预览"));
            setModal(true);
            setFixedWidth(371);

            auto* root = new QVBoxLayout(this);
            root->setContentsMargins(12, 10, 12, 10);
            root->setSpacing(6);

            _exportBlocks = new QCheckBox(QStringLiteral("导出已完成的块"), this);
            _exportBlocks->setObjectName(QStringLiteral("checkExportBlocks"));
            _exportBlocks->setChecked(exportBlocks);
            root->addWidget(_exportBlocks);

            _buildTexture = new QCheckBox(QStringLiteral("生成预览纹理"), this);
            _buildTexture->setObjectName(QStringLiteral("checkBuildTexture"));
            _buildTexture->setChecked(buildTexture);
            root->addWidget(_buildTexture);

            auto* outputRow = new QWidget(this);
            auto* outputLayout = new QHBoxLayout(outputRow);
            outputLayout->setContentsMargins(0, 0, 0, 0);
            outputLayout->setSpacing(6);
            outputLayout->addWidget(new QLabel(QStringLiteral("输出文件夹:"), outputRow));
            _outputFolder = new QLineEdit(std::move(outputFolder), outputRow);
            _outputFolder->setObjectName(QStringLiteral("editFolder"));
            _browseButton = new QPushButton(QStringLiteral("…"), outputRow);
            _browseButton->setObjectName(QStringLiteral("buttonBrowseFolder"));
            _browseButton->setToolTip(QStringLiteral("浏览"));
            _browseButton->setFixedWidth(32);
            outputLayout->addWidget(_outputFolder, 1);
            outputLayout->addWidget(_browseButton);
            root->addWidget(outputRow);
            root->addStretch();

            auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok, Qt::Horizontal, this);
            buttons->setObjectName(QStringLiteral("buttonBox"));
            buttons->setCenterButtons(true);
            buttons->setLayoutDirection(Qt::RightToLeft);
            buttons->button(QDialogButtonBox::Cancel)->setIcon(QIcon());
            buttons->button(QDialogButtonBox::Ok)->setIcon(QIcon());
            buttons->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("Cancel"));
            buttons->button(QDialogButtonBox::Ok)->setText(QStringLiteral("OK"));
            connect(buttons, &QDialogButtonBox::accepted, this, &BuildModelPreviewDialog::accept);
            connect(buttons, &QDialogButtonBox::rejected, this, &BuildModelPreviewDialog::reject);
            root->addWidget(buttons);

            connect(_exportBlocks, &QCheckBox::toggled, this, [this]() { updateDependentControls(); });
            connect(_browseButton,
                    &QPushButton::clicked,
                    this,
                    [this]()
                    {
                        const QString directory = QFileDialog::getExistingDirectory(
                            this, QStringLiteral("选择区块输出文件夹"), _outputFolder->text());
                        if (!directory.isEmpty())
                        {
                            _outputFolder->setText(directory);
                        }
                    });
            updateDependentControls();
            adjustSize();
        }

        bool exportBlocks() const
        {
            return _exportBlocks->isChecked();
        }

        bool buildTexture() const
        {
            return _buildTexture->isChecked();
        }

        QString outputFolder() const
        {
            return _outputFolder->text().trimmed();
        }

    protected:
        void accept() override
        {
            if (_exportBlocks->isChecked() && _outputFolder->text().trimmed().isEmpty())
            {
                return;
            }
            QDialog::accept();
        }

    private:
        void updateDependentControls()
        {
            const bool enabled = _exportBlocks->isChecked();
            _outputFolder->setEnabled(enabled);
            _browseButton->setEnabled(enabled);
        }

        QCheckBox* _exportBlocks = nullptr;
        QCheckBox* _buildTexture = nullptr;
        QLineEdit* _outputFolder = nullptr;
        QPushButton* _browseButton = nullptr;
    };

} // namespace

GenerateModelDialog::GenerateModelDialog(QWidget* parent) : QDialog(parent)
{
    setObjectName(QStringLiteral("BuildModelDialog"));
    setProperty("referenceModelDialog", true);
    setWindowTitle(tr("生成网格"));
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint);
    setModal(true);
    setFixedWidth(404);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(12, 10, 12, 10);
    root->setSpacing(4);

    auto* titleBar = new DialogTitleBar(this);
    titleBar->setObjectName(QStringLiteral("DialogTitleBar"));
    titleBar->setFixedHeight(30);
    auto* titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    auto* title = new QLabel(tr("生成网格"), titleBar);
    title->setObjectName(QStringLiteral("DialogTitle"));
    auto* closeButton = new QPushButton(QStringLiteral("×"), titleBar);
    closeButton->setObjectName(QStringLiteral("DialogClose"));
    closeButton->setFixedSize(30, 30);
    closeButton->setToolTip(tr("关闭"));
    titleLayout->addWidget(title);
    titleLayout->addStretch();
    titleLayout->addWidget(closeButton);
    connect(closeButton, &QPushButton::clicked, this, &GenerateModelDialog::reject);
    root->addWidget(titleBar);

    QToolButton* generalToggle = nullptr;
    QFrame* generalBody = makeSection(this, tr("一般"), QStringLiteral("workflowGeneralGroup"), &generalToggle, true);
    auto* generalForm = makeForm(generalBody);

    _sourceCombo = makeCombo(generalBody, "modelSourceCombo");
    _sourceItemCombo = makeCombo(generalBody, "modelSourceItemCombo");
    _sourceItemCombo->hide();
    generalForm->addRow(tr("源数据:"), _sourceCombo);

    _surfaceTypeCombo = makeCombo(generalBody, "modelSurfaceTypeCombo");
    _surfaceTypeCombo->addItem(tr("任意（3D）"), QStringLiteral("arbitrary_3d"));
    _surfaceTypeCombo->addItem(tr("高度场（2.5D）"), QStringLiteral("height_field"));
    generalForm->addRow(tr("表面类型:"), _surfaceTypeCombo);

    _qualityLabel = new QLabel(tr("质量:"), generalBody);
    _qualityLabel->setObjectName(QStringLiteral("labelQuality"));
    _qualityCombo = makeCombo(generalBody, "modelQualityCombo");
    _qualityCombo->addItem(tr("超高"), QStringLiteral("highest"));
    _qualityCombo->addItem(tr("高"), QStringLiteral("high"));
    _qualityCombo->addItem(tr("中"), QStringLiteral("medium"));
    _qualityCombo->addItem(tr("低"), QStringLiteral("low"));
    _qualityCombo->addItem(tr("最低"), QStringLiteral("lowest"));
    _qualityCombo->setCurrentIndex(2);
    generalForm->addRow(_qualityLabel, _qualityCombo);

    _faceCountModeCombo = makeCombo(generalBody, "modelFaceCountModeCombo");
    _faceCountModeCombo->addItem(tr("高"), QStringLiteral("high"));
    _faceCountModeCombo->addItem(tr("中"), QStringLiteral("medium"));
    _faceCountModeCombo->addItem(tr("低"), QStringLiteral("low"));
    _faceCountModeCombo->addItem(tr("自定义"), QStringLiteral("custom"));
    generalForm->addRow(tr("面数:"), _faceCountModeCombo);

    _customFaceCountLabel = new QLabel(tr("自定义面数:"), generalBody);
    _customFaceCountSpin = new QSpinBox(generalBody);
    _customFaceCountSpin->setObjectName(QStringLiteral("modelCustomFaceCountSpin"));
    _customFaceCountSpin->setRange(1, 2000000);
    _customFaceCountSpin->setValue(200000);
    _customFaceCountSpin->setGroupSeparatorShown(true);
    generalForm->addRow(_customFaceCountLabel, _customFaceCountSpin);

    _rpcHeightMinLabel = new QLabel(tr("RPC 最低高程:"), generalBody);
    _rpcHeightMaxLabel = new QLabel(tr("RPC 最高高程:"), generalBody);
    _rpcHeightMinSpin = new QDoubleSpinBox(generalBody);
    _rpcHeightMaxSpin = new QDoubleSpinBox(generalBody);
    _rpcHeightMinSpin->setObjectName(QStringLiteral("rpcHeightMinMetersSpin"));
    _rpcHeightMaxSpin->setObjectName(QStringLiteral("rpcHeightMaxMetersSpin"));
    for (QDoubleSpinBox* spinBox : {_rpcHeightMinSpin, _rpcHeightMaxSpin})
    {
        spinBox->setRange(-1000000.0, 1000000.0);
        spinBox->setDecimals(3);
        spinBox->setSuffix(tr(" m"));
    }
    generalForm->addRow(_rpcHeightMinLabel, _rpcHeightMinSpin);
    generalForm->addRow(_rpcHeightMaxLabel, _rpcHeightMaxSpin);

    _saveAfterEachStepCheck = new QCheckBox(tr("在每个步骤完成后保存项目"), generalBody);
    _saveAfterEachStepCheck->setObjectName(QStringLiteral("checkSaveProject"));
    _saveAfterEachStepCheck->setToolTip(tr("当前模型任务为单阶段提交，此选项仅按参考界面展示。"));
    _saveAfterEachStepCheck->setEnabled(false);
    generalForm->addRow(_saveAfterEachStepCheck);
    setLabelColumnWidth(generalForm, 168);
    root->addWidget(generalBody->parentWidget());

    QToolButton* blocksToggle = nullptr;
    QFrame* blocksBody = makeSection(this, tr("区块"), QStringLiteral("groupBlocks"), &blocksToggle, true);
    auto* blocksForm = makeForm(blocksBody);

    _splitInBlocksCheck = new QCheckBox(tr("分割成区块"), blocksBody);
    _splitInBlocksCheck->setObjectName(QStringLiteral("checkSplitInBlocks"));
    _splitInBlocksCheck->setToolTip(tr("参考界面已对齐；分块模型调度尚未接入当前生产流程。"));
    blocksForm->addRow(_splitInBlocksCheck);

    _coordinateSystemCombo = makeCombo(blocksBody, "comboCoordinateSystem");
    _coordinateSystemCombo->addItem(QStringLiteral("Local Coordinates (m)"), QStringLiteral("local"));
    blocksForm->addRow(tr("坐标系统:"), _coordinateSystemCombo);

    _blockSizeSpin = new QDoubleSpinBox(blocksBody);
    _blockSizeSpin->setObjectName(QStringLiteral("editBlocksSize"));
    _blockSizeSpin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    _blockSizeSpin->setDecimals(3);
    _blockSizeSpin->setRange(0.001, 1000000000.0);
    _blockSizeSpin->setValue(250.0);
    blocksForm->addRow(tr("区块大小 (m):"), _blockSizeSpin);

    auto* origin = new QWidget(blocksBody);
    origin->setObjectName(QStringLiteral("widgetBlocksCoordinates"));
    auto* originLayout = new QHBoxLayout(origin);
    originLayout->setContentsMargins(0, 0, 0, 0);
    originLayout->setSpacing(6);
    originLayout->addWidget(new QLabel(QStringLiteral("X:"), origin));
    _blockOriginXSpin = new QDoubleSpinBox(origin);
    _blockOriginXSpin->setObjectName(QStringLiteral("editBlocksOriginX"));
    _blockOriginXSpin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    _blockOriginXSpin->setRange(-1000000000.0, 1000000000.0);
    _blockOriginXSpin->setMaximumWidth(82);
    originLayout->addWidget(_blockOriginXSpin);
    originLayout->addWidget(new QLabel(QStringLiteral("Y:"), origin));
    _blockOriginYSpin = new QDoubleSpinBox(origin);
    _blockOriginYSpin->setObjectName(QStringLiteral("editBlocksOriginY"));
    _blockOriginYSpin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    _blockOriginYSpin->setRange(-1000000000.0, 1000000000.0);
    _blockOriginYSpin->setMaximumWidth(82);
    originLayout->addWidget(_blockOriginYSpin);
    blocksForm->addRow(tr("网格原点:"), origin);

    auto* boundaryRow = new QWidget(blocksBody);
    auto* boundaryLayout = new QHBoxLayout(boundaryRow);
    boundaryLayout->setContentsMargins(0, 0, 0, 0);
    _skipBoundaryBlocksCheck = new QCheckBox(tr("跳过边界外的块"), boundaryRow);
    _skipBoundaryBlocksCheck->setObjectName(QStringLiteral("checkClipToBoundary"));
    _blocksPreviewButton = new QPushButton(tr("预览..."), boundaryRow);
    _blocksPreviewButton->setObjectName(QStringLiteral("buttonBlocksPreview"));
    boundaryLayout->addWidget(_skipBoundaryBlocksCheck);
    boundaryLayout->addStretch();
    boundaryLayout->addWidget(_blocksPreviewButton);
    blocksForm->addRow(boundaryRow);
    setLabelColumnWidth(blocksForm, 92);
    root->addWidget(blocksBody->parentWidget());

    _advancedContent = makeSection(this, tr("高级"), QStringLiteral("workflowAdvancedGroup"), &_advancedToggle, false);
    auto* advancedForm = makeForm(_advancedContent);

    _interpolationCombo = makeCombo(_advancedContent, "modelInterpolationCombo");
    _interpolationCombo->addItem(tr("已禁用"), QStringLiteral("disabled"));
    _interpolationCombo->addItem(tr("已启用（默认）"), QStringLiteral("enabled"));
    _interpolationCombo->addItem(tr("推断"), QStringLiteral("extrapolated"));
    _interpolationCombo->setCurrentIndex(1);
    advancedForm->addRow(tr("插值:"), _interpolationCombo);

    _depthFilteringLabel = new QLabel(tr("深度过滤:"), _advancedContent);
    _depthFilteringLabel->setObjectName(QStringLiteral("labelFilterMode"));
    _depthFilteringCombo = makeCombo(_advancedContent, "modelDepthFilteringCombo");
    _depthFilteringCombo->addItem(tr("已禁用"), QStringLiteral("disabled"));
    _depthFilteringCombo->addItem(tr("轻度"), QStringLiteral("mild"));
    _depthFilteringCombo->addItem(tr("中度"), QStringLiteral("moderate"));
    _depthFilteringCombo->addItem(tr("激进"), QStringLiteral("aggressive"));
    _depthFilteringCombo->setCurrentIndex(1);
    advancedForm->addRow(_depthFilteringLabel, _depthFilteringCombo);

    _pointClassesLabel = new QLabel(tr("点类: 所有"), _advancedContent);
    _pointClassesLabel->setObjectName(QStringLiteral("labelClasses"));
    _selectPointClassesButton = new QPushButton(tr("请选择..."), _advancedContent);
    _selectPointClassesButton->setObjectName(QStringLiteral("buttonSelectClasses"));
    auto* pointClassesField = new QWidget(_advancedContent);
    pointClassesField->setObjectName(QStringLiteral("widgetClasses"));
    auto* pointClassesLayout = new QHBoxLayout(pointClassesField);
    pointClassesLayout->setContentsMargins(0, 0, 0, 0);
    pointClassesLayout->addStretch();
    pointClassesLayout->addWidget(_selectPointClassesButton);
    advancedForm->addRow(_pointClassesLabel, pointClassesField);

    _vertexColorsCheck = new QCheckBox(tr("计算顶点颜色"), _advancedContent);
    _vertexColorsCheck->setObjectName(QStringLiteral("checkVertexColors"));
    _vertexColorsCheck->setChecked(true);
    _vertexColorsCheck->setToolTip(tr("当前参考生产链固定生成顶点颜色。"));
    advancedForm->addRow(_vertexColorsCheck);

    _strictVolumetricMasksCheck = new QCheckBox(tr("使用严格的体积掩模"), _advancedContent);
    _strictVolumetricMasksCheck->setObjectName(QStringLiteral("checkStrictVolumetricMasks"));
    _strictVolumetricMasksCheck->setToolTip(tr("逐相机影像掩模已自动接入；严格体积掩模属于不同的数据语义。"));
    _strictVolumetricMasksCheck->setEnabled(false);
    advancedForm->addRow(_strictVolumetricMasksCheck);

    _reuseDepthMapsCheck = new QCheckBox(tr("重用深度图"), _advancedContent);
    _reuseDepthMapsCheck->setObjectName(QStringLiteral("reuseDepthMapsCheck"));
    _reuseDepthMapsCheck->setChecked(true);
    advancedForm->addRow(_reuseDepthMapsCheck);

    _replaceDefaultCheck = new QCheckBox(tr("要替换默认模型吗"), _advancedContent);
    _replaceDefaultCheck->setObjectName(QStringLiteral("replaceDefaultModelCheck"));
    advancedForm->addRow(_replaceDefaultCheck);
    setLabelColumnWidth(advancedForm, 168);
    root->addWidget(_advancedContent->parentWidget());

    auto* buttonBox = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok, Qt::Horizontal, this);
    buttonBox->setObjectName(QStringLiteral("workflowButtonBox"));
    buttonBox->setCenterButtons(true);
    buttonBox->setLayoutDirection(Qt::RightToLeft);
    buttonBox->button(QDialogButtonBox::Cancel)->setIcon(QIcon());
    buttonBox->button(QDialogButtonBox::Ok)->setIcon(QIcon());
    buttonBox->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("Cancel"));
    buttonBox->button(QDialogButtonBox::Ok)->setText(QStringLiteral("OK"));
    buttonBox->button(QDialogButtonBox::Cancel)->setFixedWidth(86);
    buttonBox->button(QDialogButtonBox::Ok)->setFixedWidth(86);
    _okButton = buttonBox->button(QDialogButtonBox::Ok);
    root->addWidget(buttonBox, 0, Qt::AlignHCenter);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &GenerateModelDialog::onRun);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &GenerateModelDialog::reject);
    connect(_sourceCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::onSourceTypeChanged);
    connect(_sourceItemCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_surfaceTypeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_qualityCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_faceCountModeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            [this]()
            {
                updateCustomFaceCountVisibility();
                emitSettingsNow();
            });
    connect(
        _customFaceCountSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &GenerateModelDialog::emitSettingsNow);
    connect(_rpcHeightMinSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_rpcHeightMaxSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_splitInBlocksCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);
    connect(_blocksPreviewButton,
            &QPushButton::clicked,
            this,
            [this]()
            {
                BuildModelPreviewDialog preview(
                    _exportCompletedBlocks, _generatePreviewTextures, _blockOutputFolder, this);
                if (preview.exec() == QDialog::Accepted)
                {
                    _exportCompletedBlocks = preview.exportBlocks();
                    _generatePreviewTextures = preview.buildTexture();
                    _blockOutputFolder = preview.outputFolder();
                }
            });
    connect(_interpolationCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_depthFilteringCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_vertexColorsCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);
    connect(_reuseDepthMapsCheck,
            &QCheckBox::toggled,
            this,
            [this](bool checked)
            {
                if (_hasReusableDepthMaps)
                {
                    _reuseDepthMapsRequested = checked;
                }
                emitSettingsNow();
            });
    connect(_replaceDefaultCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);

    setAdvancedExpanded(false);
    updateCustomFaceCountVisibility();
    refreshSourceTypes();
    adjustSize();
    setFixedWidth(404);
}

void GenerateModelDialog::applySettings(const QJsonObject& settings)
{
    _pendingSourceData = settings.value(QLatin1String(kSourceData)).toString();
    _pendingSourcePath = settings.value(QLatin1String(kSourcePath))
                             .toString(settings.value(QStringLiteral("denseCloudPath")).toString());
    QString computeMode = settings.value(QLatin1String(kComputeMode)).toString().trimmed().toLower();
    if (computeMode.isEmpty())
    {
        const QString backend = settings.value(QStringLiteral("patch_match_backend"))
                                    .toString(settings.value(QStringLiteral("processingDevice")).toString())
                                    .trimmed()
                                    .toLower();
        computeMode = backend == QStringLiteral("cuda") || backend == QStringLiteral("opencl")
                          ? backend
                          : QStringLiteral("hybrid");
    }
    if (computeMode == QStringLiteral("cuda") || computeMode == QStringLiteral("opencl") ||
        computeMode == QStringLiteral("hybrid"))
    {
        _computeMode = computeMode;
    }

    const QString faceCountMode = settings.value(QStringLiteral("faceCountMode")).toString().trimmed();
    const int modeIndex = _faceCountModeCombo->findData(faceCountMode);
    if (modeIndex >= 0)
    {
        _faceCountModeCombo->setCurrentIndex(modeIndex);
    }
    else if (settings.contains(QStringLiteral("targetFaces")))
    {
        const int legacyFaces = settings.value(QStringLiteral("targetFaces")).toInt();
        const QString legacyMode = legacyFaces <= 200000 ? QStringLiteral("high") : QStringLiteral("custom");
        _faceCountModeCombo->setCurrentIndex(_faceCountModeCombo->findData(legacyMode));
        _customFaceCountSpin->setValue(qBound(1, legacyFaces, 2000000));
    }
    _customFaceCountSpin->setValue(
        qBound(1, settings.value(QStringLiteral("faceCountCustom")).toInt(_customFaceCountSpin->value()), 2000000));

    const int qualityIndex =
        _qualityCombo->findData(settings.value(QStringLiteral("depthQualityProfile")).toString().trimmed());
    if (qualityIndex >= 0)
    {
        _qualityCombo->setCurrentIndex(qualityIndex);
    }
    const int interpolationIndex =
        _interpolationCombo->findData(settings.value(QStringLiteral("interpolation")).toString().trimmed());
    if (interpolationIndex >= 0)
    {
        _interpolationCombo->setCurrentIndex(interpolationIndex);
    }
    const int filteringIndex =
        _depthFilteringCombo->findData(settings.value(QStringLiteral("depthFiltering")).toString().trimmed());
    if (filteringIndex >= 0)
    {
        _depthFilteringCombo->setCurrentIndex(filteringIndex);
    }
    if (settings.contains(QStringLiteral("rpcHeightMinMeters")))
    {
        _rpcHeightMinSpin->setValue(settings.value(QStringLiteral("rpcHeightMinMeters")).toDouble());
    }
    if (settings.contains(QStringLiteral("rpcHeightMaxMeters")))
    {
        _rpcHeightMaxSpin->setValue(settings.value(QStringLiteral("rpcHeightMaxMeters")).toDouble());
    }
    _reuseDepthMapsRequested = settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    _reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested);
    _replaceDefaultCheck->setChecked(settings.value(QStringLiteral("replaceDefaultModel")).toBool(false));
    updateCustomFaceCountVisibility();
    updateAvailability();
}

void GenerateModelDialog::setSourceCandidates(const QJsonArray& candidates)
{
    _candidates = QJsonArray();
    _hasReusableDepthMaps = false;
    bool hasDepthCandidate = false;
    bool hasRpcCandidate = false;
    bool canGenerateDepthMaps = false;
    for (const QJsonValue& value : candidates)
    {
        const QJsonObject candidate = value.toObject();
        const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
        if (sourceData == QStringLiteral("tie_points") && candidate.value(QLatin1String(kSupported)).toBool(false))
        {
            canGenerateDepthMaps = true;
        }
        if (sourceData == QStringLiteral("depth_maps"))
        {
            _candidates.append(candidate);
            hasDepthCandidate = true;
            if (candidate.value(QLatin1String(kSupported)).toBool(false) &&
                candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true) &&
                !candidate.value(QLatin1String(kSourcePath)).toString().trimmed().isEmpty())
            {
                _hasReusableDepthMaps = true;
            }
            if (candidate.contains(QStringLiteral("rpcHeightMinMeters")))
            {
                _rpcHeightMinSpin->setValue(candidate.value(QStringLiteral("rpcHeightMinMeters")).toDouble());
            }
            if (candidate.contains(QStringLiteral("rpcHeightMaxMeters")))
            {
                _rpcHeightMaxSpin->setValue(candidate.value(QStringLiteral("rpcHeightMaxMeters")).toDouble());
            }
        }
        else if (sourceData == QStringLiteral("rpc_height_plane_sweep") && hasValidRpcImagePair(candidate))
        {
            _candidates.append(candidate);
            hasRpcCandidate = true;
            if (candidate.contains(QStringLiteral("rpcHeightMinMeters")))
            {
                _rpcHeightMinSpin->setValue(candidate.value(QStringLiteral("rpcHeightMinMeters")).toDouble());
            }
            if (candidate.contains(QStringLiteral("rpcHeightMaxMeters")))
            {
                _rpcHeightMaxSpin->setValue(candidate.value(QStringLiteral("rpcHeightMaxMeters")).toDouble());
            }
        }
        else if (sourceData == QStringLiteral("point_cloud") || sourceData == QStringLiteral("tie_points"))
        {
            QJsonObject visibleCandidate = candidate;
            visibleCandidate[QLatin1String(kSupported)] = false;
            visibleCandidate[QLatin1String(kNote)] =
                QStringLiteral("参考 recovered 生产链尚未闭合该数据源；不会回退到 PlaScan 旧建模算法。");
            _candidates.append(visibleCandidate);
        }
    }

    if (!hasDepthCandidate && !hasRpcCandidate)
    {
        QJsonObject automaticDepthMaps;
        automaticDepthMaps[QLatin1String(kSourceData)] = QStringLiteral("depth_maps");
        automaticDepthMaps[QLatin1String(kSourceLabel)] = QStringLiteral("深度图");
        automaticDepthMaps[QLatin1String(kSourcePath)] = QString();
        automaticDepthMaps[QLatin1String(kDisplay)] = QStringLiteral("自动生成深度图");
        automaticDepthMaps[QLatin1String(kSupported)] = canGenerateDepthMaps;
        automaticDepthMaps[QLatin1String(kAutomaticDepthMaps)] = true;
        automaticDepthMaps[QLatin1String(kNote)] =
            canGenerateDepthMaps ? QStringLiteral("缺少深度图时将自动估计深度图，再执行参考已验证的 recovered OOC。")
                                 : QStringLiteral("需要先完成通过质量门控的照片对齐，才能自动估计深度图。");
        _candidates.prepend(automaticDepthMaps);

        if (_pendingSourceData.isEmpty() || _pendingSourceData == QStringLiteral("tie_points"))
        {
            _pendingSourceData = QStringLiteral("depth_maps");
            _pendingSourcePath.clear();
        }
    }
    refreshSourceTypes();
}

QJsonObject GenerateModelDialog::collectSettings() const
{
    const QJsonObject candidate = currentCandidate();
    const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
    const QString sourcePath = candidate.value(QLatin1String(kSourcePath)).toString();
    const QString faceCountMode = _faceCountModeCombo->currentData().toString();
    const int customFaces = _customFaceCountSpin->value();
    const int targetFaces = faceCountMode == QStringLiteral("custom") ? customFaces : 0;
    const bool rpcMode = usesRpcHeightPlaneSweep();

    QJsonObject settings;
    settings[QStringLiteral("modelGenerationContractRevision")] = 2;
    settings[QLatin1String(kSourceData)] = sourceData;
    settings[QLatin1String(kSourceLabel)] =
        candidate.value(QLatin1String(kSourceLabel)).toString(defaultSourceLabel(sourceData));
    settings[QLatin1String(kSourcePath)] = sourcePath;
    settings[QStringLiteral("source_display")] = candidate.value(QLatin1String(kDisplay)).toString();
    settings[QStringLiteral("source_supported")] = candidate.value(QLatin1String(kSupported)).toBool(false);
    settings[QStringLiteral("depthQualityProfile")] =
        rpcMode ? QStringLiteral("medium") : _qualityCombo->currentData().toString();
    settings[QStringLiteral("surfaceQualityProfile")] =
        rpcMode ? QStringLiteral("rpc_height_plane_sweep") : QStringLiteral("recovered_ooc");
    settings[QStringLiteral("reconstruction_mode")] =
        rpcMode ? QStringLiteral("rpc_height_plane_sweep") : QStringLiteral("recovered_ooc");
    settings[QStringLiteral("faceCountMode")] = faceCountMode;
    settings[QStringLiteral("faceCountCustom")] = customFaces;
    settings[QStringLiteral("simplifyTargetFaces")] = targetFaces;
    settings[QStringLiteral("requestedTargetFaces")] = targetFaces;
    settings[QStringLiteral("export_format")] = QStringLiteral("PLY");
    settings[QStringLiteral("depthFiltering")] = _depthFilteringCombo->currentData().toString();
    settings[QStringLiteral("interpolation")] = _interpolationCombo->currentData().toString();
    if (rpcMode)
    {
        settings[QStringLiteral("rpcImagePaths")] = candidate.value(QStringLiteral("rpcImagePaths"));
        settings[QStringLiteral("rpcHeightMinMeters")] = _rpcHeightMinSpin->value();
        settings[QStringLiteral("rpcHeightMaxMeters")] = _rpcHeightMaxSpin->value();
    }

    const bool selectedDepthBatchCompatible = candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true);
    const bool reuseDepthMaps = _hasReusableDepthMaps && selectedDepthBatchCompatible && _reuseDepthMapsRequested;
    settings[QStringLiteral("reuseDepthMaps")] = reuseDepthMaps;
    const bool automaticDepthMaps = candidate.value(QLatin1String(kAutomaticDepthMaps)).toBool(false);
    settings[QStringLiteral("automatic_depth_maps")] = automaticDepthMaps;
    settings[QStringLiteral("force_depth_recompute")] =
        automaticDepthMaps || (sourceData == QStringLiteral("depth_maps") && !reuseDepthMaps);
    settings[QStringLiteral("replaceDefaultModel")] = _replaceDefaultCheck->isChecked();
    settings[QLatin1String(kComputeMode)] = _computeMode;
    settings[QStringLiteral("patch_match_backend")] =
        _computeMode == QStringLiteral("hybrid") ? QStringLiteral("auto") : _computeMode;
    settings[QStringLiteral("processingDevice")] =
        _computeMode == QStringLiteral("hybrid") ? QStringLiteral("auto") : _computeMode;

    if (sourceData == QStringLiteral("point_cloud") || sourceData == QStringLiteral("tie_points") ||
        sourceData == QStringLiteral("model"))
    {
        settings[QStringLiteral("denseCloudPath")] = sourcePath;
    }
    if (sourceData == QStringLiteral("depth_maps"))
    {
        settings[QStringLiteral("depthMapSourcePath")] = sourcePath;
    }
    return settings;
}

QJsonObject GenerateModelDialog::currentCandidate() const
{
    return _sourceItemCombo->currentData().toJsonObject();
}

bool GenerateModelDialog::usesRecoveredModelPipeline() const
{
    return currentCandidate().value(QLatin1String(kSourceData)).toString() == QStringLiteral("depth_maps") &&
           !usesRpcHeightPlaneSweep();
}

bool GenerateModelDialog::usesRpcHeightPlaneSweep() const
{
    const QJsonObject candidate = currentCandidate();
    return candidate.value(QLatin1String(kSourceData)).toString() == QStringLiteral("rpc_height_plane_sweep") &&
           hasValidRpcImagePair(candidate);
}

void GenerateModelDialog::refreshSourceTypes()
{
    const QString currentSource =
        !_pendingSourceData.isEmpty() ? _pendingSourceData : _sourceCombo->currentData().toString();
    _sourceCombo->blockSignals(true);
    _sourceCombo->clear();

    QStringList addedTypes;
    for (const QJsonValue& value : _candidates)
    {
        const QJsonObject candidate = value.toObject();
        const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
        if (sourceData.isEmpty() || addedTypes.contains(sourceData))
        {
            continue;
        }
        addedTypes.push_back(sourceData);
        _sourceCombo->addItem(candidate.value(QLatin1String(kSourceLabel)).toString(defaultSourceLabel(sourceData)),
                              sourceData);
        const int itemIndex = _sourceCombo->count() - 1;
        _sourceCombo->setItemData(
            itemIndex, candidate.value(QLatin1String(kSupported)).toBool(false), Qt::UserRole - 1);
        _sourceCombo->setItemData(itemIndex, candidate.value(QLatin1String(kNote)).toString(), Qt::ToolTipRole);
    }

    if (_sourceCombo->count() == 0)
    {
        _sourceCombo->addItem(tr("无可用源数据"), QString());
    }
    const int index = _sourceCombo->findData(currentSource);
    if (index >= 0)
    {
        _sourceCombo->setCurrentIndex(index);
    }
    _sourceCombo->blockSignals(false);
    refreshSourceItems();
}

void GenerateModelDialog::refreshSourceItems()
{
    const QString sourceData = _sourceCombo->currentData().toString();
    const QString currentPath = !_pendingSourcePath.isEmpty()
                                    ? _pendingSourcePath
                                    : currentCandidate().value(QLatin1String(kSourcePath)).toString();
    _sourceItemCombo->blockSignals(true);
    _sourceItemCombo->clear();

    for (const QJsonValue& value : _candidates)
    {
        const QJsonObject candidate = value.toObject();
        if (candidate.value(QLatin1String(kSourceData)).toString() != sourceData)
        {
            continue;
        }
        const QString display =
            candidate.value(QLatin1String(kDisplay)).toString(candidate.value(QLatin1String(kSourcePath)).toString());
        _sourceItemCombo->addItem(display, candidate);
    }

    int index = -1;
    for (int i = 0; i < _sourceItemCombo->count(); ++i)
    {
        const QJsonObject candidate = _sourceItemCombo->itemData(i).toJsonObject();
        if (candidate.value(QLatin1String(kSourcePath)).toString() == currentPath)
        {
            index = i;
            break;
        }
    }
    if (index >= 0)
    {
        _sourceItemCombo->setCurrentIndex(index);
    }
    else if (_sourceItemCombo->count() > 0)
    {
        _sourceItemCombo->setCurrentIndex(0);
    }

    _sourceItemCombo->blockSignals(false);
    _pendingSourceData.clear();
    _pendingSourcePath.clear();
    updateAvailability();
    emitSettingsNow();
}

void GenerateModelDialog::setAdvancedExpanded(bool expanded)
{
    const QSignalBlocker blocker(_advancedToggle);
    _advancedToggle->setChecked(expanded);
    _advancedToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    _advancedContent->setVisible(expanded);
    adjustSize();
}

void GenerateModelDialog::updateAvailability()
{
    const QJsonObject candidate = currentCandidate();
    const bool supported = candidate.value(QLatin1String(kSupported)).toBool(false);
    const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
    const bool hasCandidate = !sourceData.isEmpty();
    const bool selectedDepthBatchCompatible = candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true);
    const bool canReuseDepthMaps = _hasReusableDepthMaps && selectedDepthBatchCompatible;
    const bool rpcMode = usesRpcHeightPlaneSweep();
    const bool usesDepth = usesRecoveredModelPipeline();

    _qualityLabel->setVisible(usesDepth);
    _qualityCombo->setVisible(usesDepth);
    _surfaceTypeCombo->setEnabled(!usesDepth && !rpcMode);
    if (usesDepth || rpcMode)
    {
        _surfaceTypeCombo->setCurrentIndex(_surfaceTypeCombo->findData(QStringLiteral("arbitrary_3d")));
    }
    const bool showRpcHeights = sourceData == QStringLiteral("rpc_height_plane_sweep");
    _rpcHeightMinLabel->setVisible(showRpcHeights);
    _rpcHeightMinSpin->setVisible(showRpcHeights);
    _rpcHeightMaxLabel->setVisible(showRpcHeights);
    _rpcHeightMaxSpin->setVisible(showRpcHeights);
    updateCustomFaceCountVisibility();

    _reuseDepthMapsCheck->setEnabled(canReuseDepthMaps && !rpcMode);
    if (sourceData == QStringLiteral("depth_maps") && !selectedDepthBatchCompatible)
    {
        _reuseDepthMapsRequested = false;
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(false);
        const QString reason = candidate.value(QLatin1String(kDepthBatchCompatibilityReason)).toString();
        _reuseDepthMapsCheck->setToolTip(reason.isEmpty()
                                             ? tr("现有深度图与当前工程不兼容；将自动重新估计深度图。")
                                             : tr("现有深度图与当前工程不兼容；将自动重新估计：%1").arg(reason));
    }
    else if (_hasReusableDepthMaps)
    {
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested);
        _reuseDepthMapsCheck->setToolTip(tr("复用兼容深度图；实际执行参数将写入模型记录。"));
    }
    else
    {
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(false);
        _reuseDepthMapsCheck->setToolTip(tr("当前项目没有可复用的深度图；生成模型时将先自动估计深度图。"));
    }

    _depthFilteringLabel->setEnabled(usesDepth && !_reuseDepthMapsCheck->isChecked());
    _depthFilteringCombo->setEnabled(usesDepth && !_reuseDepthMapsCheck->isChecked());
    const bool usesPointClasses =
        sourceData == QStringLiteral("point_cloud") || sourceData == QStringLiteral("tie_points");
    _pointClassesLabel->setEnabled(usesPointClasses);
    _selectPointClassesButton->setEnabled(usesPointClasses);

    const bool blocksEnabled = _splitInBlocksCheck->isChecked();
    _coordinateSystemCombo->setEnabled(blocksEnabled);
    _blockSizeSpin->setEnabled(blocksEnabled);
    _blockOriginXSpin->setEnabled(blocksEnabled);
    _blockOriginYSpin->setEnabled(blocksEnabled);
    _skipBoundaryBlocksCheck->setEnabled(blocksEnabled);
    _blocksPreviewButton->setEnabled(blocksEnabled);

    const bool validRpcConfiguration = !showRpcHeights || (rpcMode && std::isfinite(_rpcHeightMinSpin->value()) &&
                                                           std::isfinite(_rpcHeightMaxSpin->value()) &&
                                                           _rpcHeightMinSpin->value() < _rpcHeightMaxSpin->value());
    const bool validSurface = _surfaceTypeCombo->currentData().toString() == QStringLiteral("arbitrary_3d");
    const bool validReferenceOptions =
        validSurface && !_splitInBlocksCheck->isChecked() && _vertexColorsCheck->isChecked();
    const QString note = candidate.value(QLatin1String(kNote)).toString();
    _sourceCombo->setToolTip(note);

    QString actionTip = note;
    if (!validRpcConfiguration)
    {
        actionTip = tr("RPC 最低高程必须小于最高高程。");
    }
    else if (!validSurface)
    {
        actionTip = tr("当前 recovered 生产路径只支持“任意（3D）”表面。");
    }
    else if (_splitInBlocksCheck->isChecked())
    {
        actionTip = tr("分块模型调度尚未接入当前生产流程。");
    }
    else if (!_vertexColorsCheck->isChecked())
    {
        actionTip = tr("当前参考生产链固定生成顶点颜色。");
    }
    _okButton->setToolTip(actionTip);
    _okButton->setEnabled(hasCandidate && supported && validRpcConfiguration && validReferenceOptions);
    adjustSize();
}

void GenerateModelDialog::updateCustomFaceCountVisibility()
{
    const bool custom = _faceCountModeCombo->currentData().toString() == QStringLiteral("custom");
    _customFaceCountLabel->setVisible(custom);
    _customFaceCountSpin->setVisible(custom);
}

void GenerateModelDialog::emitSettingsNow()
{
    updateAvailability();
    emit settingsChanged(collectSettings());
}

void GenerateModelDialog::onRun()
{
    updateAvailability();
    if (!_okButton->isEnabled())
    {
        return;
    }

    const QJsonObject settings = collectSettings();
    if (!settings.value(QStringLiteral("source_supported")).toBool(false))
    {
        return;
    }
    accept();
    emit runRequested(settings);
}

void GenerateModelDialog::onSourceTypeChanged()
{
    refreshSourceItems();
}

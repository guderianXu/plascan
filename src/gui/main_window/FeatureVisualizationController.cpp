#include "FeatureVisualizationController.h"

#include "CanvasWidget.h"
#include "MainWindow.h"
#include "ProjectManager.h"
#include "settings/DialogSettingKeys.h"
#include "settings/DialogSettingStore.h"
#include "tie_points/FeaturePointVisualizationDialog.h"

#include <QColor>
#include <QMainWindow>

namespace
{

QColor colorFromJson(const QJsonObject &value, const QColor &fallback)
{
    if (value.isEmpty())
    {
        return fallback;
    }
    return QColor(value.value(QStringLiteral("r")).toInt(fallback.red()),
                  value.value(QStringLiteral("g")).toInt(fallback.green()),
                  value.value(QStringLiteral("b")).toInt(fallback.blue()));
}

QJsonObject colorToJson(const QColor &color)
{
    return QJsonObject{
        {QStringLiteral("r"), color.red()},
        {QStringLiteral("g"), color.green()},
        {QStringLiteral("b"), color.blue()}};
}

LayerRenderer::FeatureDisplayOptions optionsFromJson(const QJsonObject &settings)
{
    LayerRenderer::FeatureDisplayOptions options;
    options.pointSource = xjw::gui::views::featurePointSourceFromToken(
        settings.value(QStringLiteral("pointSource")).toString());
    options.showPoints = settings.value(QStringLiteral("showPoints")).toBool(options.showPoints);
    options.showResiduals =
        settings.value(QStringLiteral("showResiduals")).toBool(options.showResiduals);
    options.residualScale =
        settings.value(QStringLiteral("residualScale")).toDouble(options.residualScale);
    options.minimumResidualPx =
        settings.value(QStringLiteral("minimumResidualPx")).toDouble(options.minimumResidualPx);
    options.maximumResidualLengthPx = settings.value(QStringLiteral("maximumResidualLengthPx"))
                                          .toDouble(options.maximumResidualLengthPx);
    options.pointSize = settings.value(QStringLiteral("pointSize")).toInt(options.pointSize);
    options.opacity = settings.value(QStringLiteral("opacity")).toInt(options.opacity);
    options.maxDisplayCount =
        settings.value(QStringLiteral("maxDisplayCount")).toInt(options.maxDisplayCount);
    options.pointColor = colorFromJson(settings.value(QStringLiteral("pointColor")).toObject(),
                                       options.pointColor);
    return options;
}

QJsonObject optionsToJson(const LayerRenderer::FeatureDisplayOptions &options)
{
    return QJsonObject{
        {QStringLiteral("pointSource"),
         xjw::gui::views::featurePointSourceToken(options.pointSource)},
        {QStringLiteral("showPoints"), options.showPoints},
        {QStringLiteral("showResiduals"), options.showResiduals},
        {QStringLiteral("residualScale"), options.residualScale},
        {QStringLiteral("minimumResidualPx"), options.minimumResidualPx},
        {QStringLiteral("maximumResidualLengthPx"), options.maximumResidualLengthPx},
        {QStringLiteral("pointSize"), options.pointSize},
        {QStringLiteral("opacity"), options.opacity},
        {QStringLiteral("maxDisplayCount"), options.maxDisplayCount},
        {QStringLiteral("pointColor"), colorToJson(options.pointColor)}};
}

} // namespace

FeatureVisualizationController::FeatureVisualizationController(QMainWindow *mainWindow,
                                                                 QObject *parent)
    : QObject(parent)
    , _mainWindow(mainWindow)
{
}

void FeatureVisualizationController::setProjectManager(ProjectManager *projectManager)
{
    _projectManager = projectManager;
}

DialogSettingStore *FeatureVisualizationController::ensureSettingStore()
{
    if (!_projectManager)
    {
        return nullptr;
    }
    if (!_settingStore)
    {
        _settingStore = new DialogSettingStore(DialogSettingKeys::FeaturePointVisualization, this);
        _settingStore->setChangeCallback([this]()
        {
            if (_projectManager)
            {
                _projectManager->markWorkspaceDirty();
            }
        });
    }
    _settingStore->setProjectPath(_projectManager->currentProjectPath());
    return _settingStore;
}

QJsonObject FeatureVisualizationController::loadSettings(const QJsonObject &fallback)
{
    if (DialogSettingStore *store = ensureSettingStore())
    {
        const QJsonObject saved = store->load();
        if (!saved.isEmpty())
        {
            return saved;
        }
    }
    return fallback.value(QStringLiteral("feature_display")).toObject();
}

void FeatureVisualizationController::openDialog()
{
    if (!_mainWindow)
    {
        return;
    }

    auto *dialog = new FeaturePointVisualizationDialog(_mainWindow);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    const QJsonObject saved = loadSettings();
    if (!saved.isEmpty())
    {
        dialog->setDisplayOptions(optionsFromJson(saved));
    }

    auto *mainWindow = qobject_cast<MainWindow *>(_mainWindow.data());
    if (CanvasWidget *canvas = mainWindow ? mainWindow->canvas() : nullptr)
    {
        LayerRenderer::FeatureDisplayOptions current = dialog->getDisplayOptions();
        current.showPoints = canvas->showsInterestPoints();
        current.showResiduals = canvas->showsFeatureResiduals();
        dialog->setDisplayOptions(current);
        connect(canvas,
                &CanvasWidget::featurePointStatusChanged,
                dialog,
                &FeaturePointVisualizationDialog::setPointStatus);
        connect(canvas,
                &CanvasWidget::featureResidualStatusChanged,
                dialog,
                &FeaturePointVisualizationDialog::setResidualStatus);
        if (!canvas->featurePointStatusMessage().isEmpty()
            && canvas->featurePointStatusSource() == current.pointSource)
        {
            dialog->setPointStatus(canvas->featurePointStatusMessage(),
                                   canvas->featurePointStatusAvailable(),
                                   canvas->featurePointStatusCount());
        }
        if (!canvas->featureResidualStatusMessage().isEmpty())
        {
            dialog->setResidualStatus(canvas->featureResidualStatusMessage(),
                                      canvas->featureResidualStatusAvailable(),
                                      canvas->featureResidualStatusCount());
        }
    }

    connect(dialog,
            &FeaturePointVisualizationDialog::displayOptionsChanged,
            this,
            [this](const LayerRenderer::FeatureDisplayOptions &options)
    {
        emit optionsChanged(options);
        if (DialogSettingStore *store = ensureSettingStore())
        {
            store->save(optionsToJson(options));
        }
    });
    dialog->show();
}

void FeatureVisualizationController::applySavedOptions(const QJsonObject &uiSettings)
{
    const QJsonObject settings = loadSettings(uiSettings);
    if (!settings.isEmpty())
    {
        emit optionsChanged(optionsFromJson(settings));
    }
}

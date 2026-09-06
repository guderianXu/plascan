#include "BrowserDebugBridge.h"

#include "Logger.h"
#include "ProjectManager.h"
#include "TaskStatusWidget.h"
#include "TaskRuntimeService.h"
#include "WorkPanelWidget.h"

#include <QAbstractButton>
#include <QAbstractItemView>
#include <QAbstractSlider>
#include <QAction>
#include <QApplication>
#include <QBuffer>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMainWindow>
#include <QMenu>
#include <QMetaObject>
#include <QPixmap>
#include <QPointer>
#include <QProgressBar>
#include <QSet>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTabWidget>
#include <QTableView>
#include <QTimer>
#include <QToolButton>
#include <QTreeView>
#include <QWidget>

#include <algorithm>

namespace xjw::gui::runtime
{
    namespace
    {
        constexpr int MaximumRequestBytes = 256 * 1024;
        constexpr int MaximumLogEntries = 500;
        constexpr int MaximumArtifacts = 100;
        constexpr int MaximumTextLength = 512;
        constexpr int MaximumUiDepth = 20;

        QString limitedText(QString text, int maximum = MaximumTextLength)
        {
            text.replace(QLatin1Char('\0'), QChar::ReplacementCharacter);
            if (text.size() <= maximum)
            {
                return text;
            }
            return text.left(maximum) + QStringLiteral("…");
        }

        QString logLevelName(int level)
        {
            switch (level)
            {
            case Logger::Debug:
                return QStringLiteral("debug");
            case Logger::Info:
                return QStringLiteral("info");
            case Logger::Warn:
                return QStringLiteral("warning");
            case Logger::Error:
                return QStringLiteral("error");
            default:
                return QStringLiteral("unknown");
            }
        }

        QJsonObject geometrySnapshot(const QWidget* widget)
        {
            const QRect geometry = widget->geometry();
            const QPoint global = widget->mapToGlobal(QPoint(0, 0));
            return {
                {QStringLiteral("x"), geometry.x()},
                {QStringLiteral("y"), geometry.y()},
                {QStringLiteral("width"), geometry.width()},
                {QStringLiteral("height"), geometry.height()},
                {QStringLiteral("global_x"), global.x()},
                {QStringLiteral("global_y"), global.y()},
            };
        }

        void addWidgetProperties(const QWidget* widget, QJsonObject* snapshot)
        {
            if (const auto* button = qobject_cast<const QAbstractButton*>(widget))
            {
                snapshot->insert(QStringLiteral("text"), limitedText(button->text()));
                snapshot->insert(QStringLiteral("checkable"), button->isCheckable());
                snapshot->insert(QStringLiteral("checked"), button->isChecked());
            }
            else if (const auto* label = qobject_cast<const QLabel*>(widget))
            {
                snapshot->insert(QStringLiteral("text"), limitedText(label->text()));
            }
            else if (const auto* line_edit = qobject_cast<const QLineEdit*>(widget))
            {
                const bool masked = line_edit->echoMode() != QLineEdit::Normal;
                snapshot->insert(QStringLiteral("text"),
                                 masked ? QStringLiteral("<masked>") : limitedText(line_edit->text()));
                snapshot->insert(QStringLiteral("placeholder"), limitedText(line_edit->placeholderText()));
            }
            else if (const auto* combo = qobject_cast<const QComboBox*>(widget))
            {
                snapshot->insert(QStringLiteral("current_index"), combo->currentIndex());
                snapshot->insert(QStringLiteral("current_text"), limitedText(combo->currentText()));
                snapshot->insert(QStringLiteral("count"), combo->count());
            }
            else if (const auto* spin = qobject_cast<const QSpinBox*>(widget))
            {
                snapshot->insert(QStringLiteral("value"), spin->value());
                snapshot->insert(QStringLiteral("minimum"), spin->minimum());
                snapshot->insert(QStringLiteral("maximum"), spin->maximum());
            }
            else if (const auto* spin = qobject_cast<const QDoubleSpinBox*>(widget))
            {
                snapshot->insert(QStringLiteral("value"), spin->value());
                snapshot->insert(QStringLiteral("minimum"), spin->minimum());
                snapshot->insert(QStringLiteral("maximum"), spin->maximum());
            }
            else if (const auto* progress = qobject_cast<const QProgressBar*>(widget))
            {
                snapshot->insert(QStringLiteral("value"), progress->value());
                snapshot->insert(QStringLiteral("minimum"), progress->minimum());
                snapshot->insert(QStringLiteral("maximum"), progress->maximum());
            }
            else if (const auto* slider = qobject_cast<const QAbstractSlider*>(widget))
            {
                snapshot->insert(QStringLiteral("value"), slider->value());
                snapshot->insert(QStringLiteral("minimum"), slider->minimum());
                snapshot->insert(QStringLiteral("maximum"), slider->maximum());
            }
            else if (const auto* tabs = qobject_cast<const QTabWidget*>(widget))
            {
                snapshot->insert(QStringLiteral("current_index"), tabs->currentIndex());
                snapshot->insert(QStringLiteral("count"), tabs->count());
                if (tabs->currentIndex() >= 0)
                {
                    snapshot->insert(QStringLiteral("current_text"), limitedText(tabs->tabText(tabs->currentIndex())));
                }
            }

            if (const auto* view = qobject_cast<const QAbstractItemView*>(widget))
            {
                const QAbstractItemModel* model = view->model();
                snapshot->insert(QStringLiteral("row_count"), model ? model->rowCount() : 0);
                snapshot->insert(QStringLiteral("column_count"), model ? model->columnCount() : 0);
            }
        }

        QJsonObject widgetSnapshot(const QWidget* widget, int depth)
        {
            QJsonObject snapshot{
                {QStringLiteral("class"), QString::fromLatin1(widget->metaObject()->className())},
                {QStringLiteral("object_name"), widget->objectName()},
                {QStringLiteral("visible"), widget->isVisible()},
                {QStringLiteral("enabled"), widget->isEnabled()},
                {QStringLiteral("focused"), widget->hasFocus()},
                {QStringLiteral("window_title"), limitedText(widget->windowTitle())},
                {QStringLiteral("geometry"), geometrySnapshot(widget)},
            };
            addWidgetProperties(widget, &snapshot);

            QJsonArray children;
            if (depth < MaximumUiDepth)
            {
                for (QWidget* child : widget->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly))
                {
                    if (!child->objectName().isEmpty() || child->isVisible())
                    {
                        children.append(widgetSnapshot(child, depth + 1));
                    }
                }
            }
            snapshot.insert(QStringLiteral("children"), children);
            return snapshot;
        }

        QJsonObject actionSnapshot(const QAction* action)
        {
            return {
                {QStringLiteral("class"), QString::fromLatin1(action->metaObject()->className())},
                {QStringLiteral("object_name"), action->objectName()},
                {QStringLiteral("text"), limitedText(action->text())},
                {QStringLiteral("enabled"), action->isEnabled()},
                {QStringLiteral("visible"), action->isVisible()},
                {QStringLiteral("checkable"), action->isCheckable()},
                {QStringLiteral("checked"), action->isChecked()},
            };
        }

        void collectArtifacts(const QJsonValue& value, const QString& keyPath, int depth, QJsonArray* artifacts)
        {
            if (depth > 8 || artifacts->size() >= MaximumArtifacts)
            {
                return;
            }
            if (value.isObject())
            {
                const QJsonObject object = value.toObject();
                for (auto it = object.constBegin(); it != object.constEnd(); ++it)
                {
                    collectArtifacts(it.value(),
                                     keyPath.isEmpty() ? it.key() : keyPath + QLatin1Char('.') + it.key(),
                                     depth + 1,
                                     artifacts);
                }
                return;
            }
            if (value.isArray())
            {
                int index = 0;
                for (const QJsonValue& item : value.toArray())
                {
                    collectArtifacts(item, QStringLiteral("%1[%2]").arg(keyPath).arg(index++), depth + 1, artifacts);
                }
                return;
            }
            if (!value.isString())
            {
                return;
            }
            const QString lowered = keyPath.toLower();
            if (!lowered.contains(QStringLiteral("path")) && !lowered.contains(QStringLiteral("file")) &&
                !lowered.contains(QStringLiteral("output")))
            {
                return;
            }
            const QString path = value.toString().trimmed();
            if (!path.isEmpty())
            {
                artifacts->append(QJsonObject{
                    {QStringLiteral("key"), keyPath},
                    {QStringLiteral("path"), path},
                    {QStringLiteral("exists"), QFileInfo::exists(path)},
                });
            }
        }

        QJsonObject successResult(const QJsonValue& result = QJsonObject())
        {
            return {
                {QStringLiteral("ok"), true},
                {QStringLiteral("result"), result},
            };
        }

        QJsonObject errorResult(const QString& code, const QString& message)
        {
            return {
                {QStringLiteral("ok"), false},
                {QStringLiteral("error"),
                 QJsonObject{
                     {QStringLiteral("code"), code},
                     {QStringLiteral("message"), message},
                 }},
            };
        }
    } // namespace

    BrowserDebugBridge::BrowserDebugBridge(QWidget* rootWidget, QObject* parent)
        : QObject(parent), _rootWidget(rootWidget),
          _projectManager(rootWidget ? rootWidget->findChild<ProjectManager*>(QStringLiteral("ProjectManager"))
                                     : nullptr),
          _taskRuntimeService(
              rootWidget ? rootWidget->findChild<TaskRuntimeService*>(QStringLiteral("TaskRuntimeService")) : nullptr),
          _server(new QLocalServer(this))
    {
        connect(_server, &QLocalServer::newConnection, this, &BrowserDebugBridge::acceptConnections);

        QPointer<BrowserDebugBridge> self(this);
        _logSinkId = Logger::instance()->registerSink(
            [self](const Logger::Entry& entry)
            {
                if (!self)
                {
                    return;
                }
                const QString timestamp = QString::fromStdString(entry.timestamp);
                const QString message = QString::fromUtf8(entry.message.data(), static_cast<int>(entry.message.size()));
                const QString formatted =
                    QString::fromUtf8(entry.formatted.data(), static_cast<int>(entry.formatted.size()));
                const int level = static_cast<int>(entry.level);
                QMetaObject::invokeMethod(
                    self,
                    [self, level, timestamp, message, formatted]()
                    {
                        if (self)
                        {
                            self->appendLog(level, timestamp, message, formatted);
                        }
                    },
                    Qt::QueuedConnection);
            });
    }

    BrowserDebugBridge::~BrowserDebugBridge()
    {
        if (_logSinkId != 0)
        {
            Logger::instance()->unregisterSink(_logSinkId);
        }
        if (_server->isListening())
        {
            _server->close();
        }
        if (!_serverName.isEmpty())
        {
            QLocalServer::removeServer(_serverName);
        }
    }

    bool BrowserDebugBridge::start(QString* errorMessage)
    {
        if (qEnvironmentVariable("PLASCAN_BROWSER_TEST") != QStringLiteral("1"))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("PLASCAN_BROWSER_TEST is not enabled");
            }
            return false;
        }

        _serverName = qEnvironmentVariable("PLASCAN_BROWSER_DEBUG_SOCKET").trimmed();
        _token = qgetenv("PLASCAN_BROWSER_DEBUG_TOKEN").trimmed();
        if (_serverName.isEmpty() || _token.size() < 24)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("debug socket or session token is missing");
            }
            return false;
        }

        QLocalServer::removeServer(_serverName);
        _server->setSocketOptions(QLocalServer::UserAccessOption);
        if (!_server->listen(_serverName))
        {
            if (errorMessage)
            {
                *errorMessage = _server->errorString();
            }
            return false;
        }
        return true;
    }

    QString BrowserDebugBridge::serverName() const
    {
        return _serverName;
    }

    void BrowserDebugBridge::acceptConnections()
    {
        while (_server->hasPendingConnections())
        {
            QLocalSocket* socket = _server->nextPendingConnection();
            if (!socket)
            {
                continue;
            }
            _buffers.insert(socket, QByteArray());
            connect(socket, &QLocalSocket::readyRead, this, [this, socket]() { readRequest(socket); });
            connect(socket,
                    &QLocalSocket::disconnected,
                    this,
                    [this, socket]()
                    {
                        _buffers.remove(socket);
                        socket->deleteLater();
                    });
        }
    }

    void BrowserDebugBridge::readRequest(QLocalSocket* socket)
    {
        QByteArray& buffer = _buffers[socket];
        buffer.append(socket->readAll());
        if (buffer.size() > MaximumRequestBytes)
        {
            sendResponse(
                socket,
                errorResult(QStringLiteral("request_too_large"), QStringLiteral("request exceeds the 256 KiB limit")));
            socket->disconnectFromServer();
            return;
        }

        qsizetype newline = -1;
        while ((newline = buffer.indexOf('\n')) >= 0)
        {
            const QByteArray line = buffer.left(newline).trimmed();
            buffer.remove(0, newline + 1);
            if (line.isEmpty())
            {
                continue;
            }
            QJsonParseError parse_error;
            const QJsonDocument document = QJsonDocument::fromJson(line, &parse_error);
            QJsonObject response;
            if (parse_error.error != QJsonParseError::NoError || !document.isObject())
            {
                response = errorResult(QStringLiteral("invalid_json"), parse_error.errorString());
            }
            else
            {
                response = handleRequest(document.object());
            }
            sendResponse(socket, response);
        }
    }

    void BrowserDebugBridge::sendResponse(QLocalSocket* socket, const QJsonObject& response)
    {
        socket->write(QJsonDocument(response).toJson(QJsonDocument::Compact));
        socket->write("\n");
        socket->flush();
    }

    QJsonObject BrowserDebugBridge::handleRequest(const QJsonObject& request)
    {
        QJsonObject response;
        if (request.value(QStringLiteral("token")).toString().toUtf8() != _token)
        {
            response = errorResult(QStringLiteral("unauthorized"), QStringLiteral("invalid session token"));
        }
        else
        {
            const QString method = request.value(QStringLiteral("method")).toString();
            const QJsonObject parameters = request.value(QStringLiteral("params")).toObject();
            if (method == QStringLiteral("ping"))
            {
                response = successResult(QJsonObject{
                    {QStringLiteral("schema_version"), 1},
                    {QStringLiteral("pid"), QCoreApplication::applicationPid()},
                });
            }
            else if (method == QStringLiteral("snapshot"))
            {
                response = successResult(snapshot());
            }
            else if (method == QStringLiteral("ui_tree"))
            {
                response = successResult(windowSnapshots());
            }
            else if (method == QStringLiteral("logs"))
            {
                response = successResult(logSnapshots());
            }
            else if (method == QStringLiteral("screenshot"))
            {
                response = successResult(screenshot());
            }
            else if (method == QStringLiteral("interact"))
            {
                response = interact(parameters);
            }
            else if (method == QStringLiteral("close_dialog"))
            {
                response = closeDialog();
            }
            else if (method == QStringLiteral("task_command"))
            {
                response = taskCommand(parameters);
            }
            else
            {
                response = errorResult(QStringLiteral("unknown_method"), QStringLiteral("method is not allow-listed"));
            }
        }
        if (request.contains(QStringLiteral("id")))
        {
            response.insert(QStringLiteral("id"), request.value(QStringLiteral("id")));
        }
        return response;
    }

    QJsonObject BrowserDebugBridge::snapshot() const
    {
        QWidget* active_window = QApplication::activeWindow();
        QWidget* modal_widget = QApplication::activeModalWidget();
        QString recent_error;
        for (qsizetype index = _logs.size(); index > 0; --index)
        {
            const QJsonObject entry = _logs.at(index - 1).toObject();
            const QString level = entry.value(QStringLiteral("level")).toString();
            if (level == QStringLiteral("error") || level == QStringLiteral("warning"))
            {
                recent_error = entry.value(QStringLiteral("message")).toString();
                break;
            }
        }

        return {
            {QStringLiteral("schema_version"), 1},
            {QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
            {QStringLiteral("application"),
             QJsonObject{
                 {QStringLiteral("name"), QCoreApplication::applicationName()},
                 {QStringLiteral("version"), QCoreApplication::applicationVersion()},
                 {QStringLiteral("pid"), QCoreApplication::applicationPid()},
                 {QStringLiteral("active_window"), active_window ? active_window->windowTitle() : QString()},
                 {QStringLiteral("modal_window"), modal_widget ? modal_widget->windowTitle() : QString()},
             }},
            {QStringLiteral("project"), projectSnapshot()},
            {QStringLiteral("tasks"), taskSnapshots()},
            {QStringLiteral("recent_error"), recent_error},
            {QStringLiteral("logs"), logSnapshots()},
            {QStringLiteral("windows"), windowSnapshots()},
        };
    }

    QJsonObject BrowserDebugBridge::projectSnapshot() const
    {
        if (!_projectManager)
        {
            return {{QStringLiteral("available"), false}};
        }
        const auto session = _projectManager->currentSessionContext();
        const QJsonObject meta = _projectManager->currentMeta();
        QJsonArray artifacts;
        collectArtifacts(meta, QString(), 0, &artifacts);
        return {
            {QStringLiteral("available"), true},
            {QStringLiteral("open"), !session.projectPath.isEmpty()},
            {QStringLiteral("path"), session.projectPath},
            {QStringLiteral("chunk_id"), session.chunkId},
            {QStringLiteral("generation"), static_cast<double>(session.generation)},
            {QStringLiteral("dirty"), _projectManager->isDirty()},
            {QStringLiteral("image_count"), _projectManager->getAllImages().size()},
            {QStringLiteral("artifacts"), artifacts},
        };
    }

    QJsonArray BrowserDebugBridge::taskSnapshots() const
    {
        if (_rootWidget)
        {
            if (const auto* work_panel = _rootWidget->findChild<WorkPanelWidget*>(QStringLiteral("workPanel")))
            {
                return work_panel->taskSnapshots();
            }
        }
        QJsonArray tasks;
        if (!_rootWidget)
        {
            return tasks;
        }
        for (const TaskStatusWidget* status : _rootWidget->findChildren<TaskStatusWidget*>())
        {
            if (!status->isActive() && !status->isCancelling())
            {
                continue;
            }
            tasks.append(QJsonObject{
                {QStringLiteral("object_name"), status->objectName()},
                {QStringLiteral("status_text"), status->statusText()},
                {QStringLiteral("detail_text"), status->detailText()},
                {QStringLiteral("active"), status->isActive()},
                {QStringLiteral("cancelling"), status->isCancelling()},
                {QStringLiteral("progress_value"), status->progressValue()},
                {QStringLiteral("progress_maximum"), status->progressMaximum()},
                {QStringLiteral("elapsed_ms"), static_cast<double>(status->elapsedMilliseconds())},
            });
        }
        return tasks;
    }

    QJsonObject BrowserDebugBridge::taskCommand(const QJsonObject& parameters)
    {
        if (!_taskRuntimeService)
        {
            return errorResult(QStringLiteral("task_runtime_unavailable"),
                               QStringLiteral("task runtime service is unavailable"));
        }
        const QString action = parameters.value(QStringLiteral("action")).toString().trimmed();
        const QString run_id = parameters.value(QStringLiteral("run_id")).toString().trimmed();
        if (action.isEmpty() || run_id.isEmpty())
        {
            return errorResult(QStringLiteral("invalid_parameters"), QStringLiteral("action and run_id are required"));
        }
        const QJsonObject result =
            _taskRuntimeService->command(action,
                                         run_id,
                                         parameters.value(QStringLiteral("reference_run_id")).toString(),
                                         parameters.value(QStringLiteral("priority")).toInt(),
                                         parameters.value(QStringLiteral("revision")).toVariant().toULongLong());
        if (!result.value(QStringLiteral("accepted")).toBool(false))
        {
            return errorResult(QStringLiteral("task_command_rejected"),
                               result.value(QStringLiteral("error")).toString());
        }
        return successResult(result);
    }

    QJsonArray BrowserDebugBridge::windowSnapshots() const
    {
        QJsonArray windows;
        for (QWidget* window : QApplication::topLevelWidgets())
        {
            if (!window->isVisible() && window != _rootWidget)
            {
                continue;
            }
            QJsonObject window_snapshot = widgetSnapshot(window, 0);
            window_snapshot.insert(QStringLiteral("modal"), window->isModal());
            window_snapshot.insert(QStringLiteral("active"), window->isActiveWindow());
            QJsonArray actions;
            for (QAction* action : window->findChildren<QAction*>())
            {
                if (!action->objectName().isEmpty())
                {
                    actions.append(actionSnapshot(action));
                }
            }
            window_snapshot.insert(QStringLiteral("actions"), actions);
            windows.append(window_snapshot);
        }
        return windows;
    }

    QJsonArray BrowserDebugBridge::logSnapshots() const
    {
        return _logs;
    }

    QJsonObject BrowserDebugBridge::screenshot() const
    {
        QWidget* target = QApplication::activeModalWidget();
        if (!target)
        {
            target = QApplication::activeWindow();
        }
        if (!target)
        {
            target = _rootWidget;
        }
        if (!target)
        {
            return {{QStringLiteral("available"), false}};
        }
        const QPixmap pixmap = target->grab();
        QByteArray bytes;
        QBuffer buffer(&bytes);
        buffer.open(QIODevice::WriteOnly);
        const bool saved = pixmap.save(&buffer, "PNG");
        return {
            {QStringLiteral("available"), saved},
            {QStringLiteral("format"), QStringLiteral("png")},
            {QStringLiteral("width"), pixmap.width()},
            {QStringLiteral("height"), pixmap.height()},
            {QStringLiteral("data_base64"), saved ? QString::fromLatin1(bytes.toBase64()) : QString()},
        };
    }

    QJsonObject BrowserDebugBridge::interact(const QJsonObject& parameters)
    {
        const QString object_name = parameters.value(QStringLiteral("object_name")).toString().trimmed();
        const QString operation = parameters.value(QStringLiteral("operation")).toString().trimmed();
        if (object_name.isEmpty())
        {
            return errorResult(QStringLiteral("invalid_parameters"), QStringLiteral("object_name is required"));
        }
        const QList<QObject*> matches = namedObjects(object_name);
        if (matches.size() != 1)
        {
            return errorResult(matches.isEmpty() ? QStringLiteral("object_not_found")
                                                 : QStringLiteral("ambiguous_object"),
                               QStringLiteral("object_name matched %1 objects").arg(matches.size()));
        }
        QObject* object = matches.constFirst();

        if (operation == QStringLiteral("activate"))
        {
            if (auto* action = qobject_cast<QAction*>(object))
            {
                if (!action->isEnabled())
                {
                    return errorResult(QStringLiteral("disabled"), QStringLiteral("action is disabled"));
                }
                QTimer::singleShot(0, action, &QAction::trigger);
                return successResult(QJsonObject{{QStringLiteral("scheduled"), true}});
            }
            if (auto* button = qobject_cast<QAbstractButton*>(object))
            {
                if (!button->isEnabled() || !button->isVisible())
                {
                    return errorResult(QStringLiteral("disabled"), QStringLiteral("button is disabled or hidden"));
                }
                QTimer::singleShot(0, button, &QAbstractButton::click);
                return successResult(QJsonObject{{QStringLiteral("scheduled"), true}});
            }
        }
        else if (operation == QStringLiteral("focus"))
        {
            if (auto* widget = qobject_cast<QWidget*>(object))
            {
                QTimer::singleShot(0, widget, [widget]() { widget->setFocus(Qt::OtherFocusReason); });
                return successResult(QJsonObject{{QStringLiteral("scheduled"), true}});
            }
        }
        else if (operation == QStringLiteral("set_text"))
        {
            if (auto* line_edit = qobject_cast<QLineEdit*>(object))
            {
                line_edit->setText(parameters.value(QStringLiteral("value")).toString());
                return successResult();
            }
        }
        else if (operation == QStringLiteral("set_value"))
        {
            const double value = parameters.value(QStringLiteral("value")).toDouble();
            if (auto* spin = qobject_cast<QSpinBox*>(object))
            {
                spin->setValue(static_cast<int>(value));
                return successResult();
            }
            if (auto* spin = qobject_cast<QDoubleSpinBox*>(object))
            {
                spin->setValue(value);
                return successResult();
            }
            if (auto* slider = qobject_cast<QAbstractSlider*>(object))
            {
                slider->setValue(static_cast<int>(value));
                return successResult();
            }
        }
        else if (operation == QStringLiteral("set_checked"))
        {
            const bool checked = parameters.value(QStringLiteral("value")).toBool();
            if (auto* button = qobject_cast<QAbstractButton*>(object); button && button->isCheckable())
            {
                button->setChecked(checked);
                return successResult();
            }
            if (auto* action = qobject_cast<QAction*>(object); action && action->isCheckable())
            {
                action->setChecked(checked);
                return successResult();
            }
        }
        else if (operation == QStringLiteral("select_index"))
        {
            const int index = parameters.value(QStringLiteral("value")).toInt(-1);
            if (auto* combo = qobject_cast<QComboBox*>(object); index >= 0 && index < combo->count())
            {
                combo->setCurrentIndex(index);
                return successResult();
            }
            if (auto* tabs = qobject_cast<QTabWidget*>(object); index >= 0 && index < tabs->count())
            {
                tabs->setCurrentIndex(index);
                return successResult();
            }
            if (auto* stack = qobject_cast<QStackedWidget*>(object); index >= 0 && index < stack->count())
            {
                stack->setCurrentIndex(index);
                return successResult();
            }
        }
        else if (operation == QStringLiteral("cancel_task"))
        {
            if (auto* status = qobject_cast<TaskStatusWidget*>(object);
                status && status->isActive() && !status->isCancelling())
            {
                auto* button = status->findChild<QToolButton*>(QStringLiteral("cancelButton"));
                if (button && button->isEnabled())
                {
                    QTimer::singleShot(0, button, &QAbstractButton::click);
                    return successResult(QJsonObject{{QStringLiteral("scheduled"), true}});
                }
            }
        }

        return errorResult(QStringLiteral("unsupported_operation"),
                           QStringLiteral("operation is not allowed for this object type"));
    }

    QJsonObject BrowserDebugBridge::closeDialog()
    {
        QDialog* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (!dialog)
        {
            for (QWidget* window : QApplication::topLevelWidgets())
            {
                if (window->isVisible())
                {
                    dialog = qobject_cast<QDialog*>(window);
                    if (dialog)
                    {
                        break;
                    }
                }
            }
        }
        if (!dialog)
        {
            return errorResult(QStringLiteral("dialog_not_found"), QStringLiteral("no visible dialog is active"));
        }
        const QString title = dialog->windowTitle();
        QTimer::singleShot(0, dialog, &QDialog::reject);
        return successResult(QJsonObject{
            {QStringLiteral("scheduled"), true},
            {QStringLiteral("window_title"), title},
        });
    }

    QList<QObject*> BrowserDebugBridge::namedObjects(const QString& objectName) const
    {
        QList<QObject*> matches;
        QSet<QObject*> seen;
        for (QWidget* window : QApplication::topLevelWidgets())
        {
            if (window->objectName() == objectName && !seen.contains(window))
            {
                matches.append(window);
                seen.insert(window);
            }
            for (QObject* object : window->findChildren<QObject*>(objectName))
            {
                if (!seen.contains(object))
                {
                    matches.append(object);
                    seen.insert(object);
                }
            }
        }
        return matches;
    }

    void
    BrowserDebugBridge::appendLog(int level, const QString& timestamp, const QString& message, const QString& formatted)
    {
        _logs.append(QJsonObject{
            {QStringLiteral("level"), logLevelName(level)},
            {QStringLiteral("timestamp"), timestamp},
            {QStringLiteral("message"), limitedText(message, 2048)},
            {QStringLiteral("formatted"), limitedText(formatted, 4096)},
        });
        while (_logs.size() > MaximumLogEntries)
        {
            _logs.removeAt(0);
        }
    }

} // namespace xjw::gui::runtime

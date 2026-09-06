#pragma once

#include <QByteArray>
#include <QHash>
#include <QJsonArray>
#include <QObject>
#include <QPointer>

class QJsonObject;
class QLocalServer;
class QLocalSocket;
class ProjectManager;
class QWidget;

namespace xjw::gui::runtime
{
    class TaskRuntimeService;
}

namespace xjw::gui::runtime
{

    class BrowserDebugBridge final : public QObject
    {
        Q_OBJECT

    public:
        explicit BrowserDebugBridge(QWidget* rootWidget, QObject* parent = nullptr);
        ~BrowserDebugBridge() override;

        bool start(QString* errorMessage = nullptr);
        QString serverName() const;

    private:
        void acceptConnections();
        void readRequest(QLocalSocket* socket);
        void sendResponse(QLocalSocket* socket, const QJsonObject& response);
        QJsonObject handleRequest(const QJsonObject& request);
        QJsonObject snapshot() const;
        QJsonObject projectSnapshot() const;
        QJsonArray taskSnapshots() const;
        QJsonArray windowSnapshots() const;
        QJsonArray logSnapshots() const;
        QJsonObject screenshot() const;
        QJsonObject interact(const QJsonObject& parameters);
        QJsonObject closeDialog();
        QJsonObject taskCommand(const QJsonObject& parameters);
        QList<QObject*> namedObjects(const QString& objectName) const;
        void appendLog(int level, const QString& timestamp, const QString& message, const QString& formatted);

        QPointer<QWidget> _rootWidget;
        QPointer<ProjectManager> _projectManager;
        QPointer<TaskRuntimeService> _taskRuntimeService;
        QLocalServer* _server{};
        QHash<QLocalSocket*, QByteArray> _buffers;
        QJsonArray _logs;
        QByteArray _token;
        QString _serverName;
        int _logSinkId{};
    };

} // namespace xjw::gui::runtime

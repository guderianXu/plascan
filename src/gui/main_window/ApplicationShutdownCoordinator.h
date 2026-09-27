#pragma once

#include <QObject>
#include <QString>
#include <QTimer>

#include <functional>

namespace xjw::gui::main_window
{

    class ApplicationShutdownCoordinator final : public QObject
    {
        Q_OBJECT

    public:
        struct Hooks
        {
            std::function<void()> requestResourceCancellation;
            std::function<bool()> projectOperationsIdle;
            std::function<bool()> tryAcquireShutdownGate;
            std::function<bool()> projectTaskDrainInProgress;
            std::function<bool(std::function<void()>)> startProjectTaskDrain;
            std::function<void()> requestRuntimeShutdown;
            std::function<bool()> runtimeShutdownComplete;
        };

        explicit ApplicationShutdownCoordinator(Hooks hooks, QObject* parent = nullptr);

        bool requestShutdown();
        bool isShutdownInProgress() const noexcept;
        bool isReadyToClose() const noexcept;

    signals:
        void progressChanged(const QString& message);
        void readyToClose();

    private:
        void schedulePoll(int delayMs = 15);
        void poll();
        void setStage(const QString& message);

        Hooks _hooks;
        QTimer _pollTimer;
        QString _stage;
        bool _shutdownInProgress = false;
        bool _shutdownGateAcquired = false;
        bool _taskDrainStarted = false;
        bool _taskDrainFinished = false;
        bool _runtimeShutdownRequested = false;
        bool _readyToClose = false;
    };

} // namespace xjw::gui::main_window

#pragma once

#include <QJsonObject>

#include <atomic>
#include <memory>

class DenseMatchRunner
{
public:
    static void run(const QJsonObject &settings,
                    std::shared_ptr<std::atomic<int>> progress = nullptr,
                    std::shared_ptr<std::atomic<bool>> cancelFlag = nullptr);
};

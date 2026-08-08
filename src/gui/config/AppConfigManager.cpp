/**
 * @file AppConfigManager.cpp
 * @brief AppConfigManager 的实现文件。
 *
 * 本文件仅包含构造函数，所有实际逻辑均由各子管理器负责实现。
 */
#include "AppConfigManager.h"

/**
 * @brief 构造函数。
 *
 * 三个轻量子管理器按值持有，与 AppConfigManager 共享确定的对象生命周期。
 */
AppConfigManager::AppConfigManager(QObject *parent)
    : QObject(parent)
{
}

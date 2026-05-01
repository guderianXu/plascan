#pragma once

/**
 * @file GlobalSettings.h
 * @brief 全局设置聚合管理器声明。
 *
 * GlobalSettings 提供统一的注册点，用于托管多个可持久化的
 * 子设置对象（例如不同对话框或模块的记忆化类）。
 *
 * 设计目标：
 *  - 每个子设置实现 `IGlobalSetting` 接口并提供一个唯一 id（作为 QSettings 的 group）。
 *  - GlobalSettings 负责遍历已注册的子设置，按 group 调用其 load/save 方法。
 */

#include <QObject>
#include <QString>
#include <QList>

class QSettings;

/**
 * @brief 子设置接口抽象类。
 *
 * 继承自 QObject 以便使用 Qt 对象树管理生命周期并支持信号槽。
 */
class IGlobalSetting : public QObject
{
	Q_OBJECT
public:
	explicit IGlobalSetting(QObject *parent = nullptr) : QObject(parent) {}
	~IGlobalSetting() override = default;

	/**
	 * @brief 返回此子设置在 QSettings 中使用的组名（唯一标识）。
	 */
	virtual QString id() const = 0;

	/**
	 * @brief 从给定的 QSettings 中读取并恢复状态。
	 * @param settings QSettings 引用，调用方已切入到对应的 group（或可直接使用）。
	 */
	virtual void load(QSettings &settings) = 0;

	/**
	 * @brief 将当前状态写入到给定的 QSettings 中。
	 * @param settings QSettings 引用，调用方已切入到对应的 group（或可直接使用）。
	 */
	virtual void save(QSettings &settings) const = 0;
};

/**
 * @class GlobalSettings
 * @brief 聚合并管理所有实现了 IGlobalSetting 的子设置对象。
 *
 * 提供注册/注销、以及一次性 loadAll()/saveAll() 方法以便在应用
 * 启动/退出或项目打开/关闭时统一调用。
 */
class GlobalSettings : public QObject
{
	Q_OBJECT
public:
	explicit GlobalSettings(QObject *parent = nullptr);

	/**
	 * @brief 注册一个子设置对象（若已注册则忽略）。
	 * @param setting 指向派生自 IGlobalSetting 的对象，所有权仍由调用方或 Qt 对象树管理，
	 *                本函数会将其父对象设置为 GlobalSettings 以便统一管理。
	 */
	void registerSetting(IGlobalSetting *setting);

	/** @brief 取消注册一个子设置对象（不删除对象，仅从管理列表移除）。 */
	void unregisterSetting(IGlobalSetting *setting);

	/** @brief 加载所有已注册子设置的数据（从 QSettings）。 */
	void loadAll();

	/** @brief 保存所有已注册子设置的数据（到 QSettings）。 */
	void saveAll() const;

	/** @brief 按 id 加载单个子设置（若存在）。 */
	void loadById(const QString &id);

	/** @brief 按 id 保存单个子设置（若存在）。 */
	void saveById(const QString &id) const;

private:
	QList<IGlobalSetting *> m_settings;
};


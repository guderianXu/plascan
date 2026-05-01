/**
 * @file GlobalSettings.cpp
 * @brief GlobalSettings 实现文件。
 *
 * 本文件实现了对 IGlobalSetting 对象的注册、按组调用 QSettings 加载/保存等逻辑。
 */

#include "GlobalSettings.h"

#include <QSettings>

GlobalSettings::GlobalSettings(QObject *parent)
	: QObject(parent)
{
}

void GlobalSettings::registerSetting(IGlobalSetting *setting)
{
	if (!setting) return;
	if (m_settings.contains(setting)) return;
	// 将父对象设置为本管理器，方便 Qt 自动管理生命周期（若调用方不需保持所有权）
	setting->setParent(this);
	m_settings.append(setting);
}

void GlobalSettings::unregisterSetting(IGlobalSetting *setting)
{
	if (!setting) return;
	m_settings.removeAll(setting);
	// 不删除对象，仅解除本管理器的父对象引用（保留对象由调用方管理）
	if (setting->parent() == this) setting->setParent(nullptr);
}

void GlobalSettings::loadAll()
{
	QSettings settings("PlaScan", "plascan_gui");
	for (IGlobalSetting *s : qAsConst(m_settings)) {
		if (!s) continue;
		settings.beginGroup(s->id());
		s->load(settings);
		settings.endGroup();
	}
}

void GlobalSettings::saveAll() const
{
	QSettings settings("PlaScan", "plascan_gui");
	for (IGlobalSetting *s : qAsConst(m_settings)) {
		if (!s) continue;
		settings.beginGroup(s->id());
		s->save(settings);
		settings.endGroup();
	}
	// 确保立即写回磁盘
	settings.sync();
}

void GlobalSettings::loadById(const QString &id)
{
	if (id.isEmpty()) return;
	for (IGlobalSetting *s : qAsConst(m_settings)) {
		if (!s) continue;
		if (s->id() == id) {
			QSettings settings("PlaScan", "plascan_gui");
			settings.beginGroup(id);
			s->load(settings);
			settings.endGroup();
			break;
		}
	}
}

void GlobalSettings::saveById(const QString &id) const
{
	if (id.isEmpty()) return;
	for (IGlobalSetting *s : qAsConst(m_settings)) {
		if (!s) continue;
		if (s->id() == id) {
			QSettings settings("PlaScan", "plascan_gui");
			settings.beginGroup(id);
			s->save(settings);
			settings.endGroup();
			settings.sync();
			break;
		}
	}
}


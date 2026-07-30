#pragma once

#include <QDialog>
#include <QProcessEnvironment>
#include <QString>

class AboutDialog : public QDialog
{
public:
    explicit AboutDialog(QWidget *parent = nullptr);
    explicit AboutDialog(const QProcessEnvironment &environment, QWidget *parent = nullptr);

    static QString pythonEnvironmentPath();
    static QString pythonEnvironmentPath(const QProcessEnvironment &environment);
    static QString pythonEnvironmentDisplayText(const QProcessEnvironment &environment);
};

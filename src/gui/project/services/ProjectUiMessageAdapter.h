#pragma once

#include <QDir>
#include <QString>
#include <QStringList>

class QWidget;

enum class UiAnswer
{
    Yes,
    No,
    Cancel
};

enum class UiReviewDecision
{
    Accept,
    Discard
};

struct UiReviewDialogRequest
{
    QString objectName;
    QString title;
    QString text;
    QString informativeText;
    QString detailedText;
    QString acceptText;
    QString discardText;
    QString acceptObjectName;
    QString discardObjectName;
    bool warningIcon = false;
};

struct UiDialogResult
{
    bool accepted = false;
    QString text;
    QStringList texts;
    double number = 0.0;
};

class ProjectUiMessageAdapter
{
public:
    virtual ~ProjectUiMessageAdapter() = default;

    virtual void information(QWidget* parent, const QString& title, const QString& text) = 0;
    virtual void warning(QWidget* parent, const QString& title, const QString& text) = 0;
    virtual void critical(QWidget* parent, const QString& title, const QString& text) = 0;
    virtual UiAnswer question(QWidget* parent, const QString& title, const QString& text, UiAnswer defaultAnswer) = 0;
    virtual UiReviewDecision review(QWidget* parent, const UiReviewDialogRequest& request)
    {
        return question(parent, request.title, request.text, UiAnswer::No) == UiAnswer::Yes ? UiReviewDecision::Accept
                                                                                            : UiReviewDecision::Discard;
    }
    virtual UiDialogResult
    getText(QWidget* parent, const QString& title, const QString& label, const QString& initial) = 0;
    virtual UiDialogResult getDouble(QWidget* parent,
                                     const QString& title,
                                     const QString& label,
                                     double initial,
                                     double minimum,
                                     double maximum,
                                     int decimals) = 0;
    virtual UiDialogResult
    getItem(QWidget* parent, const QString& title, const QString& label, const QStringList& items, int current) = 0;
    virtual UiDialogResult selectOpenFile(QWidget* parent,
                                          const QString& title,
                                          const QString& directory,
                                          const QString& filter,
                                          QDir::Filters directoryFilters) = 0;
    virtual UiDialogResult selectOpenFiles(QWidget* parent,
                                           const QString& title,
                                           const QString& directory,
                                           const QString& filter,
                                           QDir::Filters directoryFilters) = 0;
    virtual UiDialogResult selectDirectory(QWidget* parent,
                                           const QString& title,
                                           const QString& directory,
                                           QDir::Filters directoryFilters) = 0;
    virtual UiDialogResult selectSaveFile(QWidget* parent,
                                          const QString& title,
                                          const QString& directory,
                                          const QString& filter,
                                          QDir::Filters directoryFilters,
                                          const QString& defaultSuffix) = 0;
};

class QtProjectUiMessageAdapter final : public ProjectUiMessageAdapter
{
public:
    explicit QtProjectUiMessageAdapter(QWidget* parentWidget = nullptr);

    void information(QWidget* parent, const QString& title, const QString& text) override;
    void warning(QWidget* parent, const QString& title, const QString& text) override;
    void critical(QWidget* parent, const QString& title, const QString& text) override;
    UiAnswer question(QWidget* parent, const QString& title, const QString& text, UiAnswer defaultAnswer) override;
    UiReviewDecision review(QWidget* parent, const UiReviewDialogRequest& request) override;
    UiDialogResult
    getText(QWidget* parent, const QString& title, const QString& label, const QString& initial) override;
    UiDialogResult getDouble(QWidget* parent,
                             const QString& title,
                             const QString& label,
                             double initial,
                             double minimum,
                             double maximum,
                             int decimals) override;
    UiDialogResult getItem(
        QWidget* parent, const QString& title, const QString& label, const QStringList& items, int current) override;
    UiDialogResult selectOpenFile(QWidget* parent,
                                  const QString& title,
                                  const QString& directory,
                                  const QString& filter,
                                  QDir::Filters directoryFilters) override;
    UiDialogResult selectOpenFiles(QWidget* parent,
                                   const QString& title,
                                   const QString& directory,
                                   const QString& filter,
                                   QDir::Filters directoryFilters) override;
    UiDialogResult selectDirectory(QWidget* parent,
                                   const QString& title,
                                   const QString& directory,
                                   QDir::Filters directoryFilters) override;
    UiDialogResult selectSaveFile(QWidget* parent,
                                  const QString& title,
                                  const QString& directory,
                                  const QString& filter,
                                  QDir::Filters directoryFilters,
                                  const QString& defaultSuffix) override;

private:
    QWidget* resolveParent(QWidget* parent) const;
    QWidget* _parentWidget = nullptr;
};

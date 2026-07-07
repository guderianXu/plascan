#include "AboutDialog.h"

#include <QDialogButtonBox>
#include <QFont>
#include <QFormLayout>
#include <QLabel>
#include <QVBoxLayout>

AboutDialog::AboutDialog(QWidget *parent)
    : AboutDialog(QProcessEnvironment::systemEnvironment(), parent)
{
}

AboutDialog::AboutDialog(const QProcessEnvironment &environment, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("关于 PlaScan"));
    setModal(true);
    setMinimumWidth(560);

    auto *rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(18, 18, 18, 14);
    rootLayout->setSpacing(12);

    auto *titleLabel = new QLabel(tr("PlaScan"), this);
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(titleFont.pointSize() + 4);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    rootLayout->addWidget(titleLabel);

    auto *summaryLabel = new QLabel(tr("行星表面摄影测量处理系统"), this);
    summaryLabel->setObjectName(QStringLiteral("aboutSummaryLabel"));
    rootLayout->addWidget(summaryLabel);

    auto *formLayout = new QFormLayout;
    formLayout->setLabelAlignment(Qt::AlignRight | Qt::AlignTop);
    formLayout->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
    formLayout->setHorizontalSpacing(12);
    formLayout->setVerticalSpacing(8);
    rootLayout->addLayout(formLayout);

    auto *pythonValueLabel = new QLabel(pythonEnvironmentDisplayText(environment), this);
    pythonValueLabel->setObjectName(QStringLiteral("pythonEnvironmentValueLabel"));
    pythonValueLabel->setWordWrap(true);
    pythonValueLabel->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
    formLayout->addRow(tr("Python 环境:"), pythonValueLabel);

    auto *hintLabel = new QLabel(tr("优先使用 PLASCAN_PYTHON_EXECUTABLE，其次使用 PLASCAN_PYTHON；"
                                    "未配置时使用项目根目录 .venv。"),
                                  this);
    hintLabel->setObjectName(QStringLiteral("pythonEnvironmentHintLabel"));
    hintLabel->setWordWrap(true);
    rootLayout->addWidget(hintLabel);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    rootLayout->addWidget(buttons);
}

QString AboutDialog::pythonEnvironmentPath()
{
    return pythonEnvironmentPath(QProcessEnvironment::systemEnvironment());
}

QString AboutDialog::pythonEnvironmentPath(const QProcessEnvironment &environment)
{
    const QString explicitPath = environment.value(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE")).trimmed();
    if (!explicitPath.isEmpty())
    {
        return explicitPath;
    }

    return environment.value(QStringLiteral("PLASCAN_PYTHON")).trimmed();
}

QString AboutDialog::pythonEnvironmentDisplayText(const QProcessEnvironment &environment)
{
    const QString pythonPath = pythonEnvironmentPath(environment);
    if (!pythonPath.isEmpty())
    {
        return pythonPath;
    }

    return tr("未配置（请运行 scripts/env/setup_python_runtime.py 并通过开发环境启动 PlaScan）");
}

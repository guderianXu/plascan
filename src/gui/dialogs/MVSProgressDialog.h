// =============================================================================
// 文件: MVSProgressDialog.h
// 模块: GUI / Dialogs
// 说明:
//   MVS 深度图生成流水线的进度对话框。
//
//   显示内容:
//     - 当前阶段文字（如"预处理稀疏点云…"、"生成深度图 3/12…"）
//     - QProgressBar（0 ~ 100）
//     - 已用时（每秒更新）
//     - 取消按钮
//
//   使用方式:
//     1. 创建 MVSProgressDialog 并连接 DepthMapGenerator 的信号；
//     2. 调用 exec() 或 open() 模态显示；
//     3. 用户点击取消 → emits cancelled() → 调用方调用 gen->cancel()
// =============================================================================
#pragma once

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QElapsedTimer>

namespace xjw {

// =============================================================================
// MVSProgressDialog
// =============================================================================
class MVSProgressDialog : public QDialog
{
    Q_OBJECT
public:
    explicit MVSProgressDialog(QWidget *parent = nullptr,
                               Qt::WindowFlags f = Qt::Dialog);

    // -------------------------------------------------------------------------
    // 设置总任务步数（用于在标题中显示 X/N）
    // -------------------------------------------------------------------------
    void setTotalSteps(int total);

public slots:
    // ─────────────────────────────────────────────────────────────────────────
    // 连接 DepthMapGenerator::progressChanged(QString stage, float progress)
    //   progress: [0, 1]
    // ─────────────────────────────────────────────────────────────────────────
    void onProgress(const QString &stage, float progress);

    // ─────────────────────────────────────────────────────────────────────────
    // 连接 DepthMapGenerator::errorOccurred(QString)
    // ─────────────────────────────────────────────────────────────────────────
    void onError(const QString &error);

    // ─────────────────────────────────────────────────────────────────────────
    // 连接 DepthMapGenerator::finished(bool success)
    // ─────────────────────────────────────────────────────────────────────────
    void onFinished(bool success);

signals:
    /// 用户点击取消按钮时发射，调用方应调用 generator->cancel()
    void cancelled();

private slots:
    void updateElapsed();
    void onCancelClicked();

private:
    void setupUi();

    QLabel       *m_stageLabel   = nullptr;
    QProgressBar *m_progressBar  = nullptr;
    QLabel       *m_elapsedLabel = nullptr;
    QPushButton  *m_cancelBtn    = nullptr;

    QTimer        m_timer;
    QElapsedTimer m_elapsed;

    int  m_totalSteps = 0;
    bool m_finished   = false;
};

} // namespace xjw

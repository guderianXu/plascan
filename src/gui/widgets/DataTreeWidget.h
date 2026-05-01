#pragma once

#include <QWidget>
#include <QJsonObject>
#include <QStandardItem>

class QTreeView;
class QStandardItemModel;

// DataTreeWidget: 显示项目中的资源（影像/图层等）的树状视图
// 设计要点：
// - 仅负责视图显示与用户交互（选择/右键菜单），不做磁盘 I/O 或归档写操作
// - 提供 loadFromArchive(plascanPath) 方法，从 .plascan 或回退的磁盘文件读取 project_files.json 并显示
// - 通过信号把用户操作（打开/在文件管理器中显示/打包/移除引用）传递给上层组件（例如 MainWindow / ProjectManager）
class DataTreeWidget : public QWidget
{
    Q_OBJECT
public:
    explicit DataTreeWidget(QWidget *parent = nullptr);
    ~DataTreeWidget() override;

    // 从指定的 .plascan（归档文件路径）加载资源索引并刷新视图
    void loadFromArchive(const QString &plascanPath);
public slots:
    // 直接从内存提供的 JSON 元数据刷新视图（由 ProjectManager 在修改后立即调用）
    void loadFromJson(const QJsonObject &meta);

signals:
    // 用户在右键菜单选择打开时发出（传递被选中资源的路径，可能为相对或绝对）
    void openRequested(const QString &resourcePath);
    // 在文件管理器中显示（reveal）
    void revealRequested(const QString &resourcePath);
    // 将引用打包到归档（由上层实现）
    void packRequested(const QString &resourcePath);
    // 从项目中移除引用（由上层执行实际移除动作并保存元数据）
    // 支持多选：传递被选中的资源路径列表以便批量删除
    void removeRequested(const QStringList &resourcePaths);
    // 删除生成数据（从项目元数据中移除，并删除关联生成文件）
    void deleteDataRequested(const QString &section, const QStringList &resourcePaths);
    // 在工作区中心的侧边打开二维图像，用于双图对比
    void sideOpenRequested(const QString &section, const QString &resourcePath);

    // 用户在列表中“激活/点击”某个影像（单选）时发出。
    // 上层可以据此把该影像显示到中央画布。
    void imageActivated(const QString &resourcePath);
    // 用户点击任意资源项时发出（含所属分组），用于模型/点云联动显示。
    void resourceActivated(const QString &section, const QString &resourcePath);

private slots:
    void onContextMenuRequested(const QPoint &pos);

private:
    void populateFromMeta(const QJsonObject &meta);
    QJsonObject normalizeMeta(const QJsonObject &meta) const;
    QStandardItem *createSection(const QString &title, int count);
    void appendItemRow(QStandardItem *parent, const QString &name, const QString &path, const QString &storage);

    QTreeView *m_view{};
    QStandardItemModel *m_model{};
    QString m_currentPlascanPath{};
};

#pragma once

#include <QWidget>
#include <QJsonObject>
#include <QStandardItem>
#include <QStringList>

class QTreeView;
class QStandardItemModel;

// DataTreeWidget: 显示项目中的资源（影像/图层等）的树状视图
// 设计要点：
// - 仅负责视图显示与用户交互（选择/右键菜单），不做磁盘 I/O 或归档写操作
// - 通过 loadFromJson(meta) 消费 ProjectData 发出的内存元数据快照
// - 通过信号把用户操作（打开/在文件管理器中显示/打包/移除引用）传递给上层组件（例如 MainWindow / ProjectManager）
class DataTreeWidget : public QWidget
{
    Q_OBJECT
public:
    explicit DataTreeWidget(QWidget *parent = nullptr);
    ~DataTreeWidget() override;

    // 记录当前 .plascan 路径，用于解析树中相对资源路径；不读取磁盘。
    void setProjectPath(const QString &plascanPath);
    // 兼容旧调用：仅记录项目路径，不再直接读取归档或临时文件。
    void loadFromArchive(const QString &plascanPath);
public slots:
    // 直接从内存提供的 JSON 元数据刷新视图（由 ProjectManager 在修改后立即调用）
    void loadFromJson(const QJsonObject &meta);
    // 添加会话级临时模型：只显示在左侧工作区，不写入项目元数据。
    void addTransientModel(const QString &modelPath);
    // 清除所有会话级临时资源。打开/新建/切换项目时调用。
    void clearTransientResources();

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
    // 用户选择任意资源项时发出（含所属分组），仅用于属性/高亮等轻量联动。
    void resourceSelected(const QString &section, const QString &resourcePath);

private slots:
    void onContextMenuRequested(const QPoint &pos);

private:
    void populateFromMeta(const QJsonObject &meta);
    QJsonObject normalizeMeta(const QJsonObject &meta) const;
    QStandardItem *createSection(const QString &title, int count);
    void appendItemRow(QStandardItem *parent, const QString &name, const QString &path, const QString &storage);
    void sortSectionChildrenByFileName(QStandardItem *section);
    bool resourceFromIndex(const QModelIndex &index, QString *section, QString *resourcePath) const;
    QString resolveResourcePath(const QString &resourcePath) const;

    QTreeView *_view{};
    QStandardItemModel *_model{};
    QString _currentPlascanPath{};
    QJsonObject _lastMeta{};
    QStringList _transientModels{};
};

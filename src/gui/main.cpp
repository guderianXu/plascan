// =============================================================================
// 文件: main.cpp
// 模块: PlaScan GUI 入口
// 说明:
//   应用程序主入口，负责：
//   1. 创建 QApplication 实例，完成 Qt 环境初始化
//   2. 初始化全局日志系统（Logger 单例）
//   3. 构造并展示主窗口 MainWindow
//   4. 进入 Qt 事件循环，直到应用退出
// =============================================================================

#include <QApplication>
#include <QDir>
#include <QFontDatabase>
#include <QMessageBox>
#include <QMetaObject>
#include <QStandardPaths>
#include "MainWindow.h"
#include "Logger.h"
// 抑制 libtiff 读取 GDAL 写入的 GeoTIFF 时产生的 tag 42113 (GDAL_NODATA) 警告
#include <tiffio.h>

#include <exception>

class SafeApplication : public QApplication
{
public:
    using QApplication::QApplication;

    bool notify(QObject *receiver, QEvent *event) override
    {
        try
        {
            return QApplication::notify(receiver, event);
        }
        catch (const std::exception &exception)
        {
            const char *receiverClass = receiver ? receiver->metaObject()->className() : "<null>";
            LOG_ERROR("Unhandled std::exception in Qt event loop. receiver=%s, what=%s",
                      receiverClass,
                      exception.what());
            QMessageBox::critical(nullptr,
                                  QStringLiteral("PlaScan 运行异常"),
                                  QStringLiteral("发生未处理的标准异常：\n%1\n\n请查看日志获取详细信息。")
                                      .arg(QString::fromLocal8Bit(exception.what())));
            return false;
        }
        catch (...)
        {
            const char *receiverClass = receiver ? receiver->metaObject()->className() : "<null>";
            LOG_ERROR("Unhandled unknown exception in Qt event loop. receiver=%s", receiverClass);
            QMessageBox::critical(nullptr,
                                  QStringLiteral("PlaScan 运行异常"),
                                  QStringLiteral("发生未处理的未知异常。\n\n请查看日志获取详细信息。"));
            return false;
        }
    }
};

// main - 程序入口
// 参数:
//   argc - 命令行参数数量
//   argv - 命令行参数字符串数组
// 返回值: int - Qt 事件循环退出码（0 表示正常退出）
int main(int argc, char *argv[])
{
    std::set_terminate([]()
    {
        LOG_ERROR("std::terminate triggered. Possible uncaught exception across Qt boundary.");
        std::abort();
    });

    // 屏蔽 libtiff 未知 tag 警告（tag 42113 = GDAL_NODATA，libtiff 不识别但无害）
    TIFFSetWarningHandler(nullptr);

    // Qt6 高 DPI 支持（必须在 QApplication 创建前设置）
    QApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    // 创建 Qt 应用程序对象，必须在任何 Qt 对象之前创建
    SafeApplication app(argc, argv);

    // 设置应用程序全局字体（优先使用思源黑体，回退到系统默认）
    QFont appFont;
    QStringList fontFamilies = {"Noto Sans CJK SC", "Source Han Sans CN", "Microsoft YaHei", "SimHei", "sans-serif"};
    for (const QString &family : fontFamilies) 
    {
        if (QFontDatabase::families().contains(family)) 
        {
            appFont.setFamily(family);
            break;
        }
    }
    appFont.setPointSize(10);  // 基准字体大小 10pt
    app.setFont(appFont);
    
    const QString baseDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir(baseDir).mkpath(QStringLiteral("logs"));
    Logger::instance()->setLogDirectory(QDir(baseDir).filePath(QStringLiteral("logs")));
    LOG_INFO("PlaScan GUI started");
    
    try
    {
        // 创建并展示主窗口
        MainWindow mainWindow;
        mainWindow.show();

        // 进入 Qt 事件循环，阻塞直到应用退出
        return app.exec();
    }
    catch (const std::exception &exception)
    {
        LOG_ERROR("Fatal startup std::exception: %s", exception.what());
        QMessageBox::critical(nullptr,
                              QStringLiteral("PlaScan 启动失败"),
                              QStringLiteral("启动时捕获到异常：\n%1\n\n请查看日志并反馈该信息。")
                                  .arg(QString::fromLocal8Bit(exception.what())));
        return EXIT_FAILURE;
    }
    catch (...)
    {
        LOG_ERROR("Fatal startup unknown exception.");
        QMessageBox::critical(nullptr,
                              QStringLiteral("PlaScan 启动失败"),
                              QStringLiteral("启动时捕获到未知异常。\n\n请查看日志并反馈该信息。"));
        return EXIT_FAILURE;
    }
}

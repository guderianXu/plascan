#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QRegularExpression>
#include <QSet>
#include <QString>

namespace
{

    QString readTextFile(const QString& relativePath)
    {
        const QString path = QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(relativePath);
        QFile file(path);
        EXPECT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text)) << qPrintable(path);
        return file.isOpen() ? QString::fromUtf8(file.readAll()) : QString();
    }

    QSet<QString> extractPublicMethods(const QString& header)
    {
        const int class_start = header.indexOf(QStringLiteral("class ProjectManager"));
        const int public_start = header.indexOf(QStringLiteral("public:"), class_start);
        const int signals_start = header.indexOf(QStringLiteral("signals:"), public_start);
        if (class_start < 0 || public_start < 0 || signals_start < 0)
        {
            return {};
        }

        const QString public_block = header.mid(public_start, signals_start - public_start);
        const QRegularExpression method_pattern(
            QStringLiteral(R"(^\s*[A-Za-z_][A-Za-z0-9_:<>,\s*&]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\()"),
            QRegularExpression::MultilineOption);
        const QRegularExpression constructor_pattern(QStringLiteral(R"(^\s*(~?ProjectManager)\s*\()"),
                                                     QRegularExpression::MultilineOption);
        QSet<QString> methods;
        for (auto match = method_pattern.globalMatch(public_block); match.hasNext();)
        {
            methods.insert(match.next().captured(1));
        }
        for (auto match = constructor_pattern.globalMatch(public_block); match.hasNext();)
        {
            methods.insert(match.next().captured(1));
        }
        return methods;
    }

    TEST(ProjectManagerArchitectureTest, FinalFacadeOwnsOnlyOneServiceContainer)
    {
        const QString header = readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
        const QString source = readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
        const int private_start = header.indexOf(QStringLiteral("private:"));
        const int class_end = header.lastIndexOf(QStringLiteral("};"));
        ASSERT_GE(private_start, 0);
        ASSERT_GT(class_end, private_start);

        const QString private_block = header.mid(private_start, class_end - private_start);
        const QRegularExpression member_pattern(QStringLiteral(R"(\b(_[A-Za-z][A-Za-z0-9_]*)\s*;)"));
        QSet<QString> members;
        for (auto match = member_pattern.globalMatch(private_block); match.hasNext();)
        {
            members.insert(match.next().captured(1));
        }

        EXPECT_EQ(members, QSet<QString>{QStringLiteral("_serviceContainer")});
        EXPECT_EQ(source.count(QStringLiteral("std::make_unique<")), 1);
        EXPECT_TRUE(source.contains(
            QStringLiteral("std::make_unique<xjw::gui::project::ProjectServiceContainer>(projectData, parent)")));
    }

    TEST(ProjectManagerArchitectureTest, PublicCommandsAreOnlyStableServiceAccessorsAndCleanupWait)
    {
        const QSet<QString> expected{
            QStringLiteral("ProjectManager"),
            QStringLiteral("~ProjectManager"),
            QStringLiteral("services"),
            QStringLiteral("session"),
            QStringLiteral("tasks"),
            QStringLiteral("resources"),
            QStringLiteral("cleanup"),
            QStringLiteral("lifecycle"),
            QStringLiteral("waitForResourceCleanup"),
        };
        EXPECT_EQ(extractPublicMethods(readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"))),
                  expected);
    }

    TEST(ProjectManagerArchitectureTest, FinalFacadeHasNoWorkflowImplementationDependencies)
    {
        const QString header = readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
        const QString source = readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));

        for (const QString& forbidden : {
                 QStringLiteral("ProjectLifecycleController"),
                 QStringLiteral("ProjectUiCommands"),
                 QStringLiteral("QMessageBox"),
                 QStringLiteral("QtConcurrent"),
                 QStringLiteral("_projectData"),
                 QStringLiteral("_parent"),
                 QStringLiteral("_fileDialogState"),
                 QStringLiteral("_atCancelFlag"),
                 QStringLiteral("QFuture"),
             })
        {
            EXPECT_FALSE(header.contains(forbidden)) << qPrintable(forbidden);
            EXPECT_FALSE(source.contains(forbidden)) << qPrintable(forbidden);
        }
    }

    TEST(ProjectManagerArchitectureTest, RemainingResponsibilitiesLiveBehindNarrowBoundaries)
    {
        const QString session = readTextFile(QStringLiteral("src/gui/project/services/ProjectSession.h"));
        const QString lifecycle = readTextFile(QStringLiteral("src/gui/project/services/ProjectLifecycleService.h"));
        const QString resources = readTextFile(QStringLiteral("src/gui/project/services/ProjectResourceService.h"));
        const QString tasks = readTextFile(QStringLiteral("src/gui/project/tasks/ProjectTaskOrchestrator.h"));

        for (const QString& method : {
                 QStringLiteral("getPinholeNumericStatesForImages"),
                 QStringLiteral("getImageIdsForImages"),
                 QStringLiteral("getReferenceCameraGeometriesForImages"),
                 QStringLiteral("getRpcCameraImagePaths"),
             })
        {
            EXPECT_TRUE(session.contains(method)) << qPrintable(method);
        }
        for (const QString& method : {
                 QStringLiteral("createChunk"),
                 QStringLiteral("renameChunk"),
                 QStringLiteral("removeChunk"),
                 QStringLiteral("switchChunk"),
             })
        {
            EXPECT_TRUE(lifecycle.contains(method)) << qPrintable(method);
        }
        for (const QString& method : {
                 QStringLiteral("openSurveyControlDialog"),
                 QStringLiteral("runReferenceQualityCheck"),
                 QStringLiteral("prepareReferenceTerrainBundleAdjust"),
                 QStringLiteral("refreshReconstructionQualityReport"),
             })
        {
            EXPECT_TRUE(resources.contains(method)) << qPrintable(method);
        }
        for (const QString& method : {
                 QStringLiteral("beginTask"),
                 QStringLiteral("isTaskActive"),
                 QStringLiteral("reportSparseProgress"),
                 QStringLiteral("reportSparseComputeDevice"),
                 QStringLiteral("reportSparseMatchPair"),
                 QStringLiteral("reportSparseTiePointResult"),
                 QStringLiteral("finishTask"),
             })
        {
            EXPECT_TRUE(tasks.contains(method)) << qPrintable(method);
        }
    }

    TEST(ProjectManagerArchitectureTest, FacadeRetainsTheUiSignalCompatibilitySurface)
    {
        const QString header = readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
        for (const QString& signal : {
                 QStringLiteral("projectOpened"),
                 QStringLiteral("projectClosed"),
                 QStringLiteral("projectSessionChanged"),
                 QStringLiteral("projectMetadataChanged"),
                 QStringLiteral("projectMetadataUpdated"),
                 QStringLiteral("metadataDirtyChanged"),
                 QStringLiteral("imageMatchResultAppended"),
                 QStringLiteral("matchPairReady"),
                 QStringLiteral("masksGenerated"),
                 QStringLiteral("interactiveMaskSaved"),
                 QStringLiteral("interactiveMaskSaveFailed"),
                 QStringLiteral("atProgressChanged"),
                 QStringLiteral("atComputeDeviceChanged"),
                 QStringLiteral("atProgressFinished"),
                 QStringLiteral("tiePointResultReady"),
             })
        {
            const QRegularExpression declaration(
                QStringLiteral(R"(\bvoid\s+%1\s*\()").arg(QRegularExpression::escape(signal)));
            EXPECT_TRUE(declaration.match(header).hasMatch()) << qPrintable(signal);
        }
    }

} // namespace

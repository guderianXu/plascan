#include "project/services/ProjectUiMessageAdapter.h"

#include <gtest/gtest.h>

#include <vector>

namespace
{

    class FakeProjectUiMessageAdapter final : public ProjectUiMessageAdapter
    {
    public:
        std::vector<QString> calls;
        UiDialogResult nextFileResult;
        UiDialogResult nextTextResult;
        UiDialogResult nextDoubleResult;
        UiDialogResult nextItemResult;
        UiDialogResult nextFilesResult;
        UiDialogResult nextDirectoryResult;
        UiDialogResult nextSaveResult;
        UiAnswer nextAnswer = UiAnswer::Cancel;
        QDir::Filters lastDirectoryFilters;
        QString lastDefaultSuffix;

        void information(QWidget*, const QString&, const QString&) override
        {
            calls.emplace_back(QStringLiteral("information"));
        }

        void warning(QWidget*, const QString&, const QString&) override
        {
            calls.emplace_back(QStringLiteral("warning"));
        }

        void critical(QWidget*, const QString&, const QString&) override
        {
            calls.emplace_back(QStringLiteral("critical"));
        }

        UiAnswer question(QWidget*, const QString&, const QString&, UiAnswer) override
        {
            calls.emplace_back(QStringLiteral("question"));
            return nextAnswer;
        }

        UiDialogResult getText(QWidget*, const QString&, const QString&, const QString&) override
        {
            calls.emplace_back(QStringLiteral("getText"));
            return nextTextResult;
        }

        UiDialogResult getDouble(QWidget*, const QString&, const QString&, double, double, double, int) override
        {
            calls.emplace_back(QStringLiteral("getDouble"));
            return nextDoubleResult;
        }

        UiDialogResult getItem(QWidget*, const QString&, const QString&, const QStringList&, int) override
        {
            calls.emplace_back(QStringLiteral("getItem"));
            return nextItemResult;
        }

        UiDialogResult selectOpenFile(
            QWidget*, const QString&, const QString&, const QString&, QDir::Filters directoryFilters) override
        {
            calls.emplace_back(QStringLiteral("selectOpenFile"));
            lastDirectoryFilters = directoryFilters;
            return nextFileResult;
        }

        UiDialogResult selectOpenFiles(
            QWidget*, const QString&, const QString&, const QString&, QDir::Filters directoryFilters) override
        {
            calls.emplace_back(QStringLiteral("selectOpenFiles"));
            lastDirectoryFilters = directoryFilters;
            return nextFilesResult;
        }

        UiDialogResult
        selectDirectory(QWidget*, const QString&, const QString&, QDir::Filters directoryFilters) override
        {
            calls.emplace_back(QStringLiteral("selectDirectory"));
            lastDirectoryFilters = directoryFilters;
            return nextDirectoryResult;
        }

        UiDialogResult selectSaveFile(QWidget*,
                                      const QString&,
                                      const QString&,
                                      const QString&,
                                      QDir::Filters directoryFilters,
                                      const QString& defaultSuffix) override
        {
            calls.emplace_back(QStringLiteral("selectSaveFile"));
            lastDirectoryFilters = directoryFilters;
            lastDefaultSuffix = defaultSuffix;
            return nextSaveResult;
        }
    };

} // namespace

TEST(ProjectUiMessageAdapterTest, CancelledFileSelectionIsNotAccepted)
{
    FakeProjectUiMessageAdapter adapter;
    adapter.nextFileResult = UiDialogResult{};

    const UiDialogResult result =
        adapter.selectOpenFile(nullptr, QStringLiteral("打开"), {}, {}, QDir::AllEntries | QDir::Hidden);

    EXPECT_FALSE(result.accepted);
    EXPECT_TRUE(result.text.isEmpty());
    EXPECT_EQ(adapter.calls.size(), 1U);
}

TEST(ProjectUiMessageAdapterTest, AcceptedFileSelectionPreservesText)
{
    FakeProjectUiMessageAdapter adapter;
    adapter.nextFileResult.accepted = true;
    adapter.nextFileResult.text = QStringLiteral("/tmp/example.plascan");

    const UiDialogResult result =
        adapter.selectOpenFile(nullptr, QStringLiteral("打开"), {}, {}, QDir::AllEntries | QDir::Hidden);

    EXPECT_TRUE(result.accepted);
    EXPECT_EQ(result.text, QStringLiteral("/tmp/example.plascan"));
}

TEST(ProjectUiMessageAdapterTest, FakePreservesDialogResultShape)
{
    FakeProjectUiMessageAdapter adapter;
    adapter.nextAnswer = UiAnswer::Yes;
    adapter.nextTextResult = UiDialogResult{true, QStringLiteral("name"), {}, 0.0};
    adapter.nextDoubleResult = UiDialogResult{true, {}, {}, 1.25};
    adapter.nextItemResult = UiDialogResult{true, QStringLiteral("item"), {}, 0.0};
    adapter.nextFilesResult = UiDialogResult{true, {}, {QStringLiteral("a.tif"), QStringLiteral("b.tif")}, 0.0};
    adapter.nextDirectoryResult = UiDialogResult{true, QStringLiteral("/tmp/images"), {}, 0.0};
    adapter.nextSaveResult = UiDialogResult{true, QStringLiteral("/tmp/out.zip"), {}, 0.0};

    const QDir::Filters dialogFilters = QDir::AllEntries | QDir::Hidden | QDir::AllDirs | QDir::NoDotAndDotDot;
    EXPECT_EQ(adapter.question(nullptr, {}, {}, UiAnswer::No), UiAnswer::Yes);
    EXPECT_EQ(adapter.getText(nullptr, {}, {}, {}).text, QStringLiteral("name"));
    EXPECT_DOUBLE_EQ(adapter.getDouble(nullptr, {}, {}, 0.0, 0.0, 2.0, 2).number, 1.25);
    EXPECT_EQ(adapter.getItem(nullptr, {}, {}, {}, 0).text, QStringLiteral("item"));
    EXPECT_EQ(adapter.selectOpenFiles(nullptr, {}, {}, {}, dialogFilters).texts.size(), 2);
    EXPECT_EQ(adapter.selectDirectory(nullptr, {}, {}, dialogFilters).text, QStringLiteral("/tmp/images"));
    EXPECT_EQ(adapter.selectSaveFile(nullptr, {}, {}, {}, dialogFilters, QStringLiteral("plascan")).text,
              QStringLiteral("/tmp/out.zip"));
    EXPECT_EQ(adapter.lastDirectoryFilters, dialogFilters);
    EXPECT_EQ(adapter.lastDefaultSuffix, QStringLiteral("plascan"));
    EXPECT_EQ(adapter.calls.size(), 7U);
}

TEST(ProjectUiMessageAdapterTest, ReviewDefaultsToQuestionForExistingFakes)
{
    FakeProjectUiMessageAdapter adapter;
    adapter.nextAnswer = UiAnswer::Yes;
    UiReviewDialogRequest request;
    request.title = QStringLiteral("预览");
    request.text = QStringLiteral("是否保留？");

    EXPECT_EQ(adapter.review(nullptr, request), UiReviewDecision::Accept);
    ASSERT_EQ(adapter.calls.size(), 1U);
    EXPECT_EQ(adapter.calls.front(), QStringLiteral("question"));
}

#include "file/FileIO.h"
#include "file/JsonFile.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

namespace
{

    using namespace xjw::common::file;

    class StandardFileIOTest : public testing::Test
    {
    protected:
        void SetUp() override
        {
            static std::atomic<unsigned> sequence{0};
            _directory = pathFromUtf8(PLASCAN_STANDARD_IO_TMP_DIR) /
                         (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                          std::to_string(sequence.fetch_add(1)));
            ASSERT_TRUE(ensureDirectory(_directory));
        }

        void TearDown() override
        {
            std::error_code error;
            std::filesystem::remove_all(_directory, error);
            EXPECT_FALSE(error) << error.message();
        }

        std::filesystem::path _directory;
    };

    TEST_F(StandardFileIOTest, ReadsAndReplacesUnicodeFilesAndEmptyContent)
    {
        const auto path = _directory / pathFromUtf8("月球影像/报告.bin");
        EXPECT_EQ(pathToUtf8(path.filename()), "报告.bin");
        std::string error;
        ASSERT_TRUE(writeFileAtomic(path, std::string("a\0b", 3), &error)) << error;
        std::string bytes;
        ASSERT_TRUE(readFile(path, &bytes, &error)) << error;
        EXPECT_EQ(bytes, std::string("a\0b", 3));
        ASSERT_TRUE(writeFileAtomic(path, "", &error)) << error;
        ASSERT_TRUE(readFile(path, &bytes, &error)) << error;
        EXPECT_TRUE(bytes.empty());
        EXPECT_TRUE(error.empty());
    }

    TEST_F(StandardFileIOTest, AbandonedAndFailedWritesPreserveExistingDestination)
    {
        const auto path = _directory / "result.txt";
        ASSERT_TRUE(writeFileAtomic(path, "original"));
        {
            AtomicFile output(path);
            ASSERT_TRUE(output.open());
            output.stream() << "partial";
        }
        {
            AtomicFile output(path);
            ASSERT_TRUE(output.open());
            output.stream() << "failed";
            output.stream().setstate(std::ios::badbit);
            std::string error;
            EXPECT_FALSE(output.commit(&error));
            EXPECT_NE(error.find("result.txt"), std::string::npos);
        }
        std::string bytes;
        ASSERT_TRUE(readFile(path, &bytes));
        EXPECT_EQ(bytes, "original");
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(_directory), std::filesystem::directory_iterator()),
                  1);
    }

    TEST_F(StandardFileIOTest, FailedReplacementPreservesDirectoryAndCleansTemporary)
    {
        const auto path = _directory / "result";
        ASSERT_TRUE(ensureDirectory(path));
        ASSERT_TRUE(writeFileAtomic(path / "keep.txt", "keep"));
        std::string error;
        EXPECT_FALSE(writeFileAtomic(path, "replacement", &error));
        EXPECT_FALSE(error.empty());
        EXPECT_TRUE(std::filesystem::is_directory(path));
        EXPECT_TRUE(std::filesystem::exists(path / "keep.txt"));
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(_directory), std::filesystem::directory_iterator()),
                  1);
    }

    TEST_F(StandardFileIOTest, ConcurrentWritersPublishOnlyCompleteDocuments)
    {
        const auto path = _directory / "concurrent.json";
        std::atomic<int> failures{0};
        std::vector<std::jthread> writers;
        for (int index = 0; index < 8; ++index)
        {
            writers.emplace_back(
                [&, index]
                {
                    const nlohmann::json value{{"index", index},
                                               {"payload", std::string(65536, static_cast<char>('a' + index))}};
                    if (!writeJsonAtomic(path, value))
                    {
                        ++failures;
                    }
                });
        }
        writers.clear();
        EXPECT_EQ(failures.load(), 0);
        nlohmann::json document;
        ASSERT_TRUE(readJson(path, &document));
        const int index = document.at("index").get<int>();
        EXPECT_EQ(document.at("payload").get<std::string>(), std::string(65536, static_cast<char>('a' + index)));
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(_directory), std::filesystem::directory_iterator()),
                  1);
    }

    TEST_F(StandardFileIOTest, JsonPreservesUtf8AndIntegerPrecisionAndRejectsMalformedFiles)
    {
        const auto path = _directory / pathFromUtf8("相机.json");
        const nlohmann::json expected{{"名称", "月球"},
                                      {"id", std::uint64_t{9007199254740993ULL}},
                                      {"observations", nlohmann::json::array()},
                                      {"missing", nullptr}};
        ASSERT_TRUE(writeJsonAtomic(path, expected));
        nlohmann::json document;
        ASSERT_TRUE(readJson(path, &document));
        EXPECT_EQ(document, expected);
        ASSERT_TRUE(writeFileAtomic(path, "{ broken JSON"));
        std::string error;
        EXPECT_FALSE(readJson(path, &document, &error));
        EXPECT_EQ(document, expected);
        EXPECT_NE(error.find("相机.json"), std::string::npos);
        EXPECT_NE(error.find("JSON"), std::string::npos);
    }

    TEST_F(StandardFileIOTest, ReportsMissingAndInvalidOutputPaths)
    {
        std::string bytes = "old";
        std::string error;
        EXPECT_FALSE(readFile(_directory / "missing.txt", &bytes, &error));
        EXPECT_TRUE(bytes.empty());
        EXPECT_NE(error.find("missing.txt"), std::string::npos);
        ASSERT_TRUE(writeFileAtomic(_directory / "parent", "file"));
        EXPECT_FALSE(writeFileAtomic(_directory / "parent" / "child", "data", &error));
        EXPECT_FALSE(error.empty());
        EXPECT_FALSE(writeFileAtomic({}, "data", &error));
        EXPECT_FALSE(error.empty());
    }

    TEST_F(StandardFileIOTest, PreservesExistingPermissionsAndFileSymlink)
    {
        const auto target = _directory / "target";
        ASSERT_TRUE(writeFileAtomic(target, "original"));
        const auto permissions = std::filesystem::status(target).permissions();
        ASSERT_TRUE(writeFileAtomic(target, "replacement"));
        EXPECT_EQ(std::filesystem::status(target).permissions(), permissions);
        const auto alias = _directory / "alias";
        std::error_code failure;
        std::filesystem::create_symlink(target, alias, failure);
        if (failure)
        {
            GTEST_SKIP() << "File symlinks are unavailable: " << failure.message();
        }
        ASSERT_TRUE(writeFileAtomic(alias, "through alias"));
        EXPECT_TRUE(std::filesystem::is_symlink(alias));
        std::string bytes;
        ASSERT_TRUE(readFile(target, &bytes));
        EXPECT_EQ(bytes, "through alias");
    }

    TEST_F(StandardFileIOTest, InvalidUtf8JsonDoesNotReplaceExistingFile)
    {
        const auto path = _directory / "document.json";
        ASSERT_TRUE(writeFileAtomic(path, "original"));
        const nlohmann::json invalid = {{"text", std::string(1, static_cast<char>(0xff))}};
        std::string error;
        EXPECT_FALSE(writeJsonAtomic(path, invalid, &error));
        EXPECT_NE(error.find("JSON"), std::string::npos);
        std::string bytes;
        ASSERT_TRUE(readFile(path, &bytes));
        EXPECT_EQ(bytes, "original");
        EXPECT_FALSE(readFile(_directory, &bytes, &error));
    }

} // namespace

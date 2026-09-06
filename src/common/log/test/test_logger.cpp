#include "Logger.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

    class LoggerTerminalTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            Logger::instance()->setTerminalMinimumLevel(Logger::Info);
        }

        void TearDown() override
        {
            Logger::instance()->setTerminalMinimumLevel(Logger::Info);
        }
    };

    TEST_F(LoggerTerminalTest, RegisteredSinkDoesNotSuppressInfoOnTerminal)
    {
        std::vector<Logger::Entry> received;
        const int sink_id =
            Logger::instance()->registerSink([&received](const Logger::Entry& entry) { received.push_back(entry); });

        testing::internal::CaptureStdout();
        Logger::instance()->log(Logger::Info, "terminal-and-sink-marker");
        const std::string terminal_output = testing::internal::GetCapturedStdout();
        Logger::instance()->unregisterSink(sink_id);

        ASSERT_EQ(received.size(), 1u);
        EXPECT_NE(received.front().message.find("terminal-and-sink-marker"), std::string::npos);
        EXPECT_NE(terminal_output.find("[INFO] terminal-and-sink-marker"), std::string::npos);
    }

    TEST_F(LoggerTerminalTest, InfoTerminalLevelKeepsDebugInSinkOnly)
    {
        std::vector<Logger::Entry> received;
        const int sink_id =
            Logger::instance()->registerSink([&received](const Logger::Entry& entry) { received.push_back(entry); });

        testing::internal::CaptureStdout();
        Logger::instance()->log(Logger::Debug, "debug-sink-only-marker");
        const std::string terminal_output = testing::internal::GetCapturedStdout();
        Logger::instance()->unregisterSink(sink_id);

        ASSERT_EQ(received.size(), 1u);
        EXPECT_NE(received.front().message.find("debug-sink-only-marker"), std::string::npos);
        EXPECT_TRUE(terminal_output.empty());
    }

    TEST_F(LoggerTerminalTest, ThreadMinimumLevelSuppressesOnlyScopedLowPriorityEntries)
    {
        std::vector<Logger::Entry> received;
        const int sink_id =
            Logger::instance()->registerSink([&received](const Logger::Entry& entry) { received.push_back(entry); });

        Logger::instance()->setTerminalMinimumLevel(Logger::Error);
        {
            Logger::ScopedThreadMinimumLevel minimum_level(Logger::Warn);
            Logger::instance()->info("scoped-info-marker");
            Logger::instance()->warn("scoped-warning-marker");
        }
        Logger::instance()->info("restored-info-marker");
        Logger::instance()->unregisterSink(sink_id);

        ASSERT_EQ(received.size(), 2u);
        EXPECT_EQ(received[0].level, Logger::Warn);
        EXPECT_EQ(received[0].message, "scoped-warning-marker");
        EXPECT_EQ(received[1].level, Logger::Info);
        EXPECT_EQ(received[1].message, "restored-info-marker");
    }

    TEST_F(LoggerTerminalTest, StructuredEntriesCarrySequenceSessionAndExplicitContext)
    {
        Logger::instance()->setTerminalMinimumLevel(Logger::Error);
        std::vector<Logger::Entry> received;
        const int sink_id =
            Logger::instance()->registerSink([&received](const Logger::Entry& entry) { received.push_back(entry); });

        Logger::Context context;
        context.category = "Task";
        context.taskId = "mesh";
        context.stage = "reconstruct";
        const std::uint64_t sequence =
            Logger::instance()->logWithContext(Logger::Info, "structured-entry-marker", context);
        Logger::instance()->unregisterSink(sink_id);

        ASSERT_EQ(received.size(), 1u);
        EXPECT_EQ(received.front().sequence, sequence);
        EXPECT_EQ(received.front().sessionId, Logger::instance()->sessionId());
        EXPECT_EQ(received.front().category, "Task");
        EXPECT_EQ(received.front().taskId, "mesh");
        EXPECT_EQ(received.front().stage, "reconstruct");
        EXPECT_GE(Logger::instance()->latestSequence(), sequence);
        const std::vector<Logger::Entry> recent = Logger::instance()->recentEntries();
        ASSERT_FALSE(recent.empty());
        EXPECT_EQ(recent.back().sequence, sequence);
    }

    TEST_F(LoggerTerminalTest, ScopedContextIsRestoredAfterScope)
    {
        Logger::instance()->setTerminalMinimumLevel(Logger::Error);
        std::vector<Logger::Entry> received;
        const int sink_id =
            Logger::instance()->registerSink([&received](const Logger::Entry& entry) { received.push_back(entry); });
        {
            Logger::ScopedContext context({"MVS", "dense_cloud", "fusion"});
            Logger::instance()->info("scoped-context-marker");
        }
        Logger::instance()->info("restored-context-marker");
        Logger::instance()->unregisterSink(sink_id);

        ASSERT_EQ(received.size(), 2u);
        EXPECT_EQ(received[0].category, "MVS");
        EXPECT_EQ(received[0].taskId, "dense_cloud");
        EXPECT_EQ(received[0].stage, "fusion");
        EXPECT_TRUE(received[1].category.empty());
        EXPECT_TRUE(received[1].taskId.empty());
    }

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

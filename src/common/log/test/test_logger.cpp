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
    const int sink_id = Logger::instance()->registerSink(
        [&received](const Logger::Entry &entry)
        {
            received.push_back(entry);
        });

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
    const int sink_id = Logger::instance()->registerSink(
        [&received](const Logger::Entry &entry)
        {
            received.push_back(entry);
        });

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
    const int sink_id = Logger::instance()->registerSink(
        [&received](const Logger::Entry &entry)
        {
            received.push_back(entry);
        });

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

} // namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

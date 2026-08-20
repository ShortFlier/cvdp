#include "log.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

void logInit(Log_Level level, const std::string& filePath) {
    spdlog::set_level(level);

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("console", console_sink);
    // 设置格式：[2026-05-08 14:30:25.123 INFO 12345]
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e %^%l%$ %t] %v");
    spdlog::set_default_logger(logger);
}
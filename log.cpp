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

#ifndef NDEBUG
    // Debug mode: output to console
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("console", console_sink);
    // 设置格式：[2026-05-08 14:30:25.123 INFO 12345]
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e %^%l%$ %t] %v");
    spdlog::set_default_logger(logger);
#else
    // Release mode: output to file
    std::string path=filePath;
    if (filePath.empty()) {
        // Generate default path "./log/yyyy-MM-dd.txt"
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now = *std::localtime(&time_t_now);
        std::ostringstream oss;
        oss << "./log/" << std::put_time(&tm_now, "%Y-%m-%d") << ".txt";
        path = oss.str();
    }

    // Ensure the log directory exists
    fs::create_directories(fs::path(path).parent_path());

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, true);
    auto logger = std::make_shared<spdlog::logger>("file", file_sink);
    // 设置格式：[2026-05-08 14:30:25.123 INFO 12345]
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e %^%l%$ %t] %v");
    spdlog::set_default_logger(logger);
#endif
}
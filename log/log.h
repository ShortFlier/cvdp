#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/ranges.h> 


using Log_Level = spdlog::level::level_enum;

void logInit(Log_Level level, const std::string& filePath = std::string());

#define log_info(...) SPDLOG_INFO(__VA_ARGS__)

#define log_warn(...) SPDLOG_WARN(__VA_ARGS__)

#define log_error(...) SPDLOG_ERROR(__VA_ARGS__)

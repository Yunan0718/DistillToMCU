cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(distilltomcu)

# LittleFS image from spiffs_data directory
# 注意: littlefs_create_partition_image 在 IDF v5.2.x 中可能还未支持
# 改用标准 spiffs 创建（LittleFS 兼容 spiffs 分区类型）
spiffs_create_partition_image(littlefs spiffs_data FLASH_IN_PROJECT)

# Setup

## System

Windows 10/11, Ubuntu 20.04+, or macOS 12+.

## ESP-IDF

- **Version**: **v5.2.6** (the firmware in `main/` targets this release).
- Windows installer: <https://dl.espressif.com/dl/esp-idf/>
- Linux/macOS:

```bash
git clone -b v5.2.6 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3 && . ./export.sh
```

## Python

The pipeline in `poc/` requires Python 3.11:

```bash
pip install -r requirements.txt
```

Set the API keys before running online experiments:

```bash
export DEEPSEEK_API_KEY=sk-...      # required for real-teacher runs
export DASHSCOPE_API_KEY=sk-...     # optional, for the Qwen cross-model runs
```

Without keys, use the mock mode to exercise the pipeline end to end.

## Firmware

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p COMx flash monitor
```

Copy `sdkconfig.user.example` to `sdkconfig.user` and fill in the WiFi and API
credentials before flashing. `sdkconfig.user` is git-ignored.

## Hardware

ESP32-S3-DevKitC-1 (N16R8): 16 MB flash, 8 MB PSRAM. Sensor readings are
injected over USB serial, so no external sensor board is required.
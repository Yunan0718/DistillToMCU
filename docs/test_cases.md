# Test cases

Manual acceptance checks for the firmware. Run them through the serial CLI.

| ID | Case | Input / action | Expected |
|---|---|---|---|
| TC-001 | Basic chat | `hello` | LLM replies normally |
| TC-002 | Identity query | `who are you` | Describes itself as an ESP32-S3 control module |
| TC-101 | LED on | `turn on light 1` | Serial log shows LED1 ON, reply confirms |
| TC-102 | LED off | `turn off light 2` | Serial log shows LED2 OFF, reply confirms |
| TC-201 | Sensor injection | `sensor {"temperature":29.5,"humidity":60}` | Snapshot accepted; rule matching runs |
| TC-202 | Rule match | Injected state inside a learned interval | Local execution, no cloud call |
| TC-203 | Cloud fallback | Injected state outside all intervals | Cloud LLM called, action executed |
| TC-301 | Latency stats | `latstats` | Prints p50/p95/p99 match latency |
| TC-302 | Rule list | `rules` | Prints the active rule store |
| TC-401 | Persistence | Reboot after learning | Rules are restored from flash |
| TC-402 | Offline | Disconnect WiFi, inject known state | Local rules still execute |
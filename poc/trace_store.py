"""
DistillToMCU Phase 0 PoC — 轨迹存储
====================================
JSONL 格式的交互轨迹存储，模拟 MCU 上的 Flash 循环缓冲。
"""

import json
import os
from datetime import datetime
from config import OUTPUT_DIR, TRACE_FILE


class TraceStore:
    """
    每条 trace 包含完整的交互记录：
    {id, ts, sensors, user_input, llm_response, execution, feedback}
    存储为 JSONL（一行一条），模拟 MCU Flash 上的追加写入。
    """

    def __init__(self, output_dir=OUTPUT_DIR, filename=TRACE_FILE):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, filename)
        self.traces = []       # 内存中的完整列表（MCU 上只在 Flash）
        self._id_counter = 0
        # v7: 只在有新数据写入时才 truncate（参见 start_new_session），
        # 不在构造时 truncate——避免 load_all() 先构造就被清空。

    def start_new_session(self):
        """v7: 显式清空文件（开始新实验时调用，不在构造时自动清空）。"""
        open(self.path, "w", encoding="utf-8").close()
        self.traces = []
        self._id_counter = 0

    def add(self, sensors, user_input, llm_response, execution_mode, rule_id, feedback="accepted"):
        self._id_counter += 1
        trace = {
            "id": f"trace_{self._id_counter:06d}",
            "ts": int(datetime.now().timestamp()),
            "sensors": {k: round(v, 2) if isinstance(v, float) else v
                        for k, v in sensors.items()},
            "user_input": user_input,
            "intent": "",                          # 后续可由 LLM 标注
            "llm_response": llm_response,          # {reasoning, tool_calls, model, latency_ms}
            "execution": {
                "mode": execution_mode,            # "cloud" | "local"
                "rule_id": rule_id,
                "result": "success",
            },
            "feedback": {
                "type": feedback,                  # "accepted" | "corrected" | "ignored"
                "ts": None,
            }
        }
        self.traces.append(trace)

        # 追加写入 JSONL（模拟 MCU Flash 追加写）
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        return trace

    def get_last_n(self, n=50):
        """获取最近 N 条 trace"""
        return self.traces[-n:]

    def get_by_mode(self, mode, n=None):
        """按执行模式筛选（cloud / local）"""
        filtered = [t for t in self.traces if t["execution"]["mode"] == mode]
        return filtered[-n:] if n else filtered

    def count(self):
        return len(self.traces)

    def cloud_count(self):
        return len([t for t in self.traces if t["execution"]["mode"] == "cloud"])

    def local_count(self):
        return len([t for t in self.traces if t["execution"]["mode"] == "local"])

    def autonomy_rate(self):
        total = len(self.traces)
        if total == 0:
            return 0.0
        return self.local_count() / total

    def load_all(self):
        """从 JSONL 文件重新加载全部 trace"""
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.traces = [json.loads(line) for line in f if line.strip()]
            self._id_counter = len(self.traces)
        return self.traces

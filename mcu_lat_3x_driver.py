"""Three independent on-board match-latency sessions (COM6).

Each session: inject 100 run4b_seed777_seed42 sensor snapshots, read the
`latstats` distribution, then `reboot` the device so the next session starts
from cleared telemetry counters. One JSON per session + an appended log.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcu_run_4x import load_dir_traces
from serial_mcu_exp import MCU

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poc", "output")
LOG = os.path.join(OUT, "mcu_lat_3x_run.log")
N_RUNS = 3
COUNT = 100
DELAY = 5.0
PORT = "COM6"
TRACES_DIR = "run4b_seed777_seed42"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def reboot(mcu):
    mcu.ser.reset_input_buffer()
    mcu.ser.write(b"reboot\r\n")
    mcu.ser.flush()
    time.sleep(0.5)


def main():
    sensors = load_dir_traces(TRACES_DIR, COUNT)
    log(f"driver start: {len(sensors)} sensors x {N_RUNS} sessions, "
        f"port={PORT}, delay={DELAY}s")
    for run in range(1, N_RUNS + 1):
        mcu = MCU(PORT)
        s0 = mcu.stats()
        log(f"RUN {run} init: rules={s0.get('rules_total')} "
            f"active={s0.get('rules_active')} SRAM={s0.get('free_sram')}")
        lost = 0
        for i, s in enumerate(sensors):
            ok, _status = mcu.inject(s)
            if not ok:
                lost += 1
            time.sleep(max(0.0, DELAY - 1.5))
            if (i + 1) % 50 == 0:
                log(f"RUN {run} progress {i + 1}/{COUNT} lost={lost}")
        sf = mcu.stats()
        lat = mcu.latstats()
        result = {
            "run": run,
            "traces_dir": TRACES_DIR,
            "n": COUNT,
            "final": sf,
            "latstats": lat,
            "inject_lost": lost,
        }
        out_path = os.path.join(OUT, f"mcu_lat_run{run}_{int(time.time())}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log(f"RUN {run} done: AR={sf.get('ar_pct')}% local="
            f"{sf.get('local')}/{sf.get('total')} lat={lat} saved={os.path.basename(out_path)}")
        reboot(mcu)
        mcu.close()
        if run < N_RUNS:
            time.sleep(9.0)  # let the device reboot and re-enumerate
    log("ALL DONE")


if __name__ == "__main__":
    main()

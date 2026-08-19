# DistillToMCU

DistillToMCU distills a cloud LLM's control behavior into confidence-calibrated,
MCU-executable interval rules. After a short observation window, an ESP32-S3
executes locally with **zero LLM dependency**; the cloud LLM remains only as a
fallback for novel states.

This repository is the open-source companion to the manuscript in `paper/`
(LaTeX source and compiled PDF), and contains the full Python pipeline, the
ESP32-S3 firmware, the experiment results, and the figure-generation scripts.

## Key ideas

- **COMIC** — online interval-rule distillation from *positive-only* LLM
  behavioral traces: CKMS online quantiles, Welford online variance, split
  conformal calibration, MDL consolidation, and discriminative-condition
  selection.
- **PMS** — a memory-efficient Follow-the-Perturbed-Leader rule selector
  (2 bytes/rule, uint8 alpha/beta) evaluated in both stationary and drifting
  environments.
- **End-to-end hardware validation** — ESP32-S3 firmware (ESP-IDF v5.2.6) with
  a five-state rule lifecycle (candidate → verified → active → degraded →
  retired), trace-driven hardware-in-the-loop experiments, and measured
  on-device match latency (p50 ≈ 1.48 ms).
- **Fair evaluation** — all baselines share the same held-out window
  (days 22–30); teacher-replay ground truth is the majority of three T=0
  repeats; statistics include Friedman + Nemenyi tests and bootstrap 95% CIs;
  19,234 real cloud LLM calls are logged in `poc/output/`.

## Datasets

Nine datasets, each repeated with four fixed seeds (42, 123, 999, 777),
producing 36 held-out blocks:

| Dataset | Domain | Notes |
|---|---|---|
| 4 synthetic streams (per seed) | controlled | temperature/humidity/light/motion, 30-day sinusoidal structure |
| CASAS Aruba-1 (STRANDS redistribution) | smart home | real activity/location annotations; numeric sensors synthetically completed |
| UCI Occupancy Detection | office | physical temperature/humidity/light/CO2/occupancy, CC BY 4.0 |
| SML2010 | domotic house | indoor climate, ~4,137 readings at 15-minute intervals |
| Steel Industry Energy Consumption | industrial | energy usage, reactive power, power factor, CO2 |
| Air Quality | urban | CO, NOx, NO2, temperature, humidity; missing entries as absent fields |

Derived snapshots used by the experiments are included under `poc/data/`.
Original dataset files retain their own licenses (see `poc/data/` provenance
notes and the paper's data-availability section).

## Repository layout

- `poc/` — Python pipeline: distillation, baselines, experiments, teacher
  replay, statistics, and all JSON results under `poc/output/`.
- `main/` — ESP32-S3 firmware (ESP-IDF v5.2.6): rule engine, lifecycle state
  machine, actuator control, trace logger, serial CLI, WebSocket dashboard.
- `figures/` — publication-quality figure scripts (`gen_all.py`) and outputs.
- `paper/` — manuscript (`main.tex`, `draft.md`, `main.pdf`) plus provenance
  and citation-audit records.
- `docs/` — architecture and setup guides.
- `spiffs_data/` — preloaded firmware rule sets.

## Reproducing the experiments

Python 3.11 with `pip install -r requirements.txt`:

```bash
python poc/run_4x.py              # online distillation experiments (real LLM API)
python poc/run_full_analysis.py   # held-out baselines + Friedman/Nemenyi
python poc/teacher_replay.py      # precision / recall / agreement vs teacher
python poc/cross_llm_experiment.py  # DeepSeek vs Qwen cross-model agreement
python poc/merge_statistics.py    # aggregate statistics
python figures/gen_all.py         # regenerate all figures
```

Set `DEEPSEEK_API_KEY` (and optionally `DASHSCOPE_API_KEY` for Qwen) before
running online experiments. Every result JSON under `poc/output/` is the exact
artifact cited by the paper.

## Firmware (ESP32-S3)

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p COMx flash monitor
```

Hardware target: ESP32-S3-DevKitC-1 (N16R8, 16 MB flash / 8 MB PSRAM). Sensors
are injected over USB serial (trace-driven hardware-in-the-loop); no external
sensor board is required. See `docs/setup_guide.md` and `docs/ARCHITECTURE.md`.

## Citation

If you use this work, please cite the companion paper (full reference in
`paper/main.tex`).

## License

Code in this repository is released under the MIT License. The manuscript and
figures are the author's own work. Dataset files retain their original
licenses; see the paper's data-availability statement for details.

"""
DistillToMCU — UCI 空气质量实验 (UCI 360)
==========================================
空气质量/通风控制：LLM 根据污染物/温湿度决定是否通风、净化、开窗。

Usage:
    python experiment_airquality.py --seed 42 --days 30
"""

import json
import random
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from trace_store import TraceStore
from rule_engine import RuleEngine
from distiller import Distiller
import llm_client


SYSTEM_PROMPT = """You are an air-quality controller for an urban monitoring station.
You receive pollutant and climate readings. Decide if any device action is needed.

Available devices:
- fan: ventilation fan, speed (1-3)
- led: air purifier indicator, brightness (0-100)
- curtain: window, position (0-100, 0=closed, 100=open)

Rules:
- If CO concentration is high, turn on the ventilation fan
- If NO2 is high, enable the air purifier (led)
- If temperature is high, open the window (curtain)
- If no action is needed, respond with a brief status ok
- Use the available tools when device control is needed. Respond in Chinese."""


class AirQualityExecutor:
    FIELDS = ["co", "nox", "no2", "temperature", "humidity"]
    UNITS = {"co": "mg/m3", "nox": "ug/m3", "no2": "ug/m3",
             "temperature": "C", "humidity": "%"}

    def __init__(self, rule_engine, trace_store):
        self.engine = rule_engine
        self.traces = trace_store
        self.m = {'total': 0, 'local': 0, 'cloud': 0,
                  'local_lat': 0, 'cloud_lat': 0, 'cost': 0}

    def handle(self, sensors):
        self.m['total'] += 1
        matches = self.engine.match(sensors)
        rule = self.engine.resolve_conflict(matches, {})
        if rule:
            lat = random.randint(3, 15)
            self.m['local'] += 1
            self.m['local_lat'] += lat
            self.engine.update_on_execution(rule.id, 'accepted')
            dev = rule.action['device']
            cmd = rule.action['command']
            args_json = json.dumps(dict(command=cmd, **rule.action.get('params', {})))
            self.traces.add(
                sensors, '',
                llm_response={
                    'reasoning': f'Local rule: {rule.id}',
                    'tool_calls': [{'id': 'x', 'function': {
                        'name': f'{dev}_control', 'arguments': args_json}}],
                    'model': 'local', 'latency_ms': lat},
                execution_mode='local', rule_id=rule.id)
            return

        sensor_lines = []
        for k in self.FIELDS:
            v = sensors.get(k)
            if v is not None:
                unit = self.UNITS.get(k, "")
                sensor_lines.append(f"  {k}: {v}{unit}")
        query = "Current air readings:\n" + "\n".join(sensor_lines) + \
                "\n\nDecide what action to take. Use tools if needed."

        t0 = time.time()
        resp = llm_client.cloud_agent_think(
            system_prompt=SYSTEM_PROMPT,
            user_input=query,
            sensors=sensors)
        lat = resp.get('latency_ms', 2000)
        self.m['cloud'] += 1
        self.m['cloud_lat'] += lat
        self.m['cost'] += 0.0007
        self.traces.add(
            sensors, query,
            llm_response={
                'reasoning': resp.get('content', ''),
                'tool_calls': resp.get('tool_calls') or [],
                'model': resp.get('model', 'deepseek'),
                'latency_ms': lat},
            execution_mode='cloud', rule_id=None)

    def summary(self):
        m = self.m
        return {
            'ar': round(m['local'] / max(1, m['total']) * 100, 1),
            'total': m['total'], 'local': m['local'], 'cloud': m['cloud'],
            'loc_ms': round(m['local_lat'] / max(1, m['local']), 1),
            'cld_ms': round(m['cloud_lat'] / max(1, m['cloud']), 1),
            'cost': round(m['cost'], 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--backend', default=None)
    args = parser.parse_args()

    if args.backend:
        llm_client._ACTIVE_BACKEND = args.backend
    random.seed(args.seed)

    with open(os.path.join(os.path.dirname(__file__), 'data', 'airquality',
                           'snapshots.json')) as f:
        raw = json.load(f)
    per_day = max(1, len(raw) // args.days)
    by_day = [raw[i * per_day:(i + 1) * per_day] for i in range(args.days)]

    odir = args.output_dir or os.path.join(
        os.path.dirname(__file__), 'output', f'airquality_seed{args.seed}')
    os.makedirs(odir, exist_ok=True)
    ts = TraceStore(output_dir=odir)
    re = RuleEngine()
    di = Distiller(re, llm_client=llm_client)
    ex = AirQualityExecutor(re, ts)

    print('AIR QUALITY urban monitoring -> LLM')
    print(f'{args.days}d, {len(raw)} snapshots, {len(raw)//len(by_day[0])}/day')

    daily_m = []
    for d in range(args.days):
        dt0 = time.time()
        for s in by_day[d]:
            ex.handle(s)
        re.update_all_freshness()
        re.gc()
        nr, _ = di.distill(ts.traces)
        sm = ex.summary()
        daily_m.append({
            'day': d + 1, 'autonomy_rate': sm['ar'],
            'cloud_calls': sm['cloud'], 'local_calls': sm['local'],
            'total': sm['total'],
            'active_rules': re.stats().get('active_count', 0),
            'total_rules': re.stats()['total'], 'new_rules_today': nr,
            'avg_local_lat_ms': sm['loc_ms'], 'avg_cloud_lat_ms': sm['cld_ms']})
        tag = f'[+{nr}r]' if nr else ''
        print(f'  Day {d+1:2d}  {tag:6s} AR={sm["ar"]:5.1f}%  '
              f'({time.time()-dt0:.0f}s)')

    fin = ex.summary()
    st = re.stats()
    print(f'\n  FINAL: AR={fin["ar"]:.1f}% | {fin["total"]} int | '
          f'Local:{fin["local"]} Cloud:{fin["cloud"]} | '
          f'Rules:{st["total"]} active:{st.get("active_count",0)}')

    with open(os.path.join(odir, 'metrics.jsonl'), 'w') as f:
        for m in daily_m:
            f.write(json.dumps(m) + '\n')
    re.save_snapshot(os.path.join(odir, 'rules_snapshot.json'))
    with open(os.path.join(odir, 'traces.jsonl'), 'w', encoding='utf-8') as f:
        for t in ts.traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

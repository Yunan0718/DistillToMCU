"""
DistillToMCU — UCI Occupancy Detection 真实传感器实验 (v3: 零硬编码)
====================================================================
传感器读数直接发给 LLM，LLM 自己决定是否控制设备。
没有住户语音生成器、没有 if-else 翻译、没有阈值写死。

Usage:
    python experiment_uci.py --seed 42 --days 30
"""
import json, random, os, sys, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from trace_store import TraceStore
from rule_engine import RuleEngine
from distiller import Distiller
import llm_client

SYSTEM_PROMPT = """You are a smart home controller.
You receive sensor readings from a room. Decide if any device action is needed.

Available devices:
- led: brightness (0-100)
- fan: speed (1-3)
- curtain: position (0-100, 0=closed, 100=open)

Rules:
- If it's dark and someone is in the room, turn on the light
- If CO2 is high, turn on the fan for ventilation
- If it's too bright, close the curtain
- If the temperature is rising rapidly, turn on the fan
- If no action is needed, respond with a brief status ok
- Use the available tools when device control is needed. Respond in Chinese."""


class UCIExecutor:
    def __init__(self, rule_engine, trace_store):
        self.engine = rule_engine
        self.traces = trace_store
        self.m = {'total': 0, 'local': 0, 'cloud': 0,
                  'local_lat': 0, 'cloud_lat': 0, 'cost': 0}

    def handle(self, sensors):
        self.m['total'] += 1

        # 规则匹配 → 本地执行
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

        # 未命中 → 传感器直发 LLM，不加任何翻译
        sensor_lines = []
        for k in ['temperature', 'humidity', 'light', 'co2', 'motion']:
            v = sensors.get(k)
            if v is not None:
                unit = 'C' if k == 'temperature' else '%' if k == 'humidity' else 'lux' if k == 'light' else 'ppm' if k == 'co2' else ''
                sensor_lines.append(f"  {k}: {v}{unit}")
        query = "Current sensor readings:\n" + "\n".join(sensor_lines) + \
                "\n\nDecide what action to take. Use tools if needed."

        t0 = time.time()
        resp = llm_client.cloud_agent_think(
            system_prompt=SYSTEM_PROMPT,
            user_input=query,
            sensors=sensors)
        lat = resp.get('latency_ms', 2000)
        self.m['cloud'] += 1
        self.m['cloud_lat'] += lat
        # v6 修复：0.0007 已是人民币估算，不再乘 7.2
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
        t = max(1, m['total'])
        lc = max(1, m['local'])
        cc = max(1, m['cloud'])
        return {
            'ar': round(m['local'] / t * 100, 1),
            'total': t, 'local': m['local'], 'cloud': m['cloud'],
            'loc_ms': round(m['local_lat'] / lc, 1),
            'cld_ms': round(m['cloud_lat'] / cc, 1),
            'cost': round(m['cost'], 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--backend', default=None,
                        help='LLM backend (default deepseek-v4-flash)')
    args = parser.parse_args()

    if args.backend:
        llm_client._ACTIVE_BACKEND = args.backend
    random.seed(args.seed)

    # 加载 UCI enriched 数据
    with open(os.path.join(os.path.dirname(__file__), 'data', 'uci',
                          'snapshots_enriched.json')) as f:
        raw = json.load(f)

    per_day = max(1, len(raw) // args.days)
    by_day = [raw[i * per_day:(i + 1) * per_day] for i in range(args.days)]

    odir = args.output_dir or os.path.join(os.path.dirname(__file__), 'output',
                                           f'uci_v3_seed{args.seed}')
    os.makedirs(odir, exist_ok=True)
    ts = TraceStore(output_dir=odir)
    re = RuleEngine()
    di = Distiller(re, llm_client=llm_client)
    ex = UCIExecutor(re, ts)

    print(f'UCI REAL sensors -> LLM (no translation layer)')
    print(f'{args.days}d, {len(raw)} snapshots, {len(raw)//len(by_day[0])}/day')
    print(f'Sensors: Temperature + Humidity + Light + CO2 + Motion')
    print(f'LLM decides autonomously — zero hardcoded thresholds\n')

    daily_m = []
    t0 = time.time()
    for d in range(args.days):
        dt0 = time.time()
        for s in by_day[d]:
            ex.handle(s)
        re.update_all_freshness()
        re.gc()
        nr, _ = di.distill(ts.traces)
        sm = ex.summary()
        ar = sm['ar']
        daily_m.append({
            'day': d + 1, 'autonomy_rate': ar,
            'cloud_calls': sm['cloud'], 'local_calls': sm['local'],
            'total': sm['total'],
            'active_rules': re.stats().get('active_count', 0),
            'total_rules': re.stats()['total'], 'new_rules_today': nr,
            'avg_local_lat_ms': sm['loc_ms'], 'avg_cloud_lat_ms': sm['cld_ms']})
        tag = f'[+{nr}r]' if nr else ''
        dt = time.time() - dt0
        print(f'  Day {d+1:2d}  {tag:6s} AR={ar:5.1f}%  ({dt:.0f}s)')

    et = time.time() - t0
    fin = ex.summary()
    st = re.stats()
    print(f'\n  FINAL: AR={fin["ar"]:.1f}% | {fin["total"]} int | '
          f'Local:{fin["local"]} Cloud:{fin["cloud"]} | '
          f'Rules:{st["total"]} active:{st.get("active_count",0)} | '
          f'Cost:~{fin["cost"]:.2f} CNY | Time:{et:.0f}s')

    with open(os.path.join(odir, 'metrics.jsonl'), 'w') as f:
        for m in daily_m:
            f.write(json.dumps(m) + '\n')
    re.save_snapshot(os.path.join(odir, 'rules_snapshot.json'))
    with open(os.path.join(odir, 'traces.jsonl'), 'w', encoding='utf-8') as f:
        for t in ts.traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

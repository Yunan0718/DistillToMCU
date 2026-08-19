"""
DistillToMCU — STRANDS Aruba-1 真实数据实验
============================================
基于真实 CASAS Aruba-1 活动+位置标注的 LLM 行为蒸馏实验。

Usage:
    python experiment_strands.py --seed 42 --days 30
"""

import json, random, os, sys, time, argparse
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from trace_store import TraceStore
from rule_engine import RuleEngine
from distiller import Distiller
import llm_client
from config import SEED


class StrandsExecutor:
    def __init__(self, re, ts, llm):
        self.engine = re; self.traces = ts; self.llm = llm
        self.actuator_states = {}
        self.m = {'total':0,'local':0,'cloud':0,'local_lat_sum':0,'cloud_lat_sum':0,'cost':0,'tokens':0}

    def handle(self, interaction):
        self.m['total'] += 1
        sensors = interaction['sensors']
        user_input = interaction.get('user_input', '')
        activity = interaction.get('activity', '')

        if not user_input:
            act_map = {
                'Sleeping':'准备睡觉了','Meal_Preparation':'要做饭了','Eating':'吃饭了',
                'Relax':'想放松一下','Housekeeping':'该打扫了','Wash_Dishes':'该洗碗了',
                'Work':'开始工作','Enter_Home':'我回来了','Leave_Home':'出门了',
            }
            user_input = act_map.get(activity, '')
            if not user_input:
                t = sensors.get('temperature',22)
                l = sensors.get('light',500)
                if t > 28: user_input = '太热了'
                elif l < 30: user_input = '好暗啊'
                else: user_input = '一切正常'

        matches = self.engine.match(sensors)
        rule = self.engine.resolve_conflict(matches, self.actuator_states)

        if rule:
            lat = random.randint(3, 15)
            self.m['local'] += 1; self.m['local_lat_sum'] += lat
            action = rule.action
            self.engine.update_on_execution(rule.id, 'accepted')
            self.traces.add(sensors, user_input,
                llm_response={
                    'reasoning': f'Local: {rule.id}',
                    'tool_calls': [{'id': f'loc_{rule.id}', 'type': 'function',
                        'function': {'name': f'{action["device"]}_control',
                            'arguments': json.dumps(dict(command=action['command'], **action.get('params', {})))}}],
                    'model': 'local', 'latency_ms': lat},
                execution_mode='local', rule_id=rule.id)
            return {'mode': 'local', 'rule_id': rule.id}

        t0 = time.time()
        response = self.llm.cloud_agent_think(
            system_prompt='Smart home controller. Use tools for: led(brightness), fan(speed 1-3), curtain(position 0-100).',
            user_input=user_input, sensors=sensors)
        cloud_lat = response.get('latency_ms', 2000)
        self.m['cloud'] += 1; self.m['cloud_lat_sum'] += cloud_lat
        # v6 修复：0.0007 已是人民币估算，不再乘 7.2
        self.m['cost'] += 0.0007; self.m['tokens'] += 600

        tc = response.get('tool_calls') or []
        self.traces.add(sensors, user_input,
            llm_response={'reasoning': response.get('content',''), 'tool_calls': tc,
                'model': response.get('model','deepseek'), 'latency_ms': cloud_lat},
            execution_mode='cloud', rule_id=None)
        return {'mode': 'cloud'}

    def get_summary(self):
        m = self.m; t = max(1,m['total']); lc = max(1,m['local']); cc = max(1,m['cloud'])
        return {
            'autonomy_rate': round(m['local']/t*100,1),
            'cloud_call_reduction': round((1-m['cloud']/t)*100,1),
            'avg_local_lat_ms': round(m['local_lat_sum']/lc,1),
            'avg_cloud_lat_ms': round(m['cloud_lat_sum']/cc,1),
            'total': t, 'local': m['local'], 'cloud': m['cloud'],
            'est_cost_cny': round(m['cost'], 2),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--backend', default=None,
                        help='LLM backend (default deepseek-v4-flash)')
    args = parser.parse_args()

    if args.backend:
        import llm_client
        llm_client._ACTIVE_BACKEND = args.backend
    random.seed(args.seed)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'casas', 'aruba_snapshots.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        snapshots = json.load(f)

    by_day = defaultdict(list)
    for s in snapshots:
        by_day[s['day']].append(s)
    days = sorted(by_day.keys())[:args.days]

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), 'output', f'strands_seed{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    trace_store = TraceStore(output_dir=output_dir)
    rule_engine = RuleEngine()
    distiller = Distiller(rule_engine, llm_client=llm_client)
    executor = StrandsExecutor(rule_engine, trace_store, llm_client)

    print('=' * 60)
    print(f'  STRANDS Aruba-1 x Real DeepSeek (seed={args.seed}, {args.days}d)')
    print(f'  Data: 112 days, 12 activities, 10 rooms, 161K min')
    print(f'  Snapshots: {len(snapshots)} (30-min intervals)')
    print('=' * 60)
    print()

    daily = []
    exp_t0 = time.time()

    for i, day in enumerate(days):
        t0 = time.time()
        snaps = by_day[day]
        for s in snaps:
            executor.handle(s)

        rule_engine.update_all_freshness()
        rule_engine.gc()
        new_rules, _ = distiller.distill(trace_store.traces)

        s = executor.get_summary()
        stats = rule_engine.stats()
        daily.append({
            'day': i+1, 'autonomy_rate': s['autonomy_rate'],
            'cloud_calls': s['cloud'], 'local_calls': s['local'],
            'total_interactions': s['total'],
            'active_rules': stats.get('active_count',0),
            'total_rules': stats['total'], 'new_rules_today': new_rules,
            'avg_local_lat_ms': s['avg_local_lat_ms'],
            'avg_cloud_lat_ms': s['avg_cloud_lat_ms'],
        })

        dt = time.time() - t0
        tag = f'[+{new_rules}r]' if new_rules else ''
        print(f'  Day {i+1:2d}/{args.days}  {tag:8s} AR={s["autonomy_rate"]:5.1f}%  ({dt:.0f}s)')

    final = executor.get_summary()
    final_stats = rule_engine.stats()
    et = time.time() - exp_t0

    print()
    print('=' * 60)
    print('  STRANDS Aruba-1 FINAL')
    print('=' * 60)
    print(f'  Autonomy Rate:        {final["autonomy_rate"]:.1f}%')
    print(f'  Cloud Call Reduction: {final["cloud_call_reduction"]:.1f}%')
    print(f'  Total Interactions:   {final["total"]}')
    print(f'  Local / Cloud:        {final["local"]} / {final["cloud"]}')
    print(f'  Rules:                {final_stats["total"]} total, {final_stats.get("active_count",0)} active')
    print(f'  By State:             {final_stats["by_state"]}')
    print(f'  Time:                 {et:.0f}s ({et/60:.1f}m)')
    print(f'  API Cost:             ~{final["est_cost_cny"]:.2f} CNY')

    with open(os.path.join(output_dir, 'metrics.jsonl'), 'w', encoding='utf-8') as f:
        for m in daily: f.write(json.dumps(m) + '\n')
    rule_engine.save_snapshot(os.path.join(output_dir, 'rules_snapshot.json'))
    with open(os.path.join(output_dir, 'traces.jsonl'), 'w', encoding='utf-8') as f:
        for t in trace_store.traces: f.write(json.dumps(t, ensure_ascii=False) + '\n')

    print(f'\n  Output -> {output_dir}/')
    print(f'  Files: metrics.jsonl | rules_snapshot.json | traces.jsonl')


if __name__ == '__main__':
    main()

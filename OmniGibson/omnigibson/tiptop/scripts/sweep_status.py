#!/usr/bin/env python3
"""Status of the background lanes, as JSON on stdout.

Reads only files: the queue files, their reports and each run's summary.json. Runs in the system python (no
numpy), so it is safe to call while the lanes are working. Written for the eval sweep tracker; `--pretty` prints
a terminal table instead.

    python3 scripts/sweep_status.py [--pretty]
"""
import glob, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
JOB = re.compile(r'^=== JOB (\S+) \[([^\]]*)\] -> (\S+)')
DONE = re.compile(r'^=== DONE (\S+) exit (\d+)')


def queue_jobs(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, errors='ignore'):
        line = line.strip()
        if line and not line.startswith('#'):
            p = line.split()
            out.append((p[0], p[1] if len(p) > 1 else '0'))
    return out


def lane(name, queue, report):
    jobs = queue_jobs(queue)
    started, done, current = [], [], None
    if os.path.exists(report):
        for line in open(report, errors='ignore'):
            m = JOB.match(line)
            if m:
                started.append((m.group(1), m.group(2), m.group(3)))
                current = m.group(1)
            d = DONE.match(line)
            if d:
                done.append((d.group(1), int(d.group(2))))
                current = None
    return {
        'lane': name, 'queued': len(jobs), 'started': len(started), 'done': len(done),
        'current': current, 'dirs': [s[2] for s in started],
        'failed': sum(1 for _, c in done if c != 0),
    }


def scores(dirs):
    out = []
    for d in dirs:
        p = os.path.join(ROOT, d, 'summary.json')
        if not os.path.exists(p):
            continue
        try:
            s = json.load(open(p))
        except Exception:
            continue
        out.append({'task': s.get('task'), 'q': s.get('mean_q_score'),
                    'successes': s.get('successes'), 'knowledge': s.get('knowledge'),
                    'instances': s.get('instances')})
    return out


def main():
    os.chdir(ROOT)
    ev = [lane(f'ev{i}', f'runs/queue_ev{i}.txt', f'runs/queue_report_ev{i}.txt') for i in range(1, 10)]
    so = [lane(f'so{c}', f'runs/queue_so{c}.txt', f'runs/queue_report_so{c}.txt') for c in 'ABCD']
    ev = [l for l in ev if l['queued']]
    alldirs = [d for l in ev for d in l['dirs']]
    done_scores = scores(alldirs)
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    # standoff samples, the stance experiment's actual output
    n_standoff = 0
    for f in glob.glob('runs/queue_logs/*.log'):
        try:
            n_standoff += sum(1 for line in open(f, errors='ignore') if 'standoffs [' in line)
        except Exception:
            pass
    per_task = {}
    for s in done_scores:
        if s['task'] and s['q'] is not None:
            per_task.setdefault(s['task'], []).append(s['q'])
    out = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'head': head,
        'eval': {'lanes': ev,
                 'episodes_total': sum(l['queued'] for l in ev),
                 'episodes_done': sum(l['done'] for l in ev),
                 'failed': sum(l['failed'] for l in ev),
                 'per_task': {k: {'n': len(v), 'mean_q': sum(v) / len(v)} for k, v in sorted(per_task.items())}},
        'stance': {'lanes': so,
                   'jobs_total': sum(l['queued'] for l in so),
                   'jobs_done': sum(l['done'] for l in so),
                   'standoff_samples': n_standoff},
    }
    if '--pretty' in sys.argv:
        e = out['eval']
        print(f"eval   {e['episodes_done']}/{e['episodes_total']} episodes, {e['failed']} failed")
        for l in e['lanes']:
            print(f"  {l['lane']}: {l['done']}/{l['queued']}  now={l['current'] or '-'}")
        s = out['stance']
        print(f"stance {s['jobs_done']}/{s['jobs_total']} jobs, {s['standoff_samples']} standoff samples")
        for l in s['lanes']:
            print(f"  {l['lane']}: {l['done']}/{l['queued']}  now={l['current'] or '-'}")
        print(f"\ntasks with a score: {len(e['per_task'])}")
    else:
        json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

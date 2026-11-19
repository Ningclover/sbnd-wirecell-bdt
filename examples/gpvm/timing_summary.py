"""Time record of a grid table build: from the queue state changes (job_states.tsv, written by monitor_jobs.sh) and the
per-job timing files (timing_*.json, written by table_job.sh).
usage: python3 timing_summary.py <monitor folder> <folder with the timing_*.json> <file whose first word is the submit epoch>"""
import sys, glob, json, statistics as st, collections, time
mon, tdir, subf = sys.argv[1:4]
fmt = lambda s: f'{s/60:.1f} min' if s is not None else 'n/a'
def stats(v): return f'min {fmt(min(v))}, median {fmt(st.median(v))}, max {fmt(max(v))}, sum {sum(v)/3600:.2f} h' if v else 'none'
try: sub = int(open(subf).read().split()[0])
except Exception: sub = None
ev = collections.defaultdict(dict)
import os
for line in (open(f'{mon}/job_states.tsv') if os.path.exists(f'{mon}/job_states.tsv') else []):
    t, _, jid, state = line.rstrip('\n').split('\t'); ev[jid].setdefault(state, int(t))
wait = [e['R'] - sub for e in ev.values() if 'R' in e and sub]; run = [e['GONE'] - e['R'] for e in ev.values() if 'R' in e and 'GONE' in e]
never = [j for j, e in ev.items() if 'R' not in e]; held = [j for j, e in ev.items() if 'H' in e]
print(f'submitted: {time.strftime("%F %T", time.localtime(sub)) if sub else "unknown"}')
print(f'jobs seen in the queue: {len(ev)}; never seen running: {len(never)}; held at some point: {len(held)}')
print(f'wait from submission to first seen running (poll resolution): {stats(wait)}')
print(f'time seen running in the queue (poll resolution): {stats(run)}')
if ev and sub:
    last = max(max(e.values()) for e in ev.values()); print(f'submission to last job gone: {fmt(last - sub)}')
T = [json.load(open(f)) for f in sorted(glob.glob(f'{tdir}/timing_*.json'))]
print(f'per-job timing files: {len(T)}')
for s in sorted({t['sample'] for t in T}):
    x = [t for t in T if t['sample'] == s]
    print(f'== {s}: {len(x)} jobs, {sum(t["files_copied"] for t in x)} files, {sum(t["bytes_copied"] for t in x)/1e9:.2f} GB copied in')
    print(f'   copy-in per job: {stats([t["copy_in_s"] for t in x])}'); print(f'   build per job:   {stats([t["build_s"] for t in x])}')
    print(f'   per file: copy-in {sum(t["copy_in_s"] for t in x)/max(sum(t["files_copied"] for t in x),1):.2f} s, build {sum(t["build_s"] for t in x)/max(sum(t["files_copied"] for t in x),1):.2f} s')
    for cpu in sorted({t['cpu_model'] for t in x}):
        y = [t for t in x if t['cpu_model'] == cpu]
        print(f'   {cpu}: {len(y)} jobs, build {sum(t["build_s"] for t in y)/max(sum(t["files_copied"] for t in y),1):.2f} s per file')
if T: print(f'all jobs: copy-in {sum(t["copy_in_s"] for t in T)/3600:.2f} h, build {sum(t["build_s"] for t in T)/3600:.2f} h')

"""Extract per-command durations from mini-SWE-agent trajectories (assistant.timestamp -> tool.timestamp)."""
import json, os, sys, glob
rows=[]
for p in sorted(glob.glob(sys.argv[1]+'/*.traj.json')):
    d=json.load(open(p)); iid=d.get('instance_id') or os.path.basename(p).split('.traj')[0]
    m=d['messages']; step=0
    for i,x in enumerate(m):
        if x.get('role')!='assistant': continue
        ex=x.get('extra') or {}; t0=ex.get('timestamp'); acts=ex.get('actions') or []
        tools=[]
        j=i+1
        while j<len(m) and m[j].get('role') in ('tool','user') and isinstance(m[j].get('extra'),dict) and 'returncode' in m[j]['extra']:
            tools.append(m[j]); j+=1
        if not tools or t0 is None: continue
        prev=t0
        for k,tm in enumerate(tools):
            t1=tm['extra'].get('timestamp')
            cmd=acts[k].get('command') if k<len(acts) and isinstance(acts[k],dict) else None
            if t1 is None: continue
            rows.append(dict(iid=iid, step=step, cmd=cmd, dur_s=t1-prev, rc=tm['extra'].get('returncode'),
                             out_len=len(tm['extra'].get('raw_output') or ''), t_start=prev, t_end=t1, n_par=len(tools)))
            prev=t1
        step+=1
with open(sys.argv[2],'w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
print(len(rows),'tool calls from',len({r['iid'] for r in rows}),'trajectories')

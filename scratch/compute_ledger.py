"""Compute ledger for this project from SLURM accounting: per day and per job family, with
GPU-hours, CPU-core-hours and dollars at the published PARCC rates (unsubsidised / subsidised)."""
import re, subprocess, pandas as pd
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

G=_RESOLVED_G
PAT=r'^(geom_|clean_shift|hida_bank|l1norm|hillclimb|cosshift|cos_shift|imgref|img_ref|pose_feats|res_trend|shift_sweep|viewdepth|enc_extract|evaltest|chair_|airplane_|table_)'
out=subprocess.run(['sacct','-u','bonnen','-S','2026-09-08','-E','now','-X','--format=JobID,JobName%30,Partition%13,Start,Elapsed,AllocCPUS,AllocTRES%40,State','-P'],capture_output=True,text=True).stdout
rows=[]
for line in out.splitlines()[1:]:
    f=line.split('|')
    if len(f)<8 or f[3]=='None': continue
    jid,name,part,start,el,cpus,tres,state=f[:8]
    if not re.match(PAT,name): continue
    dpart,_,t=el.rpartition('-'); h,m,s=[int(x) for x in t.split(':')]; hrs=(int(dpart) if dpart else 0)*24+h+m/60+s/3600
    gpu=1 if 'gres/gpu=1' in tres else 0
    rows.append(dict(day=start[:10],start=start,job=name,part=part,hours=round(hrs,3),cpus=int(cpus or 0),gpu=gpu,state=state))
d=pd.DataFrame(rows); d['gpu_h']=d.gpu*d.hours; d['cpu_core_h']=d.cpus*d.hours*(1-d.gpu)
ru={'dgx-b200':4.51,'b200-mig45':1.13}; rs={'dgx-b200':1.00,'b200-mig45':0.25}
d['usd_unsub']=[r.gpu_h*ru.get(r.part,0)+r.cpu_core_h*0.028 for r in d.itertuples()]
d['usd_sub']=[r.gpu_h*rs.get(r.part,0)+r.cpu_core_h*0.010 for r in d.itertuples()]
d['fam']=d.job.str.replace(r'_(g\d|random|k\d+|\d+).*$','',regex=True)
pd.set_option('display.width',220)
g=d.groupby('day').agg(jobs=('job','size'),gpu_h=('gpu_h','sum'),cpu_core_h=('cpu_core_h','sum'),usd_unsub=('usd_unsub','sum'),usd_sub=('usd_sub','sum')).round(2)
print(g); print('\nTOTAL',g.sum().round(2).to_dict())
print('\nby family:'); print(d.groupby('fam').agg(n=('job','size'),gpu_h=('gpu_h','sum'),cpu_core_h=('cpu_core_h','sum'),usd_unsub=('usd_unsub','sum'),usd_sub=('usd_sub','sum')).round(2).sort_values('usd_unsub',ascending=False))
d.sort_values('start').to_csv(f'{G}/background/compute_ledger.csv',index=False); print('\nledger ->',f'{G}/background/compute_ledger.csv')

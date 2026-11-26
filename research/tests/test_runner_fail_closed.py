#!/usr/bin/env python3
"""Exercise the actual patched queue with CPU fake processes; no network/GPU/files outside tempdir.
Usage: python test_runner_fail_closed.py --runner /path/to/patched/colab/runner.py
"""
import argparse, importlib.util, json, tempfile
from pathlib import Path
from types import SimpleNamespace

p=argparse.ArgumentParser();p.add_argument('--runner',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('runner_test',a.runner)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Clock:
    t=0
    def time(self): return self.t
    def sleep(self,n): self.t+=max(n,0.001)
class Proc:
    def __init__(self,rc): self.rc=rc
    def poll(self): return self.rc
    def terminate(self): self.rc=-15
    def kill(self): self.rc=-9
class Log:
    def close(self): pass
class Saver:
    def save(self,*args,**kwargs): pass


def run_case(rc=0, step=2, produce=True, require_missing=False):
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);m.OUT=base/'out';m.OUT.mkdir();m.CODE=base;m.STOP=base/'STOP'
        clock=Clock();m.time=clock;m._log=lambda msg:None;m._gpu_used_gb=lambda:(None,None)
        launched=[]
        def launch(plan,r,t0,budget):
            name=r['name'];launched.append(name);d=m.OUT/name;d.mkdir()
            (d/'stdout.log').write_text('fake run\n')
            if name=='train':
                if produce:(d/'memory.pt').write_bytes(b'fake checkpoint')
                (d/'progress.jsonl').write_text(json.dumps({'event':'done','step':step})+'\n')
            return {'name':name,'proc':Proc(rc if name=='train' else 0),'log':Log(),'dir':d,
              't':clock.time(),'note':clock.time(),'wait':False,'gpu_gb':0,'max_min':1,
              'output_files':r.get('output_files',[]),'min_completed_step':r.get('min_completed_step')}
        m._launch=launch
        required=(m.OUT/'train'/'missing.pt') if require_missing else (m.OUT/'train'/'memory.pt')
        plan={'name':'cpu-test','max_parallel':1,'stagger_sec':0,'poll_sec':0,
          'runs':[{'name':'train','output_files':['memory.pt'],'min_completed_step':2},
                  {'name':'eval','after':['train'],'required_files':[str(required)]}]}
        status={'runs':{}}
        m._run_queue(plan,Saver(),status,lambda:None,0,10)
        return launched,status['runs']

checks={}
for label,kwargs,want in [
 ('successful_dependency',{},['train','eval']),
 ('failed_dependency',{'rc':1},['train']),
 ('exit0_incomplete_step',{'step':1},['train']),
 ('exit0_missing_output',{'produce':False},['train']),
 ('missing_required_input',{'require_missing':True},['train'])]:
    launched,status=run_case(**kwargs)
    assert launched==want,(label,launched,status)
    checks[label]=status
for label,runs in [
 ('unknown_dependency',[{'name':'eval','after':['absent']}]),
 ('cycle',[{'name':'x','after':['y']},{'name':'y','after':['x']}]),
 ('duplicate_names',[{'name':'x'},{'name':'x'}])]:
    try:m._validate_dependencies(runs)
    except ValueError as e:checks[label]=str(e)
    else:raise AssertionError(label)
print(json.dumps({'status':'passed','checks':checks,'limits':'CPU fake-process queue only; no GPU, Modal, saver backend or model integration.'},ensure_ascii=False,indent=2))

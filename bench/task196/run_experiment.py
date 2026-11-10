"""Prospective one-GPU experiment, no network or external data/model access."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import csv,datetime,hashlib,json,math,random,statistics,time
from pathlib import Path
import numpy as np
import torch,triton
from bias_gelu_tanh import run_variant

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'PROTOCOL.json').read_text())
OUT=ROOT/'output';OUT.mkdir(exist_ok=False)
T0=time.monotonic()
def deadline():
    if time.monotonic()-T0>P['internal_wall_seconds']:raise TimeoutError('Preregistered experiment wall exceeded')
def put(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2))
def emit(obj):print(json.dumps(obj),flush=True)
def ref(x,bias):return torch.nn.functional.gelu((x.float()+bias.float()).to(x.dtype),approximate='tanh')
def make(shape,dtype,seed,pattern='normal'):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    x=torch.randn(tuple(shape),generator=gen,dtype=torch.float32)
    b=torch.randn(shape[-1],generator=gen,dtype=torch.float32)
    if pattern=='zeros':x.zero_();b.zero_()
    elif pattern=='cancellation_rounding':
        x=torch.linspace(-8,8,x.numel()).reshape(shape);b=torch.linspace(1,-1,shape[-1])
    elif pattern=='large_finite':x=x.clamp(-2,2)*1000;b=b.clamp(-2,2)*1000
    elif pattern=='tails':x=torch.linspace(-12,12,x.numel()).reshape(shape);b.zero_()
    return x.to(device='cuda',dtype=dtype),b.to(device='cuda',dtype=dtype)

CHECKS=[];RAW=[]
def check_case(part,idx,shape,dtype_name,seed,pattern='normal'):
    deadline();dtype=getattr(torch,dtype_name);x,b=make(shape,dtype,seed,pattern)
    x0=x.clone();b0=b.clone();expected=ref(x,b)
    funcs={'P0_reference':lambda:ref(x,b)}
    for arm in P['arms'][1:]:funcs[arm]=lambda a=arm:run_variant(x,b,a)
    for arm,f in funcs.items():
        y=f();torch.cuda.synchronize()
        assert y.shape==x.shape and y.dtype==x.dtype and y.device==x.device
        assert y.untyped_storage().data_ptr() not in (x.untyped_storage().data_ptr(),b.untyped_storage().data_ptr())
        assert torch.equal(x,x0) and torch.equal(b,b0)
        torch.testing.assert_close(y,expected,**P['tolerances'][dtype_name])
        CHECKS.append({'part':part,'case':idx,'shape':shape,'dtype':dtype_name,'pattern':pattern,'arm':arm,'passed':True,'max_abs':float((y.float()-expected.float()).abs().max())})
    return x,b,funcs

def measure(part,shapes,seed):
    rng=random.Random(P['order_seed']+(0 if part=='development' else 1))
    for idx,shape in enumerate(shapes):
      for di,dtype_name in enumerate(P['dtypes']):
        x,b,funcs=check_case(part,idx,shape,dtype_name,seed+idx*101+di)
        for session in range(P['sessions']):
          for f in funcs.values():
            for _ in range(P['warmup']):f()
          torch.cuda.synchronize()
          for rnd in range(P['rounds']):
            deadline();order=list(P['arms']);rng.shuffle(order)
            for arm in order:
              s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
              s.record()
              for _ in range(P['calls']):funcs[arm]()
              e.record();e.synchronize()
              RAW.append({'part':part,'case':idx,'dtype':dtype_name,'shape':str(shape),'session':session,'round':rnd,'arm':arm,'us':s.elapsed_time(e)*1000/P['calls']})
        emit({'event':'case_complete','part':part,'case':idx,'dtype':dtype_name,'elapsed':time.monotonic()-T0})
        put('PROGRESS.json',{'part':part,'case':idx,'dtype':dtype_name,'checks':len(CHECKS),'raw_rows':len(RAW)})
        put('CHECKS.json',CHECKS)
        with (OUT/'raw.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(RAW[0]));writer.writeheader();writer.writerows(RAW)

def ratios(part,arm):
    cases=sorted({(r['case'],r['dtype']) for r in RAW if r['part']==part})
    vals=[]
    for case,dtype in cases:
        arms={a:{(r['session'],r['round']):r['us'] for r in RAW if r['part']==part and r['case']==case and r['dtype']==dtype and r['arm']==a} for a in [arm,'B0_row','P0_reference']}
        keys=set(arms[arm]);assert len(keys)==P['sessions']*P['rounds'] and all(set(v)==keys for v in arms.values())
        b0=statistics.median(arms['B0_row'][k]/arms[arm][k] for k in sorted(keys))
        p0=statistics.median(arms['P0_reference'][k]/arms[arm][k] for k in sorted(keys))
        vals.append({'case':case,'dtype':dtype,'speedup_vs_B0':b0,'speedup_vs_P0':p0,'median_us':statistics.median(arms[arm].values()),'paired_samples':len(keys)})
    return vals
def gm(vals):return math.exp(statistics.mean(math.log(x['speedup_vs_B0']) for x in vals))

def main():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==1
    torch.set_num_threads(2)
    put('STARTED.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'managed_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'triton':triton.__version__,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file()},'protocol':P})
    for pi,pattern in enumerate(P['adversarial_panels']):
      for di,dtype_name in enumerate(P['dtypes']):check_case('adversarial',pi,[17,264],dtype_name,P['development_seed']+10000+pi*101+di,pattern)
    measure('development',P['development_shapes'],P['development_seed'])
    scores={arm:gm(ratios('development',arm)) for arm in P['arms'][2:]}
    winner=max(scores,key=scores.get)
    if scores[winner]<P['selection']['minimum_speedup']:winner='B0_row'
    selection={'selected':winner,'development_geomean_vs_B0':scores,'rule':P['selection'],'before_confirmation':True}
    put('SELECTION_FROZEN.json',selection);emit({'event':'selection_frozen',**selection})
    frozen=hashlib.sha256((OUT/'SELECTION_FROZEN.json').read_bytes()).hexdigest()
    measure('confirmation',P['confirmation_shapes'],P['confirmation_seed'])
    assert hashlib.sha256((OUT/'SELECTION_FROZEN.json').read_bytes()).hexdigest()==frozen
    confirmation=ratios('confirmation',winner)
    clusters=sorted({x['case'] for x in confirmation})
    logs=np.array([statistics.mean(math.log(x['speedup_vs_B0']) for x in confirmation if x['case']==c) for c in clusters])
    rng=np.random.default_rng(P['bootstrap_seed']);reps=np.exp(logs[rng.integers(0,len(logs),size=(P['confirmation']['bootstrap_replicates'],len(logs)))].mean(1))
    ci=np.quantile(reps,[.025,.975]).tolist()
    result={'kind':'local3090_controlled_benchmark','selected':winner,'selection':selection,'selection_sha256':frozen,'confirmation_all_cases':confirmation,'confirmation_geomean_vs_B0':gm(confirmation),'shape_cluster_bootstrap_95CI':ci,'strict_lower_bound_gt1':ci[0]>1,'all_correctness_passed':all(x['passed'] for x in CHECKS),'checks':len(CHECKS),'raw_rows':len(RAW),'elapsed_seconds':time.monotonic()-T0,'official_score':None,'limits':['One NVIDIA3090, not seven official chips.','Authored cases, no official test suite available in inspected public tree.','Three sessions measure timing repeatability; shape-cluster bootstrap does not prove cross-device generalisation.','No confirmation reranking or adaptive thresholds.']}
    put('RESULT.json',result);emit(result)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        put('FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-T0})
        raise
